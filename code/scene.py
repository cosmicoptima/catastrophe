# TODO: rename

from archival import PreservedScene
from core import *
from selection import Selector

import asyncio
from dataclasses import dataclass
from numpy import random
import re
from typing import List, Optional, Tuple
import yaml

with open("data/prompts/generator.txt") as f:
    generator_prompt = f.read().strip()

with open("data/prompt_fragments/topics_curated.txt") as f:
    topics = f.read().splitlines()

with open("data/characters.yaml") as f:
    characters = [Character.from_dict(character) for character in yaml.safe_load(f)]


def character_by_name(name):
    for character in characters:
        if character.name == name:
            return character


class CharactersGenerator:
    pass


@dataclass(match_args=True)
class ConstantCharacters(CharactersGenerator):
    characters: List[str]


@dataclass(match_args=True)
class ChooseCharactersFrom(CharactersGenerator):
    characters: List[str]


class LocationGenerator:
    pass


@dataclass(match_args=True)
class ConstantLocation(LocationGenerator):
    location: str


@dataclass(match_args=True)
class LocationPs(LocationGenerator):
    location_ps: List[Tuple[str, float]]


location_ps = [
    ("amid an endless expanse of blinding gray", 1),
    ("on a wooden platform amid the void", 5),
    ("in the little girl and father's living room", 5),
]


class Scene:
    @classmethod
    async def create(
        cls,
        characters_generator: CharactersGenerator,
        location_generator: LocationGenerator,
        selector: Selector,
        n: int,
        base_url: str,
        model: str,
        top_p: float,
        topic: Optional[str] = None,
    ):
        scene = cls()

        scene.selector = selector
        scene.n = n
        scene.top_p = top_p

        topic = topic if topic is not None else random.choice(topics)

        match location_generator:
            case ConstantLocation(location):
                location_ = location
            case LocationPs(location_ps):
                location_ps = [(location, p / sum(p for _, p in location_ps)) for location, p in location_ps]
                location_ = random.choice([location for location, _ in location_ps], p=[p for _, p in location_ps])

        match characters_generator:
            case ConstantCharacters(characters):
                characters_ = characters
            case ChooseCharactersFrom(characters):
                prompt = generator_prompt.split("{characters}")[0].format(
                    location=location_,
                    topic_line=f"Topic: {topic}\n",
                )

                while True:
                    response = await complete_with_retry(
                        base_url,
                        model=model,
                        prompt=prompt,
                        max_tokens=100,
                        temperature=1,
                        top_p=0.95,
                        stop=["\n"]
                    )
                    characters = [character_by_name(character) for character in response.choices[0].text[1:].split(", ")]
                    if all(character is not None for character in characters):
                        characters_ = sorted(characters, key=lambda character: str.casefold(character.name))
                        break

        scene.data = SceneData(
            messages=[],
            is_complete=False,
            characters=characters_,
            location=location_,
            topic=topic,
            base_url=base_url,
            model=model,
            include_topic_line=True,
            seed=random.randint(2**32),
        )

        console.log(f"Selection profile: {selector} @ {n}")
        console.log(f"Characters: {', '.join(character.name for character in characters_)}")
        console.log(f"Location: {location_}")
        console.log(f"Topic: {topic}")

        return scene

    @classmethod
    def continue_(cls, selector, n, top_p, preserved_scene: PreservedScene):
        scene = cls()

        scene.selector = selector
        scene.n = n
        scene.top_p = top_p

        scene.data = SceneData(
            messages=preserved_scene.messages,
            is_complete=False,
            characters=characters,
            location=preserved_scene.location,
            topic=preserved_scene.topic,
            base_url="https://api.openai.com/v1",
            model="gpt-4-base",
            include_topic_line=True,
            seed=random.randint(2**32),
        )
        return scene

    def prompt(self, history=None):
        if history is None:
            history = self.data.messages.copy()

        base_prompt = generator_prompt.format(
            characters=f" {', '.join([character.name for character in self.data.characters])}",
            location=self.data.location,
            topic_line=f"Topic: {self.data.topic}\n" if self.data.include_topic_line else "",
        )

        prompt = base_prompt

        for message in history:
            prompt += f"\n{message}"

        n_tokens = len(tokenizer.encode(prompt, disallowed_special={}))
        while n_tokens > 8000:
            history.pop(0)
            prompt = f"{base_prompt}\n[...]"
            for message in history:
                prompt += f"\n{message}"

        if prompt.endswith(" [...]"):
            prompt = prompt[:-6]
        return prompt

    async def write_messages(self):
        self.data.seed = random.randint(2**32)

        history = self.data.messages.copy()
        messages = []
        completing = len(history) > 0 and history[-1].incomplete

        ns = [128] * (self.n // 128) + ([self.n % 128] if self.n % 128 != 0 else [])
        async def get_response(n):
            return await complete_with_retry(
                self.data.base_url,
                model=self.data.model,
                prompt=self.prompt(history=history) + ("" if completing else "\n"),
                n=n,
                # max_tokens=192 if len(self.data.messages) < 6 else 256,
                max_tokens=192,
                temperature=1,
                top_p=self.top_p,
                frequency_penalty=0.25,
                # stop=["\n"] if len(self.data.messages) < 6 else None,
                stop=["\n"],
                logit_bias={"4794": -3 + len(history) * 0.05}
            )

        responses = await asyncio.gather(*[get_response(n) for n in ns])
        # choices = [choice.text.splitlines() for response in responses for choice in response.choices if choice.finish_reason != "length" or len(self.data.messages) >= 6]
        choices = [choice.text.splitlines() for response in responses for choice in response.choices if choice.finish_reason != "length"]

        if choices[0][0] == "END":
            return []

        def process_message(message, completing=False, incomplete=False):
            if completing:
                speaker = history[-1].speaker
                body = message
            else:
                match_ = re.fullmatch(r"<([^>]+)>\s*(.*)", message)
                if match_ is None:
                    return

                speaker = match_.group(1)
                if character_by_name(speaker) is None:
                    return

                body = match_.group(2)

            if body.startswith("[") and body.endswith("]"):
                action = body[1:-1]
                if action in character_by_name(speaker).actions:
                    return Message(speaker, action, MessageType.ACTION, completing=completing, incomplete=incomplete)
                else:
                    return
            elif any(substring in body for substring in ["[", "]", "<", ">"]):
                return
            else:
                return Message(speaker, body, MessageType.SPEECH, completing=completing, incomplete=incomplete)

        def process(choice):
            # if len(self.data.messages) > 6:
            #     incomplete = len(choice) == 1
            #     if not incomplete:
            #         choice = choice[:-1]
            # else:
            #     incomplete = False

            # lol
            incomplete = False

            choice = choice[:3]

            messages = [process_message(message, completing=i == 0 if completing else False, incomplete=incomplete) for i, message in enumerate(choice)]

            if any(message is None for message in messages):
                return
            return messages

        unselected_choices = [process(choice) for choice in choices]
        unselected_choices = [choice for choice in unselected_choices if choice is not None]

        if len(unselected_choices) == 0:
            console.log("No valid messages. Retrying...")
            return await self.write_messages()
        elif len(unselected_choices) == 1:
            messages += unselected_choices[0]
        else:
            messages += unselected_choices[await self.selector.select(unselected_choices, self.data)]

        self.data.is_complete = messages == []

        if not self.data.is_complete:
            if messages[0].completing:
                updated_message = messages.pop(0)
                updated_message.body = history[-1].body + updated_message.body
                updated_message.completing = False
                history[-1] = updated_message
            else:
                console.log(f"Wrote message...")

        history += messages
        self.data.messages = history

        return messages

    async def write(self):
        while True:
            messages = await self.write_messages()

            for message in messages:
                yield message

            if self.data.is_complete:
                console.log("Scene complete!")
                break

            # if random.random() < (len(self.data.messages) - 6) * 0.4:
            #     self.data.include_topic_line = False
