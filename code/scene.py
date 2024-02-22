# TODO: rename

from core import *
from selection import Selector

from dataclasses import dataclass
from numpy import random
import re
from tiktoken import get_encoding
from typing import List, Optional, Tuple
import yaml

tokenizer = get_encoding("cl100k_base")

with open("data/prompts/generator.txt") as f:
    generator_prompt = f.read().strip()

with open("data/prompt_fragments/topics.txt") as f:
    topics = f.read().splitlines()

with open("data/characters.yaml") as f:
    characters = [Character(character["name"], character["actions"]) for character in yaml.safe_load(f)]


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
    # ("in Hell", 1),
    ("on a wooden platform amid the void", 2),
    ("in the little girl and father's living room", 7),
]


class Scene:
    @classmethod
    async def create(
        cls,
        characters_generator: CharactersGenerator,
        location_generator: LocationGenerator,
        selector: Selector,
        n: int,
        topic: Optional[str] = None,
    ):
        scene = cls()

        scene.selector = selector
        scene.n = n

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
                        model="gpt-4-base",
                        prompt=prompt,
                        max_tokens=100,
                        temperature=1,
                        top_p=0.96,
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
        )

        console.log(f"Characters: {', '.join(character.name for character in characters_)}")
        console.log(f"Location: {location_}")
        console.log(f"Topic: {topic}")

        scene.silenced_characters = set()
        scene.include_topic_line = True

        return scene
    
    def prompt(self):
        base_prompt = generator_prompt.format(
            characters=f" {', '.join([character.name for character in self.data.characters])}",
            location=self.data.location,
            topic_line=f"Topic: {self.data.topic}\n" if self.include_topic_line else "",
        )
        messages = self.data.messages

        prompt = base_prompt
        for message in messages:
            prompt += f"\n{message}"

        n_tokens = len(tokenizer.encode(prompt))
        while n_tokens > 8000:
            messages.pop(0)
            prompt = f"{base_prompt}\n..."
            for message in messages:
                prompt += f"\n{message}"

        return prompt

    async def write_messages(self):
        messages = []

        console.log("Generating messages...")

        response = await complete_with_retry(
            model="gpt-4-base",
            prompt=self.prompt() + "\n",
            n=self.n,
            max_tokens=500,
            temperature=1,
            top_p=0.96,
            stop=["\n"],
            logit_bias={"4794": -3 + len(self.data.messages) * 0.05}
        )

        if response.choices[0].text.startswith("END"):
            console.log("Scene complete!")
            return []

        def process(choice):
            if choice.finish_reason == "length":
                return

            match_ = re.fullmatch(r"<([^>]+)>\s*(.*)", choice.text)
            if match_ is None:
                return

            speaker = match_.group(1)
            if character_by_name(speaker) is None or speaker in self.silenced_characters:
                return

            body = match_.group(2)
            if body.startswith("[") and body.endswith("]"):
                action = body[1:-1]
                if action in character_by_name(speaker).actions:
                    return Message(speaker, action, MessageType.ACTION)
                else:
                    return
            elif any(substring in body for substring in ["[", "]", "<", ">"]):
                return
            else:
                return Message(speaker, body, MessageType.SPEECH)

        console.log("Lengths of outputs: " + ", ".join(str(len(choice.text)) for choice in response.choices))
        unselected_messages = [process(choice) for choice in response.choices]
        unselected_messages = [message for message in unselected_messages if message is not None]

        if len(unselected_messages) == 0:
            console.log("No valid messages. Retrying...")
            return await self.write_messages()
        elif len(unselected_messages) == 1:
            messages.append(unselected_messages[0])
        else:
            messages.append(unselected_messages[await self.selector.select(unselected_messages, self.data)])

        return messages

    async def write(self):
        while True:
            messages = await self.write_messages()
            if messages == []:
                self.data.is_complete = True
                break

            self.data.messages += messages
            for message in messages:
                yield message
            
            if random.random() < (len(messages) - 6) * 0.4:
                self.include_topic_line = False
