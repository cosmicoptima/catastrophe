from dotenv import load_dotenv
load_dotenv()

from abc import ABC, abstractmethod
from dataclasses import dataclass
from num2words import num2words
from numpy import random
from openai import AsyncOpenAI
import os
import re
from rich.console import Console
from typing import List, Optional, Tuple

console = Console(highlight=False)
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))

with open("assets/prompts/generator.txt") as f:
    generator_prompt = f.read().strip()

with open("assets/prompts/one_off_selector.txt") as f:
    one_off_selector_prompt = f.read().strip()

with open("assets/topics.txt") as f:
    topics = f.read().splitlines()


@dataclass
class Character:
    name: str
    actions: List[str]


characters = [
    Character("angel", ["freeze"]),
    Character("depressed man in corner", ["smoke"]),
    Character("elf", ["walk", "fall over"]),
    Character("father", ["smile", "frown", "ascend"]),
    Character("little girl", ["flip", "frown"]),
    Character("Santa", ["menace", "hypermenace"]),

    Character("mother", ["smile", "fret"]),
]


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


def same_location(location):
    return location == "in the little girl and father's living room"


@dataclass
class Message:
    speaker: str
    body: str
    is_action: bool

    def __str__(self):
        if self.is_action:
            return f"<{self.speaker}> [{self.body}]"
        else:
            return f"<{self.speaker}> {self.body}"


@dataclass
class SceneData:
    messages: List[Message]
    characters: List[Character]
    location: str
    topic: str


class Criterion(ABC):
    @abstractmethod
    async def evaluate(self, messages: List[Message], data: SceneData):
        ...


@dataclass
class Invert(Criterion):
    subcriterion: Criterion

    async def evaluate(self, messages: List[Message], data: SceneData):
        return [-score for score in await self.subcriterion.evaluate(messages)]

    def __str__(self):
        return f"Invert({self.subcriterion})"


@dataclass
class RandomChoice(Criterion):
    subcriteria: List[Criterion]
    ps: Optional[List[float]] = None

    async def evaluate(self, messages: List[Message], data: SceneData):
        return await random.choice(self.subcriteria, p=self.ps).evaluate(messages)

    def __str__(self):
        return f"RandomChoice([{', '.join(str(criterion) for criterion in self.subcriteria)}]{', ' + str(self.ps) if self.ps else ''})"


@dataclass
class Longest(Criterion):
    async def evaluate(self, messages: List[Message], data: SceneData):
        return [14 if message.is_action else len(message.body) for message in messages]

    def __str__(self):
        return "Longest"


@dataclass
class Insts:
    instructions: str


@dataclass
class BestInsts(Insts):
    instructions: str = "You will choose the best one."


@dataclass
class FrustrationInsts(Insts):
    instructions: str = "Choose the continuation that causes you to feel the angriest or most frustrated, particularly in the context of the rest of the dialog. If none of them seem to provoke you, pay attention to subtler bodily sensations. Think about the dialog conditional on it including each continuation, and take note of any tension in your chest or stomach, or any increase in heart rate. Choose the continuation that induces the most reactance."


@dataclass
class SchizoInsts(Insts):
    instructions: str = "Choose the continuation that causes you to feel the most schizo."


@dataclass
class OneOffSelector(Criterion):
    instructions: Insts

    async def evaluate(self, messages: List[Message], data: SceneData):
        continuations = "\n".join([f"#{i + 1}: {message}" for i, message in enumerate(messages)])
        history = "\n".join([str(message) for message in data.messages])

        prompt = one_off_selector_prompt.format(
            characters=f" {', '.join([character.name for character in data.characters])}",
            location=data.location,
            same_location=" also" if same_location(data.location) else "",
            topic=data.topic,

            n_continuations=num2words(len(messages)) if len(messages) < 10 else len(messages),
            instructions=self.instructions,
            continuations=continuations,
            history=history,
        )

        while True:
            console.log("Generating one-off selector response...")

            response = await openai.completions.create(
                model="gpt-4-base",
                prompt=prompt,
                max_tokens=1,
                temperature=1,
                top_p=0.96,
            )

            match = re.match(r"\d+", response.choices[0].text)
            if match is not None:
                choice = int(match.group(0))
                if 1 <= choice <= len(messages):
                    return [1 if i == choice - 1 else 0 for i in range(len(messages))]

    def __str__(self):
        return f"OneOffSelector({self.instructions})"


class Scene:
    @classmethod
    async def create(
        cls,
        characters_generator: CharactersGenerator,
        location_generator: LocationGenerator,
        criterion: Criterion,
        n: int,
        topic: Optional[str] = None
    ):
        scene = cls()

        scene.criterion = criterion
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
                    same_location=" also" if same_location(location_) else "",
                    topic=topic,
                )

                while True:
                    response = await openai.completions.create(
                        model="gpt-4-base",
                        prompt=prompt,
                        max_tokens=100,
                        temperature=1,
                        top_p=0.96,
                        stop=["."]
                    )
                    characters = [character_by_name(character) for character in response.choices[0].text[1:].split(", ")]
                    if all(character is not None for character in characters):
                        characters_ = sorted(characters, key=lambda character: str.casefold(character.name))
                        break
        
        scene.data = SceneData(
            messages=[],
            characters=characters_,
            location=location_,
            topic=topic,
        )

        return scene
    
    @property
    def prompt(self):
        prompt = generator_prompt.format(
            characters=f" {', '.join(character.name for character in self.data.characters)}",
            location=self.data.location,
            same_location=" also" if same_location(self.data.location) else "",
            topic=self.data.topic,
        )
        for message in self.data.messages:
            prompt += f"\n{message}"

        return prompt
    
    async def write_message(self):
        prompt = self.prompt + "\n"

        console.log("Generating speakers...")

        response = await openai.completions.create(
            model="gpt-4-base",
            prompt=prompt,
            n=self.n,
            max_tokens=50,
            temperature=1,
            top_p=0.96,
            stop=[">"],
        )

        speakers = [choice.text[1:] for choice in response.choices if choice.text.startswith("<")]
        speakers = [speaker for speaker in speakers if len([character for character in self.data.characters if character.name == speaker]) > 0]

        prompts = [prompt + f"<{speaker}>" for speaker in speakers]

        console.log("Generating body texts...")

        response = await openai.completions.create(
            model="gpt-4-base",
            prompt=prompts,
            max_tokens=300,
            temperature=1,
            top_p=0.96,
            stop=["\n"],
        )

        def process(i):
            if response.choices[i].finish_reason == "length":
                return

            output = response.choices[i].text[1:]
            speaker = speakers[i]

            if output.startswith("[") and output.endswith("]"):
                action = output[1:-1]
                if action in character_by_name(speaker).actions:
                    return Message(speaker, action, True)
                else:
                    return
            elif any(substring in output for substring in ["[", "]", "<", ">"]):
                return
            else:
                return Message(speaker, output, False)

        console.log("Lengths of outputs: " + ", ".join(str(len(choice.text)) for choice in response.choices))
        messages = [process(i) for i in range(len(response.choices))]
        messages = [message for message in messages if message is not None]
    
        if self.n == 1:
            return messages[0]
        else:
            scores = await self.criterion.evaluate(messages, self.data)
            return messages[scores.index(max(scores))]
    
    async def write(self):
        while True:
            message = await self.write_message()
            self.data.messages.append(message)
            yield message