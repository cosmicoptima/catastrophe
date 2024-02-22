from dotenv import load_dotenv
load_dotenv()

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from math import ceil
from num2words import num2words
from numpy import random
from openai import AsyncOpenAI, InternalServerError
import os
import re
from rich.console import Console
from tiktoken import get_encoding
from typing import List, Optional, Tuple
import yaml

console = Console(highlight=False)
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))
tokenizer = get_encoding("cl100k_base")

with open("assets/prompts/generator.txt") as f:
    generator_prompt = f.read().strip()

with open("assets/prompts/one_off_selector.txt") as f:
    one_off_selector_prompt = f.read().strip()

with open("assets/prompts/operator.txt") as f:
    operator_prompt = f.read().strip()

with open("assets/prompt_fragments/topics.txt") as f:
    topics = f.read().splitlines()

with open("assets/prompt_fragments/one_off_selector_instructions.yaml") as f:
    one_off_selector_instructions = yaml.safe_load(f)


async def complete_with_retry(**kwargs):
    while True:
        try:
            return await openai.completions.create(**kwargs)
        except InternalServerError:
            console.log("Internal server error. Retrying...")


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


location_ps = [
    # ("in Hell", 1),
    ("on a wooden platform amid the void", 2),
    ("in the little girl and father's living room", 7),
]


class MessageType(StrEnum):
    SPEECH = auto()
    ACTION = auto()
    COMMAND = auto()


@dataclass
class Message:
    speaker: str
    body: str
    type_: MessageType

    def __str__(self):
        if self.type_ == MessageType.ACTION:
            return f"<{self.speaker}> [{self.body}]"
        else:
            return f"<{self.speaker}> {self.body}"


@dataclass
class SceneData:
    messages: List[Message]
    is_complete: bool
    characters: List[Character]
    location: str
    topic: str


class MessageCriterion(ABC):
    @abstractmethod
    async def evaluate(self, messages: List[Message], data: SceneData):
        ...


@dataclass
class OneOffMessageSelector(MessageCriterion):
    instructions: str
    temperature: float = 0.3

    async def evaluate(self, messages: List[Message], data: SceneData):
        continuations = "\n".join([f"#{i + 1}: {message}" for i, message in enumerate(messages)])
        history = "".join([f"\n{message}" for message in data.messages])

        prompt = one_off_selector_prompt.format(
            characters=f" {', '.join([character.name for character in data.characters])}",
            location=data.location,
            same_location=" also" if same_location(data.location) else "",
            topic=data.topic,

            n_continuations=num2words(len(messages)) if len(messages) < 10 else len(messages),
            instructions=one_off_selector_instructions[self.instructions],
            continuations=continuations,
            history=history,
        )

        while True:
            console.log("Generating one-off message selector response...")

            response = await complete_with_retry(
                model="gpt-4-base",
                prompt=prompt,
                max_tokens=1,
                temperature=self.temperature,
                top_p=0.96,
            )

            match = re.match(r"\d+", response.choices[0].text)
            if match is not None:
                choice = int(match.group(0))
                if 1 <= choice <= len(messages):
                    console.log(f"Selected continuation #{choice}.")
                    return choice - 1

    def __str__(self):
        return f"OneOffSelector({self.instructions})"


OOMS = OneOffMessageSelector


@dataclass
class Divide(MessageCriterion):
    by: int
    using: MessageCriterion
    then: MessageCriterion

    async def evaluate(self, messages: List[Message], data: SceneData):
        chunks = [messages[i * self.by:(i + 1) * self.by] for i in range(ceil(len(messages) / self.by))]

        async def evaluate(chunk):
            return chunk[await self.using.evaluate(chunk, data)]
        
        new_messages = await asyncio.gather(*[evaluate(chunk) for chunk in chunks])
        return messages.index(new_messages[await self.then.evaluate(new_messages, data)])


@dataclass
class DivideSelf(MessageCriterion):
    by: int
    using: MessageCriterion

    async def evaluate(self, messages: List[Message], data: SceneData):
        return await Divide(by=self.by, using=self.using, then=self.using).evaluate(messages, data)


class Command:
    pass


@dataclass(match_args=True)
class Display(Command):
    message: Optional[str]


@dataclass(match_args=True)
class Silence(Command):
    character: Character


@dataclass(match_args=True)
class Unsilence(Command):
    character: Character


def parse_operator_command(command):
    display_match = re.fullmatch(r'display\s"(.*)"', command)
    if display_match is not None:
        return Display(display_match.group(1))
    elif command == "display null":
        return Display(None)

    silence_match = re.fullmatch(r'silence\s"(.*)"', command)
    if silence_match is not None:
        character = character_by_name(silence_match.group(1))

        if character is not None:
            return Silence(character)

    unsilence_match = re.fullmatch(r'unsilence\s"(.*)"', command)
    if unsilence_match is not None:
        character = character_by_name(unsilence_match.group(1))

        if character is not None:
            return Unsilence(character)


class Scene:
    @classmethod
    async def create(
        cls,
        characters_generator: CharactersGenerator,
        location_generator: LocationGenerator,
        message_criterion: MessageCriterion,
        message_n: int,
        topic: Optional[str] = None,
        operate: bool = False,
    ):
        scene = cls()

        scene.message_criterion = message_criterion
        scene.message_n = message_n

        topic = topic if topic is not None else random.choice(topics)

        scene.operate = operate

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
    
    def prompt(self, operator=False):
        prompt_template = operator_prompt if operator else generator_prompt

        characters = [character.name for character in self.data.characters]
        if operator:
            characters.append("operator")
            characters.sort(key=str.casefold)

        base_prompt = prompt_template.format(
            characters=f" {', '.join(characters)}",
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

        async def get_character_messages():
            return await complete_with_retry(
                model="gpt-4-base",
                prompt=self.prompt() + "\n",
                n=self.message_n,
                max_tokens=500,
                temperature=1,
                top_p=0.96,
                stop=["\n"],
                logit_bias={"4794": -3 + len(self.data.messages) * 0.05}
            )

        async def get_operator_message():
            return await complete_with_retry(
                model="gpt-4-base",
                prompt=self.prompt(operator=True) + "\n",
                max_tokens=500,
                temperature=1,
                top_p=0.96,
                stop=["\n"],
                logit_bias={"8043": 2}
            )

        if self.operate:
            character_response, operator_response = await asyncio.gather(
                get_character_messages(),
                get_operator_message(),
            )
        else:
            character_response = await get_character_messages()
            operator_response = None

        if character_response.choices[0].text.startswith("END"):
            console.log("Scene complete!")
            return []

        if operator_response is not None:
            operator_match = re.fullmatch(r"<operator>\s*(.*)", operator_response.choices[0].text)
            if operator_match is not None:
                command = operator_match.group(1)
                console.log(f"Operator: {command}")

                valid = True

                match parse_operator_command(command):
                    case Silence(character):
                        if character.name in self.silenced_characters:
                            valid = False
                        else:
                            self.silenced_characters.add(character.name)
                    case Unsilence(character):
                        if character.name in self.silenced_characters:
                            self.silenced_characters.remove(character.name)
                        else:
                            valid = False
                    case None:
                        valid = False

                if valid:
                    messages.append(Message(speaker="operator", body=command, type_=MessageType.COMMAND))

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

        console.log("Lengths of outputs: " + ", ".join(str(len(choice.text)) for choice in character_response.choices))
        character_messages = [process(choice) for choice in character_response.choices]
        character_messages = [message for message in character_messages if message is not None]

        if len(character_messages) == 0:
            console.log("No valid messages. Retrying...")
            return await self.write_messages()
    
        if self.message_n == 1:
            messages.append(character_messages[0])
        else:
            messages.append(character_messages[await self.message_criterion.evaluate(character_messages, self.data)])

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
