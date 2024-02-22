from core import *

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from lark import Lark, Transformer
from math import ceil
from num2words import num2words
import re
from typing import List
import yaml

with open("data/prompts/ooms.txt") as f:
    ooms_prompt = f.read().strip()

with open("data/prompt_fragments/ooms_instructions.yaml") as f:
    ooms_instructions = yaml.safe_load(f)


class Selector(ABC):
    @abstractmethod
    async def select(self, messages: List[Message], data: SceneData):
        ...


@dataclass
class OOMS(Selector):
    instructions: str
    temperature: float = 0.3

    async def select(self, messages: List[Message], data: SceneData):
        continuations = "\n".join([f"#{i + 1}: {message}" for i, message in enumerate(messages)])
        history = "".join([f"\n{message}" for message in data.messages])

        prompt = ooms_prompt.format(
            characters=f" {', '.join([character.name for character in data.characters])}",
            location=data.location,
            same_location=" also" if same_location(data.location) else "",
            topic=data.topic,

            n_continuations=num2words(len(messages)) if len(messages) < 10 else len(messages),
            instructions=ooms_instructions[self.instructions],
            continuations=continuations,
            history=history,
        )

        while True:
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
                    return choice - 1

    def __str__(self):
        return f"OOMS({self.instructions})"


@dataclass
class Divide(Selector):
    by: int
    using: Selector
    then: Selector

    async def select(self, messages: List[Message], data: SceneData):
        chunks = [messages[i * self.by:(i + 1) * self.by] for i in range(ceil(len(messages) / self.by))]

        async def select(chunk):
            return chunk[await self.using.select(chunk, data)]
        
        new_messages = await asyncio.gather(*[select(chunk) for chunk in chunks])
        return messages.index(new_messages[await self.then.select(new_messages, data)])


@dataclass
class DivideSelf(Selector):
    by: int
    using: Selector

    async def select(self, messages: List[Message], data: SceneData):
        return await Divide(by=self.by, using=self.using, then=self.using).select(messages, data)


# TODO: automatically generate grammar from criteria

selector_parser = Lark(r"""
    ?int: INT              
    ?string: /\w+/

    divide: "Divide(" INT "," criterion "," criterion ")"
    divide_self: "DivideSelf(" int "," criterion ")"
    ooms: "OOMS(" string ")"
              
    selector: divide | divide_self | ooms

    %import common.INT
    %import common.WS

    %ignore WS
""", start="selector")


class SelectorTransformer(Transformer):
    def int(self, items):
        return int(items[0])

    def string(self, items):
        return items[0]

    def divide(self, items):
        return Divide(items[0], items[1], items[2])
    
    def divide_self(self, items):
        return DivideSelf(int(items[0]), items[1])
    
    def ooms(self, items):
        return OOMS(items[0])
    
    def selector(self, items):
        return items[0]


def from_text(selector: str) -> Selector:
    return SelectorTransformer().transform(selector_parser.parse(selector))