from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass
from enum import StrEnum, auto
from openai import AsyncOpenAI, InternalServerError
import os
from rich.console import Console
from typing import List

console = Console(highlight=False)
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))


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


def same_location(location):
    return location == "in the little girl and father's living room"


class MessageType(StrEnum):
    SPEECH = auto()
    ACTION = auto()


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