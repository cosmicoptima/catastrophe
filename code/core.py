from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass
from enum import StrEnum, auto
from openai import AsyncOpenAI, InternalServerError
import os
from rich.console import Console
from tiktoken import get_encoding
from typing import List, Optional

console = Console(highlight=False)
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))
tokenizer = get_encoding("cl100k_base")


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

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["actions"])


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
    completing: bool = False
    incomplete: bool = False

    def __str__(self):
        if self.completing:
            preface = "[...]"
        else:
            preface = f"<{self.speaker}>"
        
        if self.type_ == MessageType.ACTION:
            str_ = f"{preface} [{self.body}]"
        else:
            str_ = f"{preface} {self.body}"
        
        if self.completing:
            str_ = f"[...] {str_}"
        if self.incomplete:
            str_ += " [...]"
        
        return str_


@dataclass
class SceneData:
    messages: List[Message]
    is_complete: bool
    characters: List[Character]
    location: str
    topic: str
    include_topic_line: bool