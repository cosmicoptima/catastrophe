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


async def complete_with_retry(base_url, **kwargs):
    client = AsyncOpenAI(base_url=base_url, api_key=os.getenv("API_KEY"), organization=os.getenv("ORG"))
    while True:
        try:
            return await client.completions.create(**kwargs)
        except InternalServerError:
            console.log("Internal server error. Retrying...")


async def chat_complete_with_retry(base_url, model, prompt, api_key=None, provider=None, **kwargs):
    import asyncio as aio
    from openai import RateLimitError, APITimeoutError
    client = AsyncOpenAI(base_url=base_url, api_key=api_key or "dummy", timeout=60.0)

    extra_body = {}
    if provider:
        extra_body["provider"] = {"order": [provider]}

    console.log(f"Calling selector: {model}...")
    while True:
        try:
            result = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra_body if extra_body else None,
                **kwargs
            )
            console.log(f"Selector returned: {result.choices[0].message.content[:50] if result.choices else 'empty'}...")
            return result
        except InternalServerError:
            console.log("Internal server error. Retrying...")
        except RateLimitError as e:
            console.log(f"Rate limit hit. Waiting 2s...")
            await aio.sleep(2)
        except APITimeoutError:
            console.log("Timeout. Retrying...")
        except Exception as e:
            console.log(f"Error: {type(e).__name__}: {e}")
            raise


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
    base_url: str
    model: str
    include_topic_line: bool
    seed: Optional[int] = None
    selector_base_url: Optional[str] = None
    selector_model: Optional[str] = None
    selector_api_key: Optional[str] = None
    selector_provider: Optional[str] = None


@dataclass
class PreservedScene:
    location: str
    topic: str
    messages: List[Message]

    async def replay(self):
        for message in self.messages:
            yield message