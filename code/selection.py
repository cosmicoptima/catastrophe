from core import *

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from lark import Lark, Transformer
from math import ceil, exp
from num2words import num2words
import re
from typing import List
import yaml

with open("data/prompts/gps.txt") as f:
    gps_prompt = f.read().strip()

with open("data/prompt_fragments/gps_goals.yaml") as f:
    gps_goals = yaml.safe_load(f)

with open("data/prompts/ooms.txt") as f:
    ooms_prompt = f.read().strip()

with open("data/prompt_fragments/ooms_instructions.yaml") as f:
    ooms_instructions = yaml.safe_load(f)

with open("data/prompt_fragments/ooms_instructional_addenda.yaml") as f:
    ooms_instructional_addenda = yaml.safe_load(f)

with open("data/prompts/qps.txt") as f:
    qps_prompt = f.read().strip() + " "

with open("data/prompt_fragments/qps_questions.yaml") as f:
    qps_questions = yaml.safe_load(f)


class Selector(ABC):
    @abstractmethod
    async def select(self, choices: List[List[Message]], data: SceneData):
        ...


@dataclass
class OOMS(Selector):
    instructions: str
    instructional_addenda: List[str] = None

    async def select(self, choices: List[List[Message]], data: SceneData):
        instructions = ooms_instructions[self.instructions]
        for addendum in self.instructional_addenda:
            instructions += " " + ooms_instructional_addenda[addendum]

        rendered_choices = ["\n".join([str(message) for message in choice]) for choice in choices]
        continuations = "\n\n".join([f"#{i + 1}:\n{choice}" for i, choice in enumerate(rendered_choices)])

        history_messages = data.messages
        history_truncated = False

        while True:
            if len(history_messages) == 0:
                history = "\n[No messages.]"
            else:
                history = "".join([f"\n{message}" for message in history_messages])
                if history_truncated:
                    history = f"\n[...]{history}"

            prompt = ooms_prompt.format(
                characters=f" {', '.join([character.name for character in data.characters])}",
                location=data.location,
                same_location=" also" if same_location(data.location) else "",
                # topic_prefix=f"This time the characters discuss {data.topic}. It" if data.include_topic_line else "This dialog",
                topic_prefix=f"This time the characters discuss {data.topic}. It",

                n_continuations=num2words(len(choices)) if len(choices) < 10 else len(choices),
                instructions=instructions,
                continuations=continuations,
                history=history,
            )

            if len(tokenizer.encode(prompt, disallowed_special={})) < 8000:
                break
            
            history_messages.pop(0)

        while True:
            response = await complete_with_retry(
                model="gpt-4-base",
                prompt=prompt,
                max_tokens=1,
                temperature=0,
            )

            match = re.match(r"\d+", response.choices[0].text)
            if match is not None:
                choice_index = int(match.group(0))
                if 1 <= choice_index <= len(choices):
                    return choice_index - 1

    def __str__(self):
        return f"OOMS({self.instructions})"
    

@dataclass
class QPS(Selector):
    question: str

    async def select(self, choices: List[List[Message]], data: SceneData):
        question = qps_questions[self.question]["question"]
        answer_1 = qps_questions[self.question]["answer_1"]

        def prompt_conditional_on(choice):
            history_messages = data.messages + choice
            history_truncated = False

            while True:
                history = "".join([f"\n{message}" for message in history_messages])
                if history_truncated:
                    history = f"\n[...]{history}"

                prompt = qps_prompt.format(
                    question=question,
                    answer_1=answer_1,

                    location=data.location,
                    topic_line=f"Topic: {data.topic}\n" if data.include_topic_line else "",
                    characters=", ".join([character.name for character in data.characters]),

                    history=history,
                )

                if len(tokenizer.encode(prompt, disallowed_special={})) < 8000:
                    return prompt
                
                history_messages.pop(0)
            
        prompts = [prompt_conditional_on(choice) for choice in choices]
        responses = await complete_with_retry(
            model="gpt-4-base",
            prompt=prompts,
            max_tokens=1,
            logprobs=5,
        )

        valid_tokens = [str(i + 1) for i in range(20)]

        def expected_score(choice):
            ps = [(int(token), logprob) for token, logprob in choice.logprobs.top_logprobs[0].items() if token in valid_tokens]
            total_p = sum(exp(logprob) for _, logprob in ps)
            return sum(token * exp(logprob) / total_p for token, logprob in ps)

        console.log("Expected scores:", *[f"{expected_score(choice):.3f}" for choice in responses.choices])

        return max(range(len(choices)), key=lambda i: expected_score(responses.choices[i]))
    
    def __str__(self):
        return f"QPS({self.question})"


@dataclass
class GPS(Selector):
    goal: str

    async def select(self, choices: List[List[Message]], data: SceneData):
        goal = gps_goals[self.goal]

        def prompt_conditional_on(choice):
            history_messages = data.messages + choice
            history_truncated = False

            while True:
                history = "".join([f"\n{message}" for message in history_messages])
                if history_truncated:
                    history = f"\n[...]{history}"

                prompt = gps_prompt.format(
                    goal=goal,

                    location=data.location,
                    topic_line=f"Topic: {data.topic}\n" if data.include_topic_line else "",
                    characters=", ".join([character.name for character in data.characters]),

                    history=history,
                )

                if len(tokenizer.encode(prompt, disallowed_special={})) < 8000:
                    return prompt
                
                history_messages.pop(0)
            
        prompts = [prompt_conditional_on(choice) for choice in choices]
        responses = await complete_with_retry(
            model="gpt-4-base",
            prompt=prompts,
            max_tokens=1,
            logprobs=5,
        )

        def score(choice):
            return exp(choice.logprobs.top_logprobs[0]["1"])

        console.log("Scores:", *[f"{score(choice):.3f}" for choice in responses.choices])

        return max(range(len(choices)), key=lambda i: score(responses.choices[i]))
    
    def __str__(self):
        return f"GPS({self.goal})"


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
    n: int

    async def select(self, messages: List[Message], data: SceneData):
        if self.n > 2:
            return await Divide(by=self.by, using=self.using, then=DivideSelf(by=self.by, using=self.using, n=self.n - 1)).select(messages, data)
        elif self.n == 2:
            return await Divide(by=self.by, using=self.using, then=self.using).select(messages, data)
        elif self.n == 1:
            return await self.using.select(messages, data)
        else:
            raise ValueError(f"Invalid n: {self.n}")


# TODO: automatically generate grammar from selectors

selector_parser = Lark(r"""
    COMMA: WS? "," WS?
    WORD: /\w+/
    WORDS: /\w(\w| )+\w/

    divide: "Divide(" INT COMMA selector COMMA selector ")"
    divide_self: "DivideSelf(" INT COMMA selector COMMA INT ")"
    gps: "GPS(" WORD ")"
    ooms_no_addenda: "OOMS(" WORD ")"
    ooms_addenda: "OOMS(" WORD COMMA WORDS ")"
    qps: "QPS(" WORD ")"
              
    selector: divide | divide_self | gps | ooms_no_addenda | ooms_addenda | qps

    %import common.INT
    %import common.WS
""", start="selector")


class SelectorTransformer(Transformer):
    def divide(self, c):
        return Divide(int(c[0]), c[2], c[4])
    
    def divide_self(self, c):
        return DivideSelf(int(c[0]), c[2], int(c[4]))
    
    def gps(self, c):
        return GPS(str(c[0]))
    
    def ooms_no_addenda(self, c):
        return OOMS(str(c[0]), [])
    
    def ooms_addenda(self, c):
        return OOMS(str(c[0]), str(c[2]).split(" "))
    
    def qps(self, c):
        return QPS(str(c[0]))
    
    def selector(self, c):
        return c[0]


def from_text(selector: str) -> Selector:
    return SelectorTransformer().transform(selector_parser.parse(selector))