from dotenv import load_dotenv
load_dotenv()

import asyncio
from dataclasses import dataclass
import nanoid
from num2words import num2words
from numpy import random
from openai import AsyncOpenAI
import os
from pedalboard import Pedalboard, Reverb, time_stretch # type: ignore
from pedalboard.io import AudioFile
from pydub import AudioSegment
import pygame
import re
from rich.console import Console
# from scipy.spatial import distance
from typing import Any, List, Optional

console = Console(highlight=False)
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))

with open("text_assets/generator_prompt.txt") as f:
    generator_prompt = f.read().strip()

with open("text_assets/one_off_selector_prompt.txt") as f:
    one_off_selector_prompt = f.read().strip()

with open("text_assets/topics.txt") as f:
    topics = f.read().splitlines()

SAYNEXTLINEEVENT = pygame.USEREVENT
PLAYSONGEVENT = pygame.USEREVENT + 1

pygame.init()
pygame.display.set_caption("Celeste's Christmas Catastrophe")
clock = pygame.time.Clock()

display_info = pygame.display.Info()
screen_size = (display_info.current_w, display_info.current_h)
screen = pygame.display.set_mode(screen_size)

pygame.font.init()
font = pygame.font.Font("fonts/OpenSans.ttf", 48)
bold_font = pygame.font.Font("fonts/OpenSans-Bold.ttf", 60)

pygame.mixer.init()
channel = pygame.mixer.Channel(0)


# i copied this from stackoverflow
def draw_wrapped_text(surface, text, rect):
    rect = pygame.Rect(rect)
    y = rect.top
    line_spacing = -2
    font_height = font.size("Tg")[1]
    while text:
        i = 1
        if y + font_height > rect.bottom:
            break
        while font.size(text[:i])[0] < rect.width and i < len(text):
            i += 1
        if i < len(text): 
            i = text.rfind(" ", 0, i) + 1
        image = font.render(text[:i], True, (255, 255, 255))
        surface.blit(image, (rect.left, y))
        y += font_height + line_spacing
        text = text[i:]
    return text


class Character:
    def __init__(self, name, sprite, actions, voice, effects=None, pitch=0):
        self.name = name
        self.sprite = pygame.image.load(f"image_assets/characters/{sprite}").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (int(self.sprite.get_width() * (screen_size[1] / self.sprite.get_height())), screen_size[1]))
        self.actions = actions
        self.voice = voice
        self.effects = effects if effects else Pedalboard([])
        self.pitch = pitch


class Location:
    def __init__(self, name, background, music_directory, volume):
        self.name = name
        self.background = pygame.image.load(f"image_assets/locations/{background}").convert()
        if self.background.get_width() / self.background.get_height() > screen_size[0] / screen_size[1]:
            self.background = pygame.transform.scale(self.background, (int(self.background.get_width() * (screen_size[1] / self.background.get_height())), screen_size[1]))
        else:
            self.background = pygame.transform.scale(self.background, (screen_size[0], int(self.background.get_height() * (screen_size[0] / self.background.get_width()))))
        self.songs = [f"music/{music_directory}/{song}" for song in os.listdir(f"music/{music_directory}")]
        self.volume = volume

    def play_song(self):
        pygame.mixer.music.load(random.choice(self.songs))
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent(PLAYSONGEVENT)


def reverb():
    return Reverb(room_size=0.5, dry_level=0.6)


characters = [
    Character("angel", sprite="angel.gif", actions=["freeze"], voice="shimmer", pitch=4),
    Character("depressed man in corner", sprite="depressed_man_in_corner.png", actions=["smoke"], voice="echo"),
    Character("elf", sprite="elf.png", actions=["walk", "fall over"], voice="fable"),
    Character("father", sprite="father.png", actions=["smile", "frown", "ascend"], voice="alloy"),
    Character("little girl", sprite="little_girl.png", actions=["flip", "frown"], voice="nova"),
    Character("Santa", sprite="santa.png",  actions=["menace", "hypermenace"], voice="onyx"),

    Character("mother", sprite="mother.png", actions=["smile", "fret"], voice="shimmer"),
]


def character_by_name(name):
    for character in characters:
        if character.name == name:
            return character


location_ps = [
    (Location("in Hell", background="hell.jpg", music_directory="hell", volume=0.22), 1),
    (Location("on a wooden platform amid the void", background="void.png", music_directory="void", volume=0.3), 2),
    (Location("in the little girl and father's living room", background="living_room.png", music_directory="christmas", volume=0.22), 7),
]
location_ps = [(location, p / sum(p for _, p in location_ps)) for location, p in location_ps]


def same_location(location):
    return location.name == "in the little girl and father's living room"


class Criterion:
    pass


@dataclass(match_args=True)
class Invert(Criterion):
    subcriterion: Criterion

    def __str__(self):
        return f"Invert({self.subcriterion})"


@dataclass(match_args=True)
class RandomChoice(Criterion):
    subcriteria: List[Criterion]
    ps: Optional[List[float]] = None

    def __str__(self):
        return f"RandomChoice([{', '.join(str(criterion) for criterion in self.subcriteria)}]{', ' + str(self.ps) if self.ps else ''})"


@dataclass(match_args=True)
class Longest(Criterion):
    def __str__(self):
        return "Longest"


@dataclass(match_args=True)
class OneOffSelector(Criterion):
    def __str__(self):
        return f"OneOffSelector"


# @dataclass(match_args=True)
# class RelevanceTo(Criterion):
#     target: str
# 
#     def __str__(self):
#         return f"RelevanceTo({self.target})"
# 
# 
# @dataclass(match_args=True)
# class RelevanceToTopic(Criterion):
#     def __str__(self):
#         return "RelevanceToTopic"


class Autoweaver:
    def __init__(self, n=None, criterion=None):
        self.n = self.generate_n() if n is None else n
        self.criterion = self.generate_criterion() if criterion is None else criterion

    def generate_n(self):
        n = round(random.lognormal(1.5, 1))
        if n < 1:
            n = 1
        elif n > 8:
            n = 8

        return n
    
    def generate_criterion(self):
        raise NotImplementedError

    def __str__(self):
        return f"n={self.n}, criterion={self.criterion}"


class Catastrophe:
    to_say = []

    generating = False
    complete = True

    currently_speaking = None
    currently_being_said = ""
    current_action = None

    def __init__(self, location_ps, autoweaver_generator=None):
        self.location_ps = location_ps
        self.autoweaver_generator = autoweaver_generator
        pygame.event.post(pygame.event.Event(SAYNEXTLINEEVENT))

    async def initialize_scene(self):
        self.autoweaver = self.autoweaver_generator() if self.autoweaver_generator else Autoweaver()
        self.location = random.choice([location for location, _ in self.location_ps], p=[p for _, p in self.location_ps])
        self.topic = random.choice(topics)

        console.log(f"Autoweaver: {self.autoweaver}")
        console.log(f"Location: {self.location.name}")
        console.log(f"Topic: {self.topic}")

        pygame.mixer.music.set_volume(self.location.volume)
        self.location.play_song()

        prompt = generator_prompt.split("{characters}")[0].format(
            location=self.location.name,
            same_location=" also" if same_location(self.location) else "",
            topic=self.topic,
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
                self.characters: List[Character] = sorted(characters, key=lambda character: str.casefold(character.name)) # type: ignore
                break

        console.log(f"Characters: {', '.join(character.name for character in self.characters)}")

        self.to_generate_from = []

    def character_by_name(self, name):
        return next(character for character in self.characters if character.name == name)

    def words_remaining(self):
        return sum(len(message["text"].split()) for message in self.to_say if "text" in message)

    def message_as_text(self, message):
        if "text" in message:
            return f"<{message['speaker']}> {message['text']}"
        elif "action" in message:
            return f"<{message['speaker']}> [{message['action']}]"
        else:
            raise ValueError(f"Invalid message: {message}")

    async def generate_next_line(self, begin=False, catch_up=False):
        if self.complete and not begin:
            return
        self.complete = False

        prompt = generator_prompt.format(
            characters=f" {', '.join(character.name for character in self.characters)}",
            location=self.location.name,
            same_location=" also" if same_location(self.location) else "",
            topic=self.topic,
        )
        for message in self.to_generate_from:
            prompt += f"\n{self.message_as_text(message)}"

        response = await openai.completions.create(
            model="gpt-4-base",
            prompt=prompt,
            max_tokens=1,
            temperature=1,
            top_p=0.96,
            stop=["\n\n"],
        )
        if response.choices[0].finish_reason == "stop":
            console.log("Scene is complete.")
            self.complete = True

        if not self.complete:
            prompt += "\n"

            speakers = []
            while True:
                response = await openai.completions.create(
                    model="gpt-4-base",
                    prompt=prompt,
                    n=self.autoweaver.n - len(speakers),
                    max_tokens=50,
                    temperature=1,
                    top_p=0.96,
                    stop=[">"],
                )

                speakers_from_response = [choice.text[1:] for choice in response.choices if choice.text.startswith("<")]
                speakers_from_response = [speaker for speaker in speakers_from_response if len([character for character in self.characters if character.name == speaker]) > 0]
                speakers += speakers_from_response

                if len(speakers) >= self.autoweaver.n:
                    break

            messages = [None for _ in range(self.autoweaver.n)]

            while True:
                remaining_indices = [i for i, message in enumerate(messages) if message is None]
                prompts = [prompt + f"<{speaker}>" for i, speaker in enumerate(speakers) if i in remaining_indices]

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
                    speaker = speakers[remaining_indices[i]]

                    if output.startswith("[") and output.endswith("]"):
                        action = output[1:-1]
                        if action in self.character_by_name(speaker).actions:
                            return {"speaker": speaker, "action": action}
                        else:
                            return
                    elif any(substring in output for substring in ["[", "]", "<", ">"]):
                        return
                    else:
                        return {"speaker": speaker, "text": output}

                for i in range(len(remaining_indices)):
                    messages[remaining_indices[i]] = process(i)

                if all(message is not None for message in messages):
                    break

            messages: List[Any] = messages

            async def evaluate_by(criterion):
                match criterion:
                    case Invert(subcriterion):
                        return [-score for score in await evaluate_by(subcriterion)]
                    case RandomChoice(subcriteria, ps):
                        return await evaluate_by(random.choice(subcriteria, p=ps)) # type: ignore
                    case Longest():
                        return [len(message["text"]) if "text" in message else 14 for message in messages]
                    case OneOffSelector():
                        # TODO: the autoweaver should be able to select the speaker
                        continuations = "\n".join([f"#{i + 1}: {self.message_as_text(message)}" for i, message in enumerate(messages)])
                        history = "\n".join([self.message_as_text(message) for message in self.to_generate_from])

                        prompt = one_off_selector_prompt.format(
                            characters=f" {', '.join(character.name for character in self.characters)}",
                            location=self.location.name,
                            same_location=" also" if same_location(self.location) else "",
                            topic=self.topic,

                            n_continuations=num2words(len(messages)) if len(messages) < 10 else len(messages),
                            continuations=continuations,
                            history=history,
                        )

                        while True:
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

                    # case RelevanceTo(target):
                    #     response = await openai.embeddings.create(
                    #         input=target,
                    #         model="text-embedding-ada-002",
                    #     )
                    #     target_embedding = response.data[0].embedding

                    #     message_body_values = [[value for value in body.values()][0] for body in message_bodies]

                    #     response = await openai.embeddings.create(
                    #         input=[prompt + body_value for body_value in message_body_values],
                    #         model="text-embedding-ada-002",
                    #     )
                    #     message_body_embeddings = [response.data[i].embedding for i in range(len(message_body_values))]
                    #     return [-distance.cosine(target_embedding, body_embedding) for body_embedding in message_body_embeddings]
                    # case RelevanceToTopic():
                    #     return await evaluate_by(RelevanceTo(self.topic))

                raise TypeError(f"Invalid criterion: {criterion}")
            
            scores = [float(score) for score in await evaluate_by(self.autoweaver.criterion)]
            message = messages[scores.index(max(scores))]

            id_ = nanoid.generate()

            self.to_say.append({**message, "id": id_})
            self.to_generate_from.append(message)

            asyncio.create_task(self.generate_audio_for(id_, catch_up=catch_up))

    async def generate_next_lines(self, n, begin=False, catch_up=False):
        if self.generating:
            return
        self.generating = True

        console.log(f"Generating {n} lines...")

        if begin:
            await self.generate_next_line(begin=True, catch_up=catch_up)
            n -= 1
        for i in range(n):
            await self.generate_next_line(catch_up=catch_up and i < n - 1)

        self.generating = False

    async def generate_audio_for(self, id_, catch_up=False):
        message = next(message for message in self.to_say if message["id"] == id_)

        if "text" in message:
            response = await openai.audio.speech.create(
                model="tts-1",
                voice=self.character_by_name(message["speaker"]).voice,
                input=message["text"],
            )

            audio_id = nanoid.generate()

            response.stream_to_file(f"tmp/{audio_id}.mp3")
            audio = AudioSegment.from_mp3(f"tmp/{audio_id}.mp3")
            audio.export(f"tmp/{audio_id}-pre.wav", format="wav")

            with AudioFile(f"tmp/{audio_id}-pre.wav") as f: # type: ignore
                with AudioFile(f"tmp/{audio_id}.wav", "w", f.samplerate, f.num_channels) as g: # type: ignore
                    pre = f.read(f.frames)
                    post_effects = self.character_by_name(message["speaker"]).effects(pre, f.samplerate, reset=False)
                    post = time_stretch(post_effects, f.samplerate, pitch_shift_in_semitones=self.character_by_name(message["speaker"]).pitch)
                    g.write(post)

            index = next(i for i, message in enumerate(self.to_say) if message["id"] == id_)
            self.to_say[index]["audio_id"] = audio_id

        if self.currently_speaking is None and not catch_up:
            pygame.event.post(pygame.event.Event(SAYNEXTLINEEVENT))

    async def say_next_line(self):
        try:
            id_ = self.to_say[0]["id"]
            is_action = "action" in self.to_say[0]
        except IndexError:
            self.currently_speaking = None
            return

        if not is_action:
            while "audio_id" not in next(message_ for message_ in self.to_say if message_["id"] == id_):
                self.currently_speaking = None
                await asyncio.sleep(0.1)

        message = self.to_say.pop(next(i for i, message in enumerate(self.to_say) if message["id"] == id_))

        self.currently_speaking = message["speaker"]
        console.log(f"{self.currently_speaking} is now speaking...")

        if is_action:
            self.current_action = message["action"]
            await asyncio.sleep(1.5)
            self.current_action = None
            await self.say_next_line()
            return
        
        self.currently_being_said = message["text"]

        audio_id = message["audio_id"]
        channel.play(pygame.mixer.Sound(f"tmp/{audio_id}.wav"))
        os.remove(f"tmp/{audio_id}-pre.wav")
        os.remove(f"tmp/{audio_id}.wav")
        channel.set_endevent(SAYNEXTLINEEVENT)

    async def run(self):
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == SAYNEXTLINEEVENT:
                    self.currently_speaking = None
                    self.currently_being_said = ""

                    if len(self.to_say) > 0:
                        asyncio.create_task(self.say_next_line())
                    elif self.complete:
                        await self.initialize_scene()
                        asyncio.create_task(self.generate_next_lines(5, begin=True, catch_up=True))

                elif event.type == PLAYSONGEVENT:
                    self.location.play_song()

                elif event.type == pygame.QUIT:
                    running = False

            if self.words_remaining() < 125 and not self.complete:
                asyncio.create_task(self.generate_next_lines(3))
            await asyncio.sleep(0)

            screen.blit(self.location.background, (0, 0))

            if self.currently_speaking is not None:
                sprite = self.character_by_name(self.currently_speaking).sprite
                sprite_rect = sprite.get_rect()
                sprite_rect.center = (round(screen_size[0] / 4), round(screen_size[1] / 2))
                screen.blit(sprite, sprite_rect)
        
                screen.blit(bold_font.render(self.currently_speaking, True, (255, 255, 255)), (50, 10))
        
                message_background = pygame.Surface((screen_size[0] * 0.6, screen_size[1])).convert_alpha()
                message_background.fill((0, 0, 0, 128))
                screen.blit(message_background, (screen_size[0] * 0.6, 0))
        
                draw_message = lambda text: draw_wrapped_text(screen, text, (screen_size[0] * 0.6 + 25, 25, screen_size[0] * 0.4 - 25, screen_size[1] - 100))

                if self.current_action is not None:
                    draw_message(f"[{self.current_action}]")
                else:
                    draw_message(self.currently_being_said)

            elif self.complete:
                screen.blit(bold_font.render("Loading new scene...", True, (255, 255, 255)), (50, 25))

            else:
                screen.blit(bold_font.render("Loading...", True, (255, 255, 255)), (50, 25))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


catastrophe = Catastrophe(location_ps=location_ps, autoweaver_generator=lambda: Autoweaver(n=5, criterion=OneOffSelector()))
asyncio.run(catastrophe.run())
