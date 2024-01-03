from dotenv import load_dotenv
load_dotenv()

import asyncio
import nanoid
from numpy import random
from openai import AsyncOpenAI
import os
from pydub import AudioSegment
import pygame
from rich.console import Console

console = Console(highlight=False)
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))

with open("text_assets/prompt_template.txt") as f:
    prompt_template = f.read().strip()

with open("text_assets/topics.txt") as f:
    topics = f.read().splitlines()

songs = os.listdir("music")

SAYNEXTLINEEVENT = pygame.USEREVENT
PLAYSONGEVENT = pygame.USEREVENT + 1

pygame.init()
pygame.display.set_caption("Celeste's Christmas Catastrophe")
clock = pygame.time.Clock()

display_info = pygame.display.Info()
screen_size = (display_info.current_w, display_info.current_h)
screen = pygame.display.set_mode(screen_size)

background = pygame.image.load("image_assets/background.png").convert()
background = pygame.transform.scale(background, screen_size)

pygame.font.init()
font = pygame.font.Font("assets/OpenSans.ttf", 48)
bold_font = pygame.font.Font("assets/OpenSans-Bold.ttf", 60)

pygame.mixer.init()
channel = pygame.mixer.Channel(0)

pygame.mixer.music.set_volume(0.14)
pygame.mixer.music.load(f"music/{random.choice(songs)}")
pygame.mixer.music.play()
pygame.mixer.music.set_endevent(PLAYSONGEVENT)


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
    def __init__(self, name, sprite, p, actions, voice, description=None):
        self.name = name
        self.sprite = pygame.image.load(f"image_assets/{sprite}").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (int(self.sprite.get_width() * (screen_size[1] / self.sprite.get_height())), screen_size[1]))
        self.p = p
        self.actions = actions
        self.voice = voice
        self.description = description

    def describe(self):
        return f"- {self.name}{f' ({self.description})' if self.description else ''} [{', '.join(self.actions)}]"


characters = [
    Character("Santa", sprite="santa.png", p=1, actions=["menace", "hypermenace"], voice="onyx"),
    Character("little girl", sprite="little_girl.png", p=1, actions=["flip", "frown"], voice="nova"),
    Character("depressed man in corner", sprite="depressed_man_in_corner.png", p=1, actions=["smoke"], voice="echo", description="does not talk"),
    Character("elf", sprite="elf.png", p=1, actions=["walk", "fall over"], voice="fable"),
    Character("father", sprite="father.png", p=1, actions=["smile", "frown", "ascend"], voice="alloy"),
    Character("angel", sprite="angel.gif", p=1, actions=["freeze"], voice="shimmer"),
]


class Catastrophe:
    to_say = []

    generating = False
    complete = True

    currently_speaking = None
    currently_being_said = ""
    current_action = None

    def __init__(self, characters):
        self.characters = characters
        pygame.event.post(pygame.event.Event(SAYNEXTLINEEVENT))

    def initialize_scene(self):
        self.topic = random.choice(topics)
        self.to_generate_from = []

    def character_by_name(self, name):
        return next(character for character in self.characters if character.name == name)

    def words_remaining(self):
        return sum(len(message["text"].split()) for message in self.to_say if "text" in message)

    async def generate_next_line(self, begin=False, catch_up=False):
        if self.complete and not begin:
            return
        self.complete = False

        prompt = prompt_template.format(topic=self.topic)
        for message in self.to_generate_from:
            if "action" in message:
                prompt += f"\n{message['speaker']} [{message['action']}]"
            else:
                prompt += f"\n<{message['speaker']}> {message['text']}"

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

            while True:
                response = await openai.completions.create(
                    model="gpt-4-base",
                    prompt=prompt,
                    max_tokens=50,
                    temperature=1,
                    top_p=0.96,
                    stop=[">"],
                )
                if not response.choices[0].text.startswith("<"):
                    continue
                speaker = response.choices[0].text[1:]
                if len([character for character in characters if character.name == speaker]) > 0:
                    message = {"speaker": speaker}
                    break

            prompt += f"<{speaker}>"

            while True:
                response = await openai.completions.create(
                    model="gpt-4-base",
                    prompt=prompt,
                    max_tokens=100,
                    temperature=1.06,
                    top_p=0.96,
                    stop=["\n"],
                )
                if response.choices[0].finish_reason == "length":
                    continue
                output = response.choices[0].text[1:]
                
                if output.startswith("[") and output.endswith("]"):
                    action = output[1:-1]
                    if action in self.character_by_name(speaker).actions:
                        message["action"] = action
                        break
                    else:
                        continue
                elif "[" in output or "]" in output:
                    continue
                else:
                    message["text"] = output
                    break

            if "text" in message:
                response = await openai.audio.speech.create(
                    model="tts-1",
                    voice=self.character_by_name(message["speaker"]).voice,
                    input=message["text"],
                    speed=0.85,
                )
                audio_id = nanoid.generate()
                message["audio_id"] = audio_id
                response.stream_to_file(f"tmp/{audio_id}.mp3")
                audio = AudioSegment.from_mp3(f"tmp/{audio_id}.mp3")
                audio.export(f"tmp/{audio_id}.wav", format="wav")

            self.to_say.append(message)
            self.to_generate_from.append(message)

        if self.currently_speaking is None and not catch_up:
            pygame.event.post(pygame.event.Event(SAYNEXTLINEEVENT))

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

    async def say_next_line(self):
        try:
            message = self.to_say.pop(0)
        except IndexError:
            self.currently_speaking = None
            return

        self.currently_speaking = message["speaker"]
        console.log(f"{self.currently_speaking} is now speaking...")

        if "action" in message:
            self.current_action = message["action"]
            await asyncio.sleep(2)
            self.current_action = None
            await self.say_next_line()
            return
        
        self.currently_being_said = message["text"]

        audio_id = message["audio_id"]
        channel.play(pygame.mixer.Sound(f"tmp/{audio_id}.wav"))
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
                        self.initialize_scene()
                        asyncio.create_task(self.generate_next_lines(5, begin=True, catch_up=True))

                elif event.type == PLAYSONGEVENT:
                    pygame.mixer.music.load(f"music/{random.choice(songs)}")
                    pygame.mixer.music.play()
                    pygame.mixer.music.set_endevent(PLAYSONGEVENT)

                elif event.type == pygame.QUIT:
                    running = False

            if self.words_remaining() < 50 and not self.complete:
                asyncio.create_task(self.generate_next_lines(3))
            await asyncio.sleep(0)
        
            screen.blit(background, (0, 0))

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
                screen.blit(bold_font.render("Loading new scene...", True, (255, 255, 255)), (50, 10))

            else:
                screen.blit(bold_font.render("Loading...", True, (255, 255, 255)), (50, 10))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


catastrophe = Catastrophe(characters=characters)
asyncio.run(catastrophe.run())
