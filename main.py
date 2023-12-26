from dotenv import load_dotenv
load_dotenv()

import asyncio
from gtts import gTTS
import nanoid
from numpy import random
from openai import AsyncOpenAI
import os
from pydub import AudioSegment
import pygame
from regex import fullmatch
from rich.console import Console

# set up variables

console = Console(highlight=False)
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))

with open("text_assets/prompt_template.txt") as f:
    prompt_template = f.read().strip()

with open("text_assets/topics.txt") as f:
    topics = f.read().splitlines()

songs = os.listdir("music")
random.shuffle(songs)

screen_size = (1920, 1080)

# set up pygame

pygame.init()
pygame.display.set_caption("Celeste's Christmas Catastrophe")
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()

background = pygame.image.load("image_assets/background.png").convert()
background = pygame.transform.scale(background, screen_size)

pygame.font.init()
font = pygame.font.Font("assets/OpenSans.ttf", 48)
bold_font = pygame.font.Font("assets/OpenSans-Bold.ttf", 60)

pygame.mixer.init()
channel = pygame.mixer.Channel(0)

pygame.mixer.music.set_volume(0.125)
pygame.mixer.music.load(f"music/{songs[0]}")
pygame.mixer.music.play()
for song in songs[1:]:
    pygame.mixer.music.queue(f"music/{song}")

class Character:
    def __init__(self, name, sprite, measure, actions, description=None, centrality=1):
        self.name = name
        self.sprite = pygame.image.load(f"image_assets/{sprite}").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (int(self.sprite.get_width() * (screen_size[1] / self.sprite.get_height())), screen_size[1]))
        self.measure = measure
        self.actions = actions
        self.description = description
        self.centrality = centrality

    def describe(self):
        return f"- {self.name}{f' ({self.description})' if self.description else ''} [{', '.join(self.actions)}]"

characters = [
    Character("Santa", sprite="santa.png", measure=1000, actions=["menace", "hypermenace"]),
    Character("little girl", sprite="little_girl.png", measure=200, actions=["flip", "frown"]),
    Character("depressed man in corner", sprite="depressed_man_in_corner.png", measure=12, actions=["smoke"], description="does not talk"),
    Character("elf", sprite="elf.png", measure=8, actions=["walk", "fall over"]),
    Character("father", sprite="father.png", measure=8, actions=["smile", "frown", "ascend"]),
    Character("angel", sprite="angel.gif", measure=4, actions=["freeze"]),
    Character("Decorus", sprite="decorus.png", measure=4, actions=["glare", "blush"], centrality=25),
    Character("Secundus", sprite="secundus.png", measure=2, actions=["cry"], centrality=25),
    Character("Serpens", sprite="serpens.png", measure=2, actions=["slither", "shift"], centrality=25),
    Character("Arcanus", sprite="arcanus.png", measure=2, actions=["invert", "multiply"], centrality=25),
]

def get_character(name):
    return next(character for character in characters if character.name == name)

# get and parse response

def make_prompt():
    character_list = [character for character in characters if 1 / character.measure ** 0.5 < random.random()]
    character_list = sorted(character_list, key=lambda character: random.lognormal(character.measure * character.centrality, 1.5), reverse=True)
    character_list = "\n".join(character.describe() for character in character_list)

    global prompt, to_generate_from

    prompt = prompt_template.format(topic=random.choice(topics), characters=character_list)
    to_generate_from = prompt

make_prompt()
to_say = []
generating = False
complete = False

currently_speaking = None

async def get_next_lines(begin=False):
    global complete, generating, to_say, to_generate_from
    if generating:
        return
    if complete and not begin:
        return
    generating = True

    console.log("Generating new lines...")

    response = await openai.completions.create(
        model="gpt-4-base",
        prompt=to_generate_from,
        max_tokens=200,
        temperature=1,
        top_p=0.96,
        stop=["\n\n"],
    )

    response_text = response.choices[0].text.strip()
    if response.choices[0].finish_reason == "length":
        response_text = response_text.rsplit("\n", 1)[0]
        to_generate_from += response_text + "\n"
    elif response.choices[0].finish_reason == "stop":
        complete = True
    
    response_lines = response_text.splitlines()
    for line in response_lines:
        match = fullmatch(r"\<(?<speaker>.+)\> (?:\[(?<action>.+)\]|(?<text>.+))", line)
        if match is None:
            console.log(f"Invalid line: {line}")
            break
        
        message = {"speaker": match.group("speaker")}
        if match.group("action"):
            message["action"] = match.group("action")
        else:
            message["text"] = match.group("text")
    
        to_say.append(message)

    generating = False
    if begin:
        complete = False

    if currently_speaking is None:
        pygame.event.post(pygame.event.Event(pygame.USEREVENT))

    console.log(f"Words remaining: {words_remaining()}")
    if complete:
        console.log("Scene is complete.")

def words_remaining():
    global to_say
    return sum(len(message["text"].split()) for message in to_say if "text" in message)

# set up audio

currently_being_said = ""
current_action = None

async def say(message):
    global currently_speaking, currently_being_said, current_action
    currently_speaking = message["speaker"]

    if "text" not in message:
        current_action = message["action"]
        await asyncio.sleep(3)
        current_action = None
        if len(to_say) > 0:
            await say(to_say.pop(0))
        return

    text = message["text"]
    currently_being_said = text

    audio_id = nanoid.generate()
    gTTS(text, lang="en").save(f"tmp/{audio_id}.mp3")
    audio = AudioSegment.from_mp3(f"tmp/{audio_id}.mp3")
    audio.speedup(1.25).export(f"tmp/{audio_id}.wav", format="wav")

    channel.play(pygame.mixer.Sound(f"tmp/{audio_id}.wav"))
    channel.set_endevent(pygame.USEREVENT)

# text wrap function

def draw_wrapped_text(surface, text, rect):
    rect = pygame.Rect(rect)
    y = rect.top
    lineSpacing = -2

    # get the height of the font
    fontHeight = font.size("Tg")[1]

    while text:
        i = 1

        # determine if the row of text will be outside our area
        if y + fontHeight > rect.bottom:
            break

        # determine maximum width of line
        while font.size(text[:i])[0] < rect.width and i < len(text):
            i += 1

        # if we've wrapped the text, then adjust the wrap to the last word      
        if i < len(text): 
            i = text.rfind(" ", 0, i) + 1

        # render the line and blit it to the surface
        image = font.render(text[:i], True, (255, 255, 255))

        surface.blit(image, (rect.left, y))
        y += fontHeight + lineSpacing

        # remove the text we just blitted
        text = text[i:]

    return text

# main loop

async def main():
    global currently_speaking, currently_being_said, current_action, to_say, complete

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:
                if len(to_say) > 0:
                    asyncio.create_task(say(to_say.pop(0)))
                else:
                    currently_speaking = None
                    currently_being_said = ""

                    if complete:
                        make_prompt()
                        to_say = []
                        asyncio.create_task(get_next_lines(begin=True))
            elif event.type == pygame.QUIT:
                running = False

        if words_remaining() < 100:
            asyncio.create_task(get_next_lines())
        await asyncio.sleep(0)
    
        screen.blit(background, (0, 0))

        if currently_speaking is not None:
            rect = get_character(currently_speaking).sprite.get_rect()
            rect.center = (round(screen_size[0] / 4), round(screen_size[1] / 2))
            screen.blit(get_character(currently_speaking).sprite, rect)
    
            screen.blit(bold_font.render(currently_speaking, True, (255, 255, 255)), (50, 10))
    
            right_tint = pygame.Surface((screen_size[0] * 0.6, screen_size[1])).convert_alpha()
            right_tint.fill((0, 0, 0, 128))
            screen.blit(right_tint, (screen_size[0] * 0.6, 0))
    
            if current_action is not None:
                draw_wrapped_text(screen, f"[{current_action}]", (screen_size[0] * 0.6 + 25, 25, screen_size[0] * 0.4 - 25, screen_size[1] - 100))
            else:
                draw_wrapped_text(screen, currently_being_said, (screen_size[0] * 0.6 + 25, 25, screen_size[0] * 0.4 - 25, screen_size[1] - 100))
        elif complete or len(to_say) == 0:
            screen.blit(bold_font.render("Loading new scene...", True, (255, 255, 255)), (50, 10))
        else:
            screen.blit(bold_font.render("Loading...", True, (255, 255, 255)), (50, 10))

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()

asyncio.run(main())
