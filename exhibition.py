from scene import *

import asyncclick as click
import asyncio
import nanoid
from numpy import random
import os
from pathlib import Path
from pedalboard import Pedalboard, time_stretch
from pedalboard.io import AudioFile
from pydub import AudioSegment
import pygame
import sys

Path("./tmp").mkdir(exist_ok=True)

MESSAGEEND = pygame.USEREVENT
SONGEND = pygame.USEREVENT + 1

pygame.init()
pygame.display.set_caption("Celeste's Christmas Catastrophe")
clock = pygame.time.Clock()

display_info = pygame.display.Info()
screen_size = (display_info.current_w, display_info.current_h)
screen = pygame.display.set_mode(screen_size)

h = lambda percentile: round(display_info.current_h * percentile)
w = lambda percentile: round(display_info.current_w * percentile)

pygame.font.init()
font = pygame.font.Font("assets/fonts/OpenSans.ttf", w(0.015))
bold_font = pygame.font.Font("assets/fonts/OpenSans-Bold.ttf", w(0.019))

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


class ExhibitableCharacter:
    def __init__(self, character, sprite, voice, effects=None, pitch=0):
        self.character = character

        self.sprite = pygame.image.load(f"assets/sprites/{sprite}").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (int(self.sprite.get_width() * (screen_size[1] / self.sprite.get_height())), screen_size[1]))

        self.voice = voice
        self.effects = effects if effects is not None else Pedalboard([])
        self.pitch = pitch
    
    @classmethod
    def from_character(cls, name, sprite, voice, effects=None, pitch=0):
        return cls(character_by_name(name), sprite, voice, effects, pitch)


exhibitable_characters = [
    ExhibitableCharacter.from_character("angel", sprite="angel.gif", voice="shimmer", pitch=4),
    ExhibitableCharacter.from_character("depressed man in corner", sprite="depressed_man_in_corner.png", voice="echo"),
    ExhibitableCharacter.from_character("elf", sprite="elf.png", voice="fable"),
    ExhibitableCharacter.from_character("father", sprite="father.png", voice="alloy"),
    ExhibitableCharacter.from_character("little girl", sprite="little_girl.png", voice="nova"),
    ExhibitableCharacter.from_character("Santa", sprite="santa.png", voice="onyx"),

    ExhibitableCharacter.from_character("mother", sprite="mother.png", voice="shimmer"),
]


def exhibitable_character_by_name(name):
    for ec in exhibitable_characters:
        if ec.character.name == name:
            return ec


class ExhibitableLocation:
    def __init__(self, name, background, music_directory, volume):
        self.name = name

        self.background = pygame.image.load(f"assets/backgrounds/{background}").convert()
        if self.background.get_width() / self.background.get_height() > screen_size[0] / screen_size[1]:
            self.background = pygame.transform.scale(self.background, (int(self.background.get_width() * (screen_size[1] / self.background.get_height())), screen_size[1]))
        else:
            self.background = pygame.transform.scale(self.background, (screen_size[0], int(self.background.get_height() * (screen_size[0] / self.background.get_width()))))

        self.songs = [f"assets/music/{music_directory}/{song}" for song in os.listdir(f"assets/music/{music_directory}")]
        self.volume = volume

        self.song_queue = []

    def play_next_song(self):
        if len(self.song_queue) == 0:
            self.song_queue = list(random.permutation(self.songs))

        pygame.mixer.music.load(self.song_queue.pop(0))
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent(SONGEND)


exhibitable_locations = [
    ExhibitableLocation("in Hell", background="hell.jpg", music_directory="hell", volume=0.22),
    ExhibitableLocation("on a wooden platform amid the void", background="void.png", music_directory="void", volume=0.3),
    ExhibitableLocation("in the little girl and father's living room", background="living_room.png", music_directory="christmas", volume=0.22),
]


def exhibitable_location_by_name(name):
    for el in exhibitable_locations:
        if el.name == name:
            return el


location_ps = [
    ("in Hell", 1),
    ("on a wooden platform amid the void", 2),
    ("in the little girl and father's living room", 7),
]


class ExhibitableMessage:
    def __init__(self, message, audio_id=None):
        self.message = message
        
        if audio_id is not None:
            self.audio = pygame.mixer.Sound(f"tmp/{audio_id}.wav")
            self.audio_id = audio_id
        else:
            if not self.message.is_action:
                raise ValueError("audio_id must be provided if message is not an action")
    
    def play(self):
        if self.message.is_action:
            async def play_action():
                await asyncio.sleep(1.5)
                pygame.event.post(pygame.event.Event(MESSAGEEND))
            
            asyncio.create_task(play_action())
        else:
            channel.play(self.audio)
            channel.set_endevent(MESSAGEEND)


class Exhibition:
    def __init__(self, criterion, n, topic=None):
        self.criterion = criterion
        self.n = n
        self.topic = topic
    
    async def create_audio_for(self, message):
        ec = exhibitable_character_by_name(message.speaker)

        response = await openai.audio.speech.create(model="tts-1", voice=ec.voice, input=message.body)

        audio_id = nanoid.generate()

        response.stream_to_file(f"tmp/{audio_id}.mp3")
        audio_segment = AudioSegment.from_mp3(f"tmp/{audio_id}.mp3")
        audio_segment.export(f"tmp/{audio_id}-pre.wav", format="wav")

        with AudioFile(f"tmp/{audio_id}-pre.wav") as f:
            with AudioFile(f"tmp/{audio_id}.wav", "w", f.samplerate, f.num_channels) as g:
                audio = f.read(f.frames)
                audio = ec.effects(audio, f.samplerate, reset=False)
                audio = time_stretch(audio, f.samplerate, pitch_shift_in_semitones=ec.pitch)
                g.write(audio)
        
        os.remove(f"tmp/{audio_id}.mp3")
        os.remove(f"tmp/{audio_id}-pre.wav")

        return audio_id

    async def main_loop(self):
        el = exhibitable_location_by_name(self.scene.data.location)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MESSAGEEND:
                    os.remove(f"tmp/{self.current_em.audio_id}.wav")
                    self.current_em = None
                elif event.type == SONGEND:
                    el.play_next_song()

            if self.current_em is None and self.frontloading is None:
                try:
                    self.current_em = self.exhibition_queue.get_nowait()
                    self.current_em.play()
                except asyncio.QueueEmpty:
                    self.frontloading = 10

            screen.blit(el.background, (0, 0))

            render_on_top_left = lambda text: screen.blit(bold_font.render(text, True, (255, 255, 255)), (w(0.015), w(0.0075)))

            if self.current_em is not None:
                ec = exhibitable_character_by_name(self.current_em.message.speaker)

                ec_sprite_rect = ec.sprite.get_rect()
                ec_sprite_rect.center = (w(0.25), h(0.5))
                screen.blit(ec.sprite, ec_sprite_rect)
        
                render_on_top_left(self.current_em.message.speaker)
        
                body_underlay = pygame.Surface((w(0.6), h(1))).convert_alpha()
                body_underlay.fill((0, 0, 0, 128))
                screen.blit(body_underlay, (w(0.6), 0))
        
                render_body = lambda text: draw_wrapped_text(screen, text, (w(0.6075), w(0.0075), w(0.3925), h(1) - w(0.03)))

                if self.current_em.message.is_action:
                    render_body(f"[{self.current_em.message.body}]")
                else:
                    render_body(self.current_em.message.body)

            elif self.frontloading is not None:
                render_on_top_left(f"Loading; {self.frontloading} remaining...")

            else:
                render_on_top_left("Loading...")

            pygame.display.flip()
            clock.tick(60)
            await asyncio.sleep(0)
    
    async def run(self):
        self.tts_queue = asyncio.Queue()
        self.exhibition_queue = asyncio.Queue()

        self.current_em = None
        self.frontloading = 10

        self.scene = await Scene.create(
            characters_generator=ChooseCharactersFrom([ec.character for ec in exhibitable_characters]),
            location_generator=LocationPs(location_ps),
            criterion=self.criterion,
            n=self.n,
            topic=self.topic if self.topic is not None else random.choice(topics),
        )

        el = exhibitable_location_by_name(self.scene.data.location)
        pygame.mixer.music.set_volume(el.volume)
        el.play_next_song()

        async def write():
            async for message in self.scene.write():
                self.tts_queue.put_nowait(message)
        
        async def create_audio():
            while True:
                message = await self.tts_queue.get()

                if message.is_action:
                    self.exhibition_queue.put_nowait(ExhibitableMessage(message))
                else:
                    self.exhibition_queue.put_nowait(ExhibitableMessage(message, audio_id=await self.create_audio_for(message)))
                
                if self.frontloading is not None:
                    self.frontloading -= 1
                if self.frontloading == 0:
                    self.frontloading = None
        
        await asyncio.gather(write(), create_audio(), self.main_loop())


@click.command()
@click.option("--topic")
async def main(topic):
    await Exhibition(criterion=Longest(), n=5, topic=topic).run()


if __name__ == "__main__":
    asyncio.run(main())
