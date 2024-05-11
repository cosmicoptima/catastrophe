from archival import *
from core import *
from scene import *
import selection

import asyncio
import nanoid
from numpy import random
from openai import InternalServerError
import os
from pathlib import Path
from pedalboard import Pedalboard, time_stretch # type: ignore
from pedalboard.io import AudioFile
from pydub import AudioSegment
import pygame
import sys
import yaml

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
font = pygame.font.Font("data/fonts/OpenSans.ttf", w(0.015))
bold_font = pygame.font.Font("data/fonts/OpenSans-Bold.ttf", w(0.019))

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


async def tts_with_retry(**kwargs):
    while True:
        try:
            return await openai.audio.speech.create(**kwargs)
        except InternalServerError:
            console.log("Internal server error. Retrying...")


class DisplayableCharacter:
    def __init__(self, character, sprite, voice, effects=None, pitch=None):
        self.character = character

        self.sprite = pygame.image.load(f"data/sprites/{sprite}").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (int(self.sprite.get_width() * (screen_size[1] / self.sprite.get_height())), screen_size[1]))

        self.voice = voice
        self.effects = effects if effects is not None else Pedalboard([])
        self.pitch = pitch if pitch is not None else 0
    
    @classmethod
    def from_character(cls, name, sprite, voice, effects=None, pitch=0):
        return cls(character_by_name(name), sprite, voice, effects, pitch)
    
    @classmethod
    def from_dict(cls, d):
        return cls(character_by_name(d["name"]), d["sprite"], d["voice"], d.get("effects"), d.get("pitch"))


with open("data/characters.yaml") as f:
    displayable_characters = [DisplayableCharacter.from_dict(character) for character in yaml.safe_load(f)]


def displayable_character_by_name(name):
    for dc in displayable_characters:
        if dc.character.name == name:
            return dc


class DisplayableLocation:
    def __init__(self, name, background, music_directory, volume):
        self.name = name

        self.background = pygame.image.load(f"data/backgrounds/{background}").convert()
        if self.background.get_width() / self.background.get_height() > screen_size[0] / screen_size[1]:
            self.background = pygame.transform.scale(self.background, (int(self.background.get_width() * (screen_size[1] / self.background.get_height())), screen_size[1]))
        else:
            self.background = pygame.transform.scale(self.background, (screen_size[0], int(self.background.get_height() * (screen_size[0] / self.background.get_width()))))

        self.music_directory = music_directory
        entries = sorted(os.scandir(f"data/music/{music_directory}"), key=lambda entry: entry.stat().st_mtime, reverse=True)
        self.songs = [f"data/music/{music_directory}/{entry.name}" for entry in entries]
        self.volume = volume

        self.song_queue = []

    def play_next_song(self):
        if len(self.song_queue) == 0:
            if testing_music_for == self.music_directory:
                self.song_queue = [self.songs[0]]
            else:
                self.song_queue = list(random.permutation(self.songs))

        pygame.mixer.music.load(self.song_queue.pop(0))
        pygame.mixer.music.set_volume(self.volume)
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent(SONGEND)


with open("data/locations.yaml") as f:
    displayable_locations = [DisplayableLocation(**location) for location in yaml.safe_load(f)]


def displayable_location_by_name(name):
    for dl in displayable_locations:
        if dl.name == name:
            return dl


def dl_by_music_directory(music_directory):
    for dl in displayable_locations:
        if dl.music_directory == music_directory:
            return dl


class DisplayableMessage:
    def __init__(self, message, audio_id=None):
        self.message = message
        
        if audio_id is not None:
            self.audio = pygame.mixer.Sound(f"tmp/{audio_id}.wav")
            self.audio_id = audio_id
        elif self.message.type_ == MessageType.SPEECH and len(self.message.body) > 0:
            raise ValueError("audio_id must be provided if message is not an action")
    
    def play(self):
        if self.message.type_ == MessageType.ACTION:
            async def play_action():
                await asyncio.sleep(1.5)
                pygame.event.post(pygame.event.Event(MESSAGEEND))
            
            asyncio.create_task(play_action())
        elif self.message.type_ == MessageType.SPEECH:
            if len(self.message.body) > 0:
                channel.play(self.audio)
                channel.set_endevent(MESSAGEEND)
            else:
                async def play_nothing():
                    # DO NOT REMOVE
                    pygame.mixer.music.pause()
                    await asyncio.sleep(5)
                    pygame.mixer.music.unpause()
                    # DO NOT REMOVE
                    pygame.event.post(pygame.event.Event(MESSAGEEND))
                
                asyncio.create_task(play_nothing())


class Display:
    def __init__(self, selector, n, beam_length, beam_n, top_p, topic, replay=None):
        self.selector = selector
        self.n = n
        self.beam_length = beam_length
        self.beam_n = beam_n
        self.top_p = top_p
        self.topic = topic
        self.replay = replay
    
    async def create_audio_for(self, message):
        dc = displayable_character_by_name(message.speaker)

        response = await tts_with_retry(model="tts-1", voice=dc.voice, input=message.body)

        audio_id = nanoid.generate()

        response.stream_to_file(f"tmp/{audio_id}.mp3")
        audio_segment = AudioSegment.from_mp3(f"tmp/{audio_id}.mp3")
        audio_segment.export(f"tmp/{audio_id}-pre.wav", format="wav")

        with AudioFile(f"tmp/{audio_id}-pre.wav") as f:
            with AudioFile(f"tmp/{audio_id}.wav", "w", f.samplerate, f.num_channels) as g:
                audio = f.read(f.frames)
                audio = dc.effects(audio, f.samplerate, reset=False)
                audio = time_stretch(audio, f.samplerate, pitch_shift_in_semitones=dc.pitch)
                g.write(audio)
        
        os.remove(f"tmp/{audio_id}.mp3")
        os.remove(f"tmp/{audio_id}-pre.wav")

        return audio_id

    async def main_loop(self):
        if self.replay is not None:
            el = displayable_location_by_name(self.replay.location)
        else:
            el = displayable_location_by_name(self.scene.data.location)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MESSAGEEND:
                    if self.current_em.message.type_ == MessageType.SPEECH:
                        os.remove(f"tmp/{self.current_em.audio_id}.wav")
                    self.current_em = None
                elif event.type == SONGEND:
                    el.play_next_song()
            
            if self.replay is None and self.scene.data.is_complete:
                self.frontloading = None

            if self.current_em is None and self.frontloading is None:
                try:
                    self.current_em = self.display_queue.get_nowait()
                    self.current_em.play()
                except asyncio.QueueEmpty:
                    self.frontloading = 10

            render_on_top_left = lambda text: screen.blit(bold_font.render(text, True, (255, 255, 255)), (w(0.015), w(0.0075)))

            def render_on_bottom_left(text):
                (_, height) = bold_font.size(text)
                screen.blit(bold_font.render(text, True, (255, 255, 255)), (w(0.015), h(1) - w(0.0075) - height))

            screen.blit(el.background, (0, 0))

            if self.current_em is not None:
                dc = displayable_character_by_name(self.current_em.message.speaker)

                ec_sprite_rect = dc.sprite.get_rect()
                ec_sprite_rect.center = (w(0.25), h(0.5))
                screen.blit(dc.sprite, ec_sprite_rect)
        
                render_on_top_left(self.current_em.message.speaker)
        
                body_underlay = pygame.Surface((w(0.6), h(1))).convert_alpha()
                body_underlay.fill((0, 0, 0, 128))
                screen.blit(body_underlay, (w(0.6), 0))
        
                render_body = lambda text: draw_wrapped_text(screen, text, (w(0.6075), w(0.0075), w(0.3925), h(1) - w(0.03)))

                if self.current_em.message.type_ == MessageType.ACTION:
                    render_body(f"[{self.current_em.message.body}]")
                else:
                    render_body(self.current_em.message.body)
            
            elif self.replay is None and self.scene.data.is_complete:
                render_on_top_left("Scene complete!")

            elif self.frontloading is not None:
                render_on_top_left(f"Loading; {self.frontloading} remaining...")

            else:
                render_on_top_left("Loading...")

            render_on_bottom_left(self.text_being_displayed)

            pygame.display.flip()
            clock.tick(60)
            await asyncio.sleep(0)
    
    async def run(self):
        self.tts_queue = asyncio.Queue()
        self.display_queue = asyncio.Queue()

        self.current_em = None
        self.frontloading = 6

        self.text_being_displayed = ""

        if self.replay is None:
            if testing_music_for is None:
                location_generator = LocationPs(location_ps)
            else:
                location_generator = ConstantLocation(dl_by_music_directory(testing_music_for).name)

            self.scene = await Scene.create(
                characters_generator=ChooseCharactersFrom([dc.character for dc in displayable_characters]),
                location_generator=location_generator,
                selector=self.selector,
                n=self.n,
                beam_length=self.beam_length,
                beam_n=self.beam_n,
                top_p=self.top_p,
                topic=self.topic,
            )

            self.scene_filename = make_filename(self.scene)

            el = displayable_location_by_name(self.scene.data.location)
        else:
            self.scene = None
            self.scene_filename = None

            el = displayable_location_by_name(self.replay.location)

        el.play_next_song()

        async def write():
            iterator = self.replay.replay() if self.replay is not None else self.scene.write()

            async for message in iterator:
                self.tts_queue.put_nowait(message)

                if self.replay is None:
                    save(self.scene, self.scene_filename)

        async def create_audio():
            while True:
                message = await self.tts_queue.get()

                if message.type_ == MessageType.SPEECH and len(message.body) > 0:
                    self.display_queue.put_nowait(DisplayableMessage(message, audio_id=await self.create_audio_for(message)))
                else:
                    self.display_queue.put_nowait(DisplayableMessage(message))
                
                if self.frontloading is not None:
                    self.frontloading -= 1
                if self.frontloading == 0:
                    self.frontloading = None

        await asyncio.gather(write(), create_audio(), self.main_loop())


async def display_main():
    with open("scene_options.yaml") as f:
        config = yaml.safe_load(f)

    global testing_music_for
    testing_music_for = config["testing_music_for"] if "testing_music_for" in config else None

    if "selector" in config:
        selector = selection.from_text(config["selector"])
    else:
        raise ValueError("selector must be provided in scene_options.yaml")

    if "n" in config:
        n = config["n"]
    else:
        raise ValueError("n must be provided in scene_options.yaml")

    if "beam_length" in config:
        beam_length = config["beam_length"]
    else:
        raise ValueError("beam_length must be provided in scene_options.yaml")
    
    if "beam_n" in config:
        beam_n = config["beam_n"]
    else:
        raise ValueError("beam_n must be provided in scene_options.yaml")
    
    if "top_p" in config:
        top_p = config["top_p"]
    else:
        raise ValueError("top_p must be provided in scene_options.yaml")

    topic = config["topic"] if "topic" in config else None
    replay = load(config["replay"]) if "replay" in config else None

    await Display(
        selector=selector,
        n=n,
        beam_length=beam_length,
        beam_n=beam_n,
        top_p=top_p,
        topic=topic,
        replay=replay
    ).run()


if __name__ == "__main__":
    asyncio.run(display_main())
