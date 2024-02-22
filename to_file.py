from scene import *

import asyncclick as click
import asyncio
from dataclasses import dataclass
from numpy import random
import orjson
from pathlib import Path
from time import strftime


def make_filename(scene):
    return f"{strftime('%y%m%d%H%M%S')}-{scene.data.topic}"


def export(scene, filename):
    data = {
        "messages": [{"speaker": message.speaker, "body": message.body, "type": str(message.type_)} for message in scene.data.messages],
        "location": scene.data.location,
        "topic": scene.data.topic,
    }
    Path("./output").mkdir(exist_ok=True)
    with open(f"./output/{filename}.json", "wb") as f:
        f.write(orjson.dumps(data))


@dataclass
class PreservedScene:
    location: str
    messages: List[Message]

    async def replay(self):
        for message in self.messages:
            yield message


def import_(full_filename):
    with open(full_filename, "rb") as f:
        data = orjson.loads(f.read())
    
    return PreservedScene(location=data["location"], messages=[Message(speaker=message["speaker"], body=message["body"], type_=MessageType(message["type"])) for message in data["messages"]])


@click.command()
async def to_file_main():
    scene = await Scene.create(
        characters_generator=ChooseCharactersFrom([character.name for character in characters]),
        location_generator=LocationPs(location_ps),
        criterion=Longest(),
        n=1,
        topic=random.choice(topics)
    )

    async for message in scene.write():
        console.log(str(message))

    export(scene, make_filename(scene))


if __name__ == "__main__":
    asyncio.run(to_file_main())
