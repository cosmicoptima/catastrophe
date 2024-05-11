from scene import *
import selection

import asyncclick as click
from dataclasses import dataclass
import orjson
from pathlib import Path
import random
from time import strftime


@dataclass
class PreservedScene:
    location: str
    messages: List[Message]

    async def replay(self):
        for message in self.messages:
            yield message


def make_filename(scene):
    return f"{strftime('%y%m%d%H%M%S')}-{random.randrange(100000):06}-{scene.data.topic}"


def save(scene, filename):
    data = {
        "messages": [{"speaker": message.speaker, "body": message.body, "type": str(message.type_)} for message in scene.data.messages],
        "location": scene.data.location,
        "topic": scene.data.topic,
    }
    Path("output").mkdir(exist_ok=True)
    with open(f"output/{filename}.json", "wb") as f:
        f.write(orjson.dumps(data))


def load(full_filename):
    with open(full_filename, "rb") as f:
        data = orjson.loads(f.read())
    
    return PreservedScene(location=data["location"], messages=[Message(speaker=message["speaker"], body=message["body"], type_=MessageType(message["type"])) for message in data["messages"]])


scenes_complete = 0


async def write_scene():
    with open("scene_options.yaml") as f:
        config = yaml.safe_load(f)

    if "selector" in config:
        selector = selection.from_text(config["selector"])
    else:
        raise ValueError("selector must be provided in scene_options.yaml")

    if "n" in config:
        n = config["n"]
    else:
        raise ValueError("n must be provided in scene_options.yaml")
    
    if "top_p" in config:
        top_p = config["top_p"]
    else:
        raise ValueError("top_p must be provided in scene_options.yaml")

    topic = config["topic"] if "topic" in config else None
    additional_topic = config["additional_topic"] if "additional_topic" in config else None

    scene = await Scene.create(
        characters_generator=ChooseCharactersFrom(characters),
        location_generator=LocationPs(location_ps),
        selector=selector,
        n=n,
        top_p=top_p,
        topic=topic,
        additional_topic=additional_topic,
    )
    filename = make_filename(scene)
    
    global scenes_complete

    async for message in scene.write():
        part_1, part_2 = str(message).split(">", 1)
        message = f"[blue]{part_1}>[/blue]{part_2}"

        console.log(f"[violet]Wrote message:[/violet] {message}")
        console.log(f"[blue]{scenes_complete}[/blue] [violet]scenes complete.[/violet]")
        save(scene, filename)

    scenes_complete += 1


@click.command()
@click.argument("n", type=int)
async def archival_main(n):
    await asyncio.gather(*[write_scene() for _ in range(n)])


if __name__ == "__main__":
    asyncio.run(archival_main())