from scene import *

from dataclasses import dataclass
import orjson
from pathlib import Path
from time import strftime


@dataclass
class PreservedScene:
    location: str
    messages: List[Message]

    async def replay(self):
        for message in self.messages:
            yield message


def make_filename(scene):
    return f"{strftime('%y%m%d%H%M%S')}-{scene.data.topic}"


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