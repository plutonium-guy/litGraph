from enum import Enum
from typing import Any, Mapping

class StreamPart(str, Enum):
    DELTA: StreamPart
    TOOL_CALL_DELTA: StreamPart
    DONE: StreamPart
    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> StreamPart: ...

def stream_part(event: Mapping[str, Any]) -> StreamPart: ...
