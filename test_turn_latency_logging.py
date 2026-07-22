import contextlib
import io
from types import SimpleNamespace

from agent_v2 import Aiva
from livekit.agents.llm import ChatMessage


def _aiva():
    aiva = Aiva.__new__(Aiva)
    aiva.current_stage = "opener"
    aiva._user_turn_count = 0
    aiva._call_ending = False
    return aiva


def _check():
    aiva = _aiva()
    item = ChatMessage(
        role="user",
        content=["Yes, this is Staffing Solutions."],
        metrics={"end_of_turn_delay": 2.5, "transcription_delay": 0.187},
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        aiva._on_conversation_item_added(SimpleNamespace(item=item))

    assert output.getvalue().strip() == (
        "[TURN LATENCY] stage=opener end_of_turn=2.500s transcription=0.187s"
    )
    assert aiva._user_turn_count == 1

    no_metrics = ChatMessage(role="user", content=["Hello?"])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        aiva._on_conversation_item_added(SimpleNamespace(item=no_metrics))
    assert output.getvalue() == ""
    assert aiva._user_turn_count == 2

    print("agent_v2: turn-latency logging checks passed")


if __name__ == "__main__":
    _check()
