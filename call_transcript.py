import asyncio

from livekit.agents.llm import ChatMessage

from db import add_call_turn, finish_call


class CallTranscriptRecorder:
    """Persist finalized SDK conversation turns without blocking audio."""

    def __init__(self, call_id, stage_getter):
        self.call_id = call_id
        self._stage_getter = stage_getter
        self._queue = asyncio.Queue()
        self._turn_number = 0
        self._worker = asyncio.create_task(self._write_turns())
        self._closed = False

    def on_conversation_item_added(self, event):
        if self._closed:
            return

        item = event.item
        if not isinstance(item, ChatMessage) or item.role not in {"user", "assistant"}:
            return

        text = item.text_content.strip()
        if not text:
            return

        self._turn_number += 1
        self._queue.put_nowait((
            self._turn_number,
            item.role,
            text,
            self._stage_getter(),
        ))

    async def _write_turns(self):
        while True:
            turn = await self._queue.get()
            try:
                if turn is None:
                    return
                await asyncio.to_thread(add_call_turn, self.call_id, *turn)
            except Exception as exc:
                print(f"ERROR: failed to persist call {self.call_id} turn: {exc}")
            finally:
                self._queue.task_done()

    async def close(self, final_call_result):
        if self._closed:
            return
        self._closed = True
        await self._queue.join()
        self._queue.put_nowait(None)
        await self._worker
        await asyncio.to_thread(finish_call, self.call_id, final_call_result)
