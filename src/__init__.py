import asyncio
from datetime import datetime
import logging
import typing
from aiogram.types import Message
from dataclasses import dataclass


class QueryStatusManager:
    _instances: dict[int, list["QueryStatusManager"]] = {}

    def __init__(self, msg: Message) -> None:
        self.msg: Message = msg
        self.content = []
        self.media_group = None

        if msg.chat.id not in self._instances:
            self._instances[msg.chat.id] = [self]

    async def set_media_grouped(self, media_group_id: str):
        for instance in self._instances[self.msg.chat.id]:
            if instance.media_group == media_group_id:
                return instance
        self.media_group = media_group_id
        return self

    async def update_message(self, text: str):
        self.content.append(text)
        await self.msg.edit_text("\n".join(self.content))

    async def edit_last_line(self, text: str):
        self.content.pop()
        self.content.append(text)
        await self.msg.edit_text("\n".join(self.content))

    async def close(self):
        await self.msg.delete()


@dataclass
class MediaGroupQueue:
    items: dict[
        str, tuple[datetime, int, int]
    ]  # media_group_id -> (timestamp, chat_id, in_progress_count)
    work_queue: dict[str, asyncio.Queue]
    max_age_secs: int = 300
    lock: asyncio.Lock = asyncio.Lock()

    async def add(self, media_group_id: str, chat_id: int):
        if not media_group_id:
            return False
        async with self.lock:
            if data := self.items.get(media_group_id):
                self.items[media_group_id] = (
                    datetime.now(),
                    chat_id,
                    data[2] + 1,
                )
                return True

            self.items[media_group_id] = (datetime.now(), chat_id, 1)
            logging.info(
                f"Added media group {media_group_id} to queue. Currently {self.items[media_group_id][2]} in progress."
            )
            self.work_queue[media_group_id] = asyncio.Queue()
            return False

    async def submit_task(self, media_group_id: str, image: typing.BinaryIO):
        queue = self.work_queue[media_group_id]
        await queue.put(image)

    async def set_processed(self, media_group_id: str):
        async with self.lock:
            if data := self.items.get(media_group_id):
                self.items[media_group_id] = (data[0], data[1], data[2] - 1)
                logging.info(
                    f"Media group {media_group_id} processed one item. Remaining in progress: {self.items[media_group_id][2]}"
                )

    async def get_unprocessed(self):
        async with self.lock:
            now = datetime.now()
            to_remove = [
                (mg_id, chat_id)
                for mg_id, (ts, chat_id, in_prog) in self.items.items()
                if in_prog == 0 and (now - ts).seconds > self.max_age_secs
            ]
            for mg_id, _ in to_remove:
                del self.items[mg_id]

            return to_remove
