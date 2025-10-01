from aiogram.types import Message


class QueryStatusManager:
    def __init__(self, msg: Message) -> None:
        self.msg: Message = msg

    async def update_message(self, text: str):
        await self.msg.edit_text(text)

    async def close(self):
        await self.msg.delete()
