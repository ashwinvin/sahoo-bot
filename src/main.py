import sys
import asyncio
import logging
from os import getenv
from datetime import datetime
from dotenv import load_dotenv

from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties

from google.genai import types, Client

from db import DBConn
from llm.modules import UserSupportAgent
from src.llm.tools import ChromaSingleton


load_dotenv(".env")

TOKEN = getenv("BOT_TOKEN")
ADMIN = getenv("ADMIN")
C_TOKEN = getenv("CHROMA_KEY")

dp = Dispatcher()
db_con = DBConn()


@dp.message(F.chat.username == ADMIN)
async def customer_handler(
    message: Message, bot: Bot, g_client: Client, user_agent: UserSupportAgent
) -> None:
    await message.chat.do(action="typing")
    db_con.insert_user(message.chat.username)  # type: ignore
    images = query = None

    if images := message.photo:
        images = [await bot.download(photo) for photo in images]

    if message.text:
        query = message.text
    elif m_voice := message.voice:
        voice = await bot.download(m_voice)
        assert voice is not None
        query = g_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Generate a transcript of the speech in the language it was spoken in."
                "Make sure to only respond with transcription, do not add filler sentences.",
                types.Part.from_bytes(
                    data=voice.read(),
                    mime_type="audio/mp3",
                ),
            ],
        ).text
        logging.info(f"Transcription: {query}")

    if not query and not images:
        await message.answer("Unsupported message format.")
        return

    answer = await user_agent.acall(
        query=query, images=images, user_id=message.chat.username
    )
    await message.answer(answer.response)


async def cron_manager(bot: Bot):
    while True:
        logging.info("Checking for pending reminders...")
        pending_reminders = db_con.get_all_pending_reminders()

        for reminder_id, user_id, reminder_text, remind_at in pending_reminders:
            if remind_at != datetime.now().replace(second=0, microsecond=0):
                continue

            logging.debug(
                f"Sending reminder to {user_id}: {reminder_text} at {remind_at}"
            )
            await bot.send_message(chat_id=user_id, text=f"Reminder: {reminder_text}")
            db_con.update_reminder_status(reminder_id, "sent")

        await asyncio.sleep(60)


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))  # type: ignore
    g_client = Client(api_key=getenv("GEMINI_KEY"))
    user_agent = UserSupportAgent(db=db_con)
    cs = await ChromaSingleton()
    await cs.setup()


    asyncio.create_task(cron_manager(bot), name="CronManager")
    await dp.start_polling(bot, g_client=g_client, user_agent=user_agent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
    db_con.close()
