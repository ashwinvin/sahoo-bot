import asyncio
import io
import logging
import sys

from os import getenv
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram import F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Message

from google.genai import types, Client

from db import DBConn
from llm.modules import UserSupportAgent


load_dotenv(".env")

TOKEN = getenv("BOT_TOKEN")
ADMIN = getenv("ADMIN")

dp = Dispatcher()
db_con = DBConn()
user_agent = UserSupportAgent(db=db_con)


@dp.message(F.chat.username == ADMIN)
async def customer_handler(message: Message, bot: Bot, g_client: Client) -> None:
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

    answer = user_agent(query=query, images=images, user_id=message.chat.username)
    await message.answer(answer.response)


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    g_client = Client(api_key=getenv("GEMINI_KEY"))
    # And the run events dispatching
    await dp.start_polling(bot, g_client=g_client)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
    db_con.close()
