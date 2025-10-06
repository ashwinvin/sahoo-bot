import sys
import asyncio
import logging
from io import BytesIO
from os import getenv
from datetime import datetime
from dotenv import load_dotenv

from pdf2image import convert_from_bytes
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from google.genai import types, Client

from db import DBConn
from llm.modules import UserSupportAgent
from src import QueryStatusManager
from src.llm.tools import ChromaSingleton


load_dotenv(".env")

TOKEN = getenv("BOT_TOKEN")
ADMIN = getenv("ADMIN")
C_TOKEN = getenv("CHROMA_KEY")

dp = Dispatcher()
db_con = DBConn()


@dp.message()
async def customer_handler(
    message: Message, bot: Bot, g_client: Client, user_agent: UserSupportAgent
) -> None:
    await message.chat.do(action="typing")
    status_msg = await message.reply("Analysing your query...")
    status_manager = QueryStatusManager(status_msg)

    db_con.insert_user(message.chat.id)  # type: ignore
    images = query = file_id = None
    is_doc = False

    if images := message.photo:
        await status_manager.update_message("Downloading images...")
        file_id = images[0].file_id
        images = [await bot.download(photo) for photo in images]  # TODO: BLUNDER!!

    if message.text:
        query = message.text

    elif doc_meta := message.document:
        is_doc = True
        await status_manager.update_message(f"Downloading {doc_meta.file_name}")
        logging.info(f"Document mime type: {doc_meta}")
        file_id = doc_meta.file_id
        document = await bot.download(doc_meta)

        assert document is not None
        assert doc_meta.file_name is not None

        if doc_meta.mime_type == "application/pdf":
            await status_manager.update_message("Preprocessing pdf document...")
            pages = convert_from_bytes(document.read())
            images = []

            for page in pages:
                img_byte_arr = BytesIO()
                page.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                images.append(img_byte_arr)

            query = "The user has sent a PDF document. Please analyze the images extracted from the PDF."

        elif doc_meta.mime_type == "application/binary" and doc_meta.file_name.endswith(
            ".md"
        ):
            query = f"The user has a markdown file with following content: \n {document.read()}."

    elif m_voice := (message.voice or message.audio):
        await status_manager.update_message("Transcribing voice message...")
        file_id = m_voice.file_id
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

    msg_id = db_con.insert_message("user", query, images, file_id, is_doc)  # type: ignore
    answer = await user_agent.acall(
        query=query,
        images=images,
        user_id=message.chat.id,
        status_manager=status_manager,
        msg_id=msg_id,
    )

    if answer.document_ids_o and answer.is_hard_retrieval_o:
        for doc_id in answer.document_ids_o:
            (txt, _, file_id, is_doc) = db_con.get_message_by_id(doc_id)
            if file_id and is_doc:
                await message.reply_document(file_id, caption=txt)  
            elif file_id:
                await message.reply_photo(file_id, caption=txt)  
            else:
                await message.reply(txt) # type: ignore  
        return
    else:
        await message.answer(answer.response)
    
    await status_manager.close()


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
