import os
import pathlib
import sys
import typing
import asyncio
import logging
from os import getenv
from io import BytesIO
from datetime import datetime
from aiogram.types.message import Message
from dotenv import load_dotenv

from pdf2image import convert_from_bytes
from aiogram.types import Document, BufferedInputFile, ErrorEvent
from aiogram.handlers import MessageHandler
from aiogram.enums import ParseMode
from aiogram import F, Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from google.genai import types, Client

from db import DBConn, DocType
from llm.modules import UserSupportAgent
from src import MediaGroupQueue, QueryStatusManager
from src.llm.tools import EmbeddingStore, McpClient


load_dotenv(".env")

TOKEN = getenv("BOT_TOKEN")
ADMIN = getenv("ADMIN")
C_TOKEN = getenv("CHROMA_KEY")

dp = Dispatcher()
db_con = DBConn()


@dp.message()
class UserHandler(MessageHandler):
    def __init__(self, event: Message, **kwargs: typing.Any) -> None:
        self.chat_history = {}
        super().__init__(event, **kwargs)

    def __getattr__(self, name: str) -> typing.Any:
        if item := self.data.get(name):
            return item
        else:
            raise AttributeError(f"{name} not found self.data")

    async def parse_document(self, doc_meta: Document):
        file_id = doc_meta.file_id
        document = await self.bot.download(doc_meta)
        images = []

        assert document is not None
        assert doc_meta.file_name is not None
        assert doc_meta.mime_type is not None

        if doc_meta.mime_type == "application/pdf":
            # await status_manager.update_message("Preprocessing pdf document...")
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

        elif doc_meta.mime_type.startswith("image"):
            query = f"The user has sent an image file named {doc_meta.file_name}. Please analyze the image."
            images = [document]
        else:
            query = f"The given mime type is not supported. {doc_meta.mime_type}"

        return query, images, file_id

    async def parse_voice(self, m_voice):
        file_id = m_voice.file_id
        voice = await self.bot.download(m_voice)
        assert voice is not None

        query = self.g_client.models.generate_content(
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
        query += "The above is a transcription of a voice message sent by the user."
        logging.info(f"Transcription: {query}")
        return query, file_id

    async def handle(self) -> typing.Any:
        await self.chat.do(action="typing")
        grouped_msg = False

        status_msg = await self.event.reply("Analysing your query...")
        status_manager = QueryStatusManager(status_msg)
        db_con.insert_user(self.chat.id)  # type: ignore
        images = query = file_id = None
        doc_type = None

        if images := self.event.photo:
            grouped_msg = await self.media_group_queue.add(
                self.event.media_group_id, self.event.chat.id
            )
            await status_manager.set_media_grouped(
                self.event.media_group_id  # type: ignore
            )

            doc_type = DocType.PHOTO
            file_id = images[0].file_id
            images = [await self.bot.download(images[-1])]

            if (
                not grouped_msg
            ):  # Wait for any other images which might be send together
                await status_manager.update_message(
                    "Waiting for any other image that might be in the album."
                )
                while True:
                    try:
                        async with asyncio.timeout(5):
                            logging.info(
                                f"MG ID: {self.event.media_group_id} recieved a new photo"
                            )
                            images.append(
                                await self.media_group_queue.work_queue[
                                    self.event.media_group_id
                                ].get()
                            )
                    except asyncio.TimeoutError:
                        logging.info(
                            f"MG ID: {self.event.media_group_id} recieved {len(images)} photos"
                        )
                        await status_manager.update_message(
                            f"Downloaded {len(images)} from the album"
                        )
                        break

            if grouped_msg:
                await self.media_group_queue.submit_task(
                    self.event.media_group_id, images[0]
                )
                await status_manager.close()
                return

        if self.event.text:
            query = self.event.text

        elif doc_meta := self.event.document:
            doc_type = DocType.DOCUMENT
            await status_manager.update_message(f"Downloading {doc_meta.file_name}")
            logging.info(f"Document mime type: {doc_meta}")
            query, images, file_id = await self.parse_document(doc_meta)

        elif m_voice := (self.event.voice or self.event.audio):
            doc_type = DocType.VOICE
            await status_manager.update_message("Transcribing voice message...")
            query, file_id = await self.parse_voice(m_voice)

        if not query and not images:
            await self.event.answer("Unsupported message format.")
            return

        msg_id = db_con.insert_message("user", query, images, file_id, doc_type)  # type: ignore

        reply_context = (
            db_con.get_message_by_id(self.event.reply_to_message.message_id)
            if self.event.reply_to_message
            else None
        )

        answer = await self.user_agent.acall(
            query=query,
            images=images,
            user_id=self.event.chat.id,
            status_manager=status_manager,
            msg_id=msg_id,
            is_grouped_msg=grouped_msg,
            chat_history=self.chat_history,
        )

        if answer.document_ids_o and answer.is_hard_retrieval_o:
            for doc_id in answer.document_ids_o:
                (txt, _, file_id, doc_type) = db_con.get_message_by_id(doc_id)

                if file_id is None:
                    await self.event.reply(txt)  # type: ignore
                    continue

                match doc_type:
                    case DocType.DOCUMENT:
                        await self.event.reply_document(file_id, caption=txt)
                    case DocType.PHOTO:
                        await self.event.reply_photo(file_id, caption=txt)
                    case DocType.VOICE:
                        await self.event.reply_voice(file_id, caption=txt)
        elif answer.output_doc:
            with open(
                pathlib.Path.cwd() / "gen_docs" / answer.output_doc,
                "rb",  # type: ignore
            ) as file:
                await self.event.reply_document(
                    BufferedInputFile(file=file.read(), filename=answer.output_doc)
                )
        else:
            await self.event.answer(answer.response)

        await status_manager.close()


@dp.error(F.update.message.as_("msg"))
async def error_handler(event: ErrorEvent, msg: Message):
    await msg.answer(f"Uh Oh! Something broke internally \n {event.exception}")
    logging.critical("Critical error caused by %s", event.exception, exc_info=True)


async def cron_manager(bot: Bot):
    while True:
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


async def media_group_ack(media_queue: MediaGroupQueue, bot: Bot):
    while True:
        await asyncio.sleep(60)
        unprocessed = await media_queue.get_unprocessed()

        for mg_id, chat_id in unprocessed:
            logging.info(f"Cleaning up media group id: {mg_id}")
            await bot.send_message(
                chat_id=chat_id,
                text="Finished processing the photo album. You can continue sending more messages.",
            )


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))  # type: ignore
    g_client = Client(api_key=getenv("GEMINI_KEY"))

    media_group_queue = MediaGroupQueue(items={}, work_queue={})
    embed_store = await EmbeddingStore.create()

    wiki_tools = await McpClient.create("wikipedia-mcp", [], {})


    user_agent = UserSupportAgent(
        db=db_con,
        embed_store=embed_store,
        wiki_tools=wiki_tools.tools,
    )

    asyncio.create_task(cron_manager(bot), name="CronManager")
    asyncio.create_task(
        media_group_ack(media_group_queue, bot), name="MediaGroupAcknowledger"
    )

    await dp.start_polling(
        bot,
        g_client=g_client,
        user_agent=user_agent,
        media_group_queue=media_group_queue,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
    db_con.close()
