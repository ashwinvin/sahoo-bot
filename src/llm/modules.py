from os import getenv
import pathlib
import dspy
import logging

from db import DBConn
from typing import Optional, BinaryIO
from llm.tools import (
    EmbeddingStore,
    create_pdf,
)
from llm.signatures import (
    Analyzer,
    ClassifyQuery,
    DocumentGenerator,
    ResponsePolisher,
    InfoAgent,
    QueryCategory,
    ScheduleAgent,
)

from src.llm.tools import convert_image
from src import QueryStatusManager


class UserSupportAgent(dspy.Module):
    def __init__(
        self,
        db: DBConn,
        embed_store: EmbeddingStore,
        wiki_tools: list[dspy.Tool],
        # docgen_tools: list[dspy.Tool],
    ):
        super().__init__()
        self.db = db
        self.embed_store = embed_store

        self.q_classifier = dspy.Predict(ClassifyQuery)
        self.schedule_agent = dspy.ReAct(
            ScheduleAgent, tools=[db.insert_reminder, db.get_pending_reminders]
        )  # noqa: F82
        self.document_generator = dspy.ChainOfThought(DocumentGenerator)

        self.analyzer = dspy.Predict(Analyzer)
        self.info_agent = dspy.ReAct(
            InfoAgent,
            tools=[
                embed_store.retrieve_relevant_info,
                embed_store.retrieve_relevant_messages,
                db.get_pending_reminders,
                db.get_message_by_id,
                *wiki_tools,
            ],
        )
        self.answer_rephraser = dspy.Predict(ResponsePolisher)

    async def aforward(
        self,
        query: str,
        images: Optional[list[BinaryIO]],
        user_id: int,
        status_manager: QueryStatusManager,
        msg_id: int,
        chat_history: dict,
        is_grouped_msg: bool = False,
    ):
        [img.seek(0) for img in images] if images else None

        imgs = [convert_image(img.read()) for img in images] if images else None

        classification = await self.q_classifier.acall(
            user_text=query,
            user_images=imgs,
        )

        logging.info(f"CAT:{classification.category}")

        if not query:
            query = (await self.analyzer.acall(context_img=imgs)).summary

        await self.embed_store.insert_message_embedding(
            content=query, user_id=user_id, is_llm=False, msg_id=msg_id
        )
        is_hard_retrieval = False
        doc_ids = []

        match classification.category:
            case QueryCategory.INFORMATION | QueryCategory.ASSIGNMENT_GENERATION:
                info_history = chat_history.get(user_id, {"info": []})["info"]

                await status_manager.update_message(
                    "🔍 Analyzing and retrieving relevent information..."
                )

                info = await self.info_agent.acall(
                    context_txt=query,
                    context_img=imgs,
                    user_id=user_id,
                    history=dspy.History(messages=info_history),
                )
                await status_manager.edit_last_line("✅ Analyzing and retrieving relevent information...")
                info_history.append(
                    {
                        "context_txt": query,
                        "context_img": imgs,
                        "user_id": user_id,
                        **info,
                    }
                )

                if len(info_history) > 20:
                    info_history.pop(0)

                proposed_ans = info.response

                if (
                    info.is_data_dump
                    and classification.category
                    is not QueryCategory.ASSIGNMENT_GENERATION
                ):
                    await status_manager.update_message("⚪ Updating Information database")
                    info_id = self.db.insert_info(proposed_ans, msg_id, str(user_id))
                    await self.embed_store.insert_info_embedding(
                        summary=proposed_ans,
                        user_id=user_id,
                        info_id=info_id,
                        msg_id=msg_id,
                        has_img=True if imgs else False,
                    )

                    await status_manager.edit_last_line(
                        "✅ Information database updated"
                    )

                    if event_prompt := info.set_event_reminder:
                        scheduled_pred = await self.schedule_agent.acall(
                            user_id=user_id,
                            content_txt=event_prompt,
                            content_img=imgs,
                        )
                        proposed_ans += scheduled_pred.response

                if info.source_documents:
                    await status_manager.update_message(
                        f"🔶 Information sourced from message ids: {info.source_documents}"
                    )
                    doc_ids = info.source_documents

                if info.is_hard_retrieval:
                    await status_manager.update_message(
                        f"🔶 Document found from message ids: {info.source_documents}"
                    )
                    is_hard_retrieval = True

                if classification.category is QueryCategory.ASSIGNMENT_GENERATION:
                    await status_manager.update_message(
                        "⚪ Generating the requested document."
                    )

                    docgen_pred = await self.document_generator.acall(
                        user_query=query,
                        context=proposed_ans,
                    )

                    o_path = create_pdf(
                        docgen_pred.file_name,
                        docgen_pred.sections,
                        docgen_pred.custom_css,
                    )
                    logging.info(f"The document has been saved to {o_path}")
                    proposed_ans = f"The document has been generated with file name: {docgen_pred.file_name}"

            case QueryCategory.SCHEDULE:
                scheduled_pred = await self.schedule_agent.acall(
                    user_id=user_id,
                    content_txt=query,
                    content_img=imgs,
                )
                proposed_ans = scheduled_pred.response

            case _:
                proposed_ans = "I'm sorry, but I couldn't understand your request."

        if is_grouped_msg:
            return

        final_ans = await self.answer_rephraser.acall(
            user_query=query,
            proposed_answer=proposed_ans,
            category=classification.category,
            is_hard_retrieval=is_hard_retrieval,
            document_ids=doc_ids,
        )

        self.db.insert_message("llm", final_ans.response)  # type: ignore
        await self.embed_store.insert_message_embedding(
            content=query, user_id=user_id, is_llm=True, msg_id=msg_id
        )
        logging.info(final_ans)
        return final_ans
