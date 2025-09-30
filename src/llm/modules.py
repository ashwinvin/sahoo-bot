import dspy
import logging

from db import DBConn
from typing import Optional, BinaryIO
from llm.tools import (
    insert_info_embedding,
    insert_message_embedding,
    retrieve_relevant_info,
    retrieve_relevant_messages,
)
from llm.signatures import (
    Analyzer,
    ClassifyQuery,
    GenerateResponse,
    InfoAgent,
    QueryCategory,
    ScheduleAgent,
)

from src.llm.tools import convert_image


class UserSupportAgent(dspy.Module):
    def __init__(self, db: DBConn):
        super().__init__()
        self.q_classifier = dspy.Predict(ClassifyQuery)
        self.schedule_agent = dspy.ReAct(
            ScheduleAgent, tools=[db.insert_reminder, db.get_pending_reminders]
        )  # noqa: F82
        self.analyzer = dspy.Predict(Analyzer)
        self.info_agent = dspy.ReAct(
            InfoAgent,
            tools=[
                retrieve_relevant_info,
                retrieve_relevant_messages,
                db.get_pending_reminders,
                db.get_message_by_id,
            ],
        )
        self.answer_rephraser = dspy.Predict(GenerateResponse)

        self.db = db

    async def aforward(
        self,
        query: str,
        images: Optional[list[BinaryIO]],
        user_id: str,
    ):
        logging.info(f"Query: {query}")

        imgs = (
            [convert_image(img.read()) for img in images]
            if images
            else None
        )

        classification = await self.q_classifier.acall(
            user_text=query,
            user_images=imgs,
        )

        logging.info(f"CAT:{classification.category}")

        msg_id = self.db.insert_message("user", query, images)  # type: ignore
        if not query:
            query = (await self.analyzer.acall(context_img=imgs)).summary

        await insert_message_embedding(
            content=query, user_id=user_id, is_llm=False, msg_id=msg_id
        )

        match classification.category:
            case QueryCategory.INFORMATION:
                info = await self.info_agent.acall(
                    context_txt=query, context_img=imgs, user_id=user_id
                )
                proposed_ans = info.response

                if info.is_data_dump:
                    info_id = self.db.insert_info(proposed_ans, msg_id, user_id)
                    await insert_info_embedding(
                        summary=proposed_ans,
                        user_id=user_id,
                        info_id=info_id,
                        msg_id=msg_id,
                    )
                    logging.info(f"Information Processed: {proposed_ans}")

                if info.has_event_data:
                    scheduled_pred = await self.schedule_agent.acall(
                        user_id=user_id,
                        content_txt=query,
                        content_img=imgs,
                    )
                    proposed_ans += scheduled_pred.response

            case QueryCategory.SCHEDULE:
                scheduled_pred = await self.schedule_agent.acall(
                        user_id=user_id,
                        content_txt=query,
                        content_img=imgs,
                    )
                proposed_ans = scheduled_pred.response

            case _:
                proposed_ans = "I'm sorry, but I couldn't understand your request."

        final_ans = await self.answer_rephraser.acall(
            user_query=query,
            proposed_answer=proposed_ans,
            category=classification.category,
        )

        self.db.insert_message("llm", final_ans.response)  # type: ignore
        await insert_message_embedding(
            content=query, user_id=user_id, is_llm=True, msg_id=msg_id
        )
        return final_ans
