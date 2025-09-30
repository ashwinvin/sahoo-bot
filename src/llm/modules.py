import dspy
import base64
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
    ClarifyQuery,
    GenerateResponse,
    InfoAgent,
    QueryCategory,
    ScheduleAgent,
)


def convert_image(file: BinaryIO) -> str:
    return "data:image/png;base64," + base64.b64encode(file.read()).decode("utf-8")


class UserSupportAgent(dspy.Module):
    def __init__(self, db: DBConn):
        super().__init__()
        self.q_classifier = dspy.ChainOfThought(ClassifyQuery)
        self.q_clarifier = dspy.Predict(ClarifyQuery)
        self.schedule_agent = dspy.ReAct(
            ScheduleAgent, tools=[db.insert_reminder, db.get_pending_reminders]
        )  # noqa: F82
        self.analyzer = dspy.Predict(Analyzer)
        self.info_agent = dspy.ReAct(
            InfoAgent, tools=[retrieve_relevant_info, retrieve_relevant_messages, db.get_pending_reminders]
        )
        self.answer_rephraser = dspy.Predict(GenerateResponse)

        self.db = db

    def forward(
        self,
        query: str,
        images: Optional[list[BinaryIO]],
        user_id: str,
    ):
        logging.info(f"Query: {query}")

        imgs = (
            [dspy.Image.from_file(convert_image(img)) for img in images]
            if images
            else None
        )

        classification = self.q_classifier(
            user_text=query,
            user_images=imgs,
        )

        logging.info(f"CAT:{classification.category}")

        msg_id = self.db.insert_message("user", query, images)  # type: ignore
        if not query:
            query = self.analyzer(context_img=imgs).summary

        insert_message_embedding(
            content=query, user_id=user_id, is_llm=False, msg_id=msg_id
        )

        match classification.category:
            # case QueryCategory.NEEDS_CLARIFICATION:
            #     logging.info(
            #         f"Needs clarification: {classification.required_clarifications}"
            #     )
            #     c_query = self.q_clarifier(
            #         unclear_query=query,
            #         required_clarifications=classification.required_clarifications,
            #     )
            #     proposed_ans = c_query.clarifying_question

            case QueryCategory.INFORMATION:
                info = self.info_agent(
                    context_txt=query, context_img=imgs, user_id=user_id
                )
                proposed_ans = info.response

                if info.is_data_dump:
                    info_id = self.db.insert_info(proposed_ans, msg_id, user_id)
                    insert_info_embedding(
                        summary=proposed_ans,
                        user_id=user_id,
                        info_id=info_id,
                        msg_id=msg_id,
                    )
                    logging.info(f"Information Processed: {proposed_ans}")

                if info.has_event_data:
                    proposed_ans += self.schedule_agent(
                        user_id=user_id,
                        content_txt=query,
                        content_img=imgs,
                    ).response

            case QueryCategory.SCHEDULE:
                proposed_ans = self.schedule_agent(
                    user_id=user_id,
                    content_txt=query,
                    content_img=imgs,
                ).response

            case _:
                proposed_ans = "I'm sorry, but I couldn't understand your request."

        final_ans = self.answer_rephraser(
            user_query=query,
            proposed_answer=proposed_ans,
            category=classification.category,
        )

        self.db.insert_message("llm", final_ans.response)  # type: ignore
        insert_message_embedding(
            content=query, user_id=user_id, is_llm=True, msg_id=msg_id
        )
        return final_ans


class AdminSupportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(Analyzer)

    def forward(self, context_txt: Optional[str], context_img):
        analysis = self.analyzer(context_img=context_img, context_txt=context_txt)
