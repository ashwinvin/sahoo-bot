import dspy
import logging
from db import DBConn
import base64
from typing import Optional, BinaryIO
from llm.signatures import (
    Analyzer,
    ClassifyQuery,
    ClarifyQuery,
    AnswerQuestion,
    GenerateResponse,
    InfoSummarizer,
    QueryCategory,
)


def convert_image(file: BinaryIO) -> str:
    return "data:image/png;base64," + base64.b64encode(file.read()).decode("utf-8")


class UserSupportAgent(dspy.Module):
    def __init__(self, db: DBConn):
        super().__init__()
        self.q_classifier = dspy.ChainOfThought(ClassifyQuery)
        self.q_clarifier = dspy.Predict(ClarifyQuery)
        self.react = dspy.ReAct(AnswerQuestion, tools=[])
        self.info_summarizer = dspy.ChainOfThought(InfoSummarizer)
        self.answer_rephraser = dspy.Predict(GenerateResponse)

        self.db = db

    def forward(self, query: str, images: Optional[list[BinaryIO]], user_id: str):
        logging.info(f"Query: {query}")
        imgs = (
            [dspy.Image.from_file(convert_image(img)) for img in images]
            if images
            else None
        )

        convo_list = self.db.get_user_conversations(user_id)
        classification = self.q_classifier(
            user_text=query, user_images=imgs, conversation_list=convo_list
        )

        logging.info(
            f"""CAT:{classification.category} NC:{classification.needs_clarification}  
            CID: {classification.conversation_id} INC: {classification.is_new_conversation}"""
        )

        if classification.is_new_conversation:
            convo_id = self.db.insert_conversation(
                classification.conversation_topic, user_id
            )
        else:
            convo_id = classification.conversation_id

        logging.info(f"Conversation ID: {convo_id}")
        msg_id = self.db.insert_message(convo_id, "user", query, images)  # type: ignore

        if classification.needs_clarification:
            logging.info(
                f"Needs clarification: {classification.required_clarifications}"
            )
            c_query = self.q_clarifier(
                unclear_query=query,
                required_clarifications=classification.required_clarifications,
            )
            proposed_ans = c_query.clarifying_question

        elif QueryCategory.INFORMATION_DUMP in classification.category:
            info = self.info_summarizer(context_txt=query, context_img=imgs)
            self.db.insert_info(info.summary, msg_id, user_id)
            logging.info(
                f"Information Processed: {info.summary} \n {info.reminder_txt} @ {info.remind_at}"
            )

            if info.reminder_txt and info.remind_at:
                self.db.insert_reminder(
                    user_id, info.reminder_txt, info.remind_at.to_datetime(), msg_id
                )

            proposed_ans = f"{info.summary} \n The above information has been noted."
        else:
            logging.info("Unknown category")
            proposed_ans = ""

        # answer = self.react(classification=classification.category, question=query)
        # logging.info("Answer Generated")

        final_ans = self.answer_rephraser(
            user_query=query,
            proposed_answer=proposed_ans,
            category=classification.category,
        )
        self.db.insert_message(convo_id, "llm", final_ans.response)  # type: ignore
        return final_ans


class AdminSupportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(Analyzer)

    def forward(self, context_txt: Optional[str], context_img):
        analysis = self.analyzer(context_img=context_img, context_txt=context_txt)
