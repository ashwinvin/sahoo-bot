import dspy
from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class Date(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int

    def to_datetime(self):
        return datetime(self.year, self.month, self.day, self.hour, self.minute)


class QueryCategory(Enum):
    GENERAL_INQUIRY = "GENERAL_INQUIRY"
    INFORMATION_DUMP = "INFORMATION_DUMP"
    SCHEDULING = "SCHEDULING"
    RESOURCE_MANAGMENT = "RESOURCE_MANAGMENT"
    OTHER = "OTHER"


# Customer Agent Signatures
class ClassifyQuery(dspy.Signature):
    """Classify the user's message into one of the following support categories and identify if it's a follow-up to a previous conversation.
    Categories:
        GENERAL_INQUIRY: The user is directly asking a question seeking immediate information or clarification on
            general matters. Examples: "What's the weather like?", "How do I reset my password?", "Explain quantum physics."
        INFORMATION_DUMP: The user is providing content (text, links, or implicitly through image uploads without
            accompanying text) that appears to be for storage, future reference, or passive ingestion by the LLM, rather
            than an immediate interactive query. This typically includes academic notes, timetables, brochures, articles,
            or other data the user wants to save without extensive immediate discussion. The user generally does not
            explicitly state it's an "information dump." If a user uploads images with no accompanying text, this is a
            strong indicator of an INFORMATION_DUMP.
        SCHEDULING: The user is asking about the timing of an event, requesting to be reminded about a specific matter,
            or attempting to set up an event. Examples: "When is the meeting?", "Remind me to call John at 3 PM,"
            "Schedule a dinner for next Tuesday."
        RESOURCE_MANAGEMENT: The user is explicitly asking to retrieve, manage (e.g., delete, update), or interact with
            previously stored documents, notes, or resources. Examples: "Show me my notes on project X," "Find the
            contract from last month," "Delete the file named 'draft.docx'."
        OTHER: The message content does not clearly fit into any of the above defined categories.
    Follow-up Detection:
        Determine if the current message is a continuation or follow-up to a previously discussed topic.
    """

    user_text: str = dspy.InputField(
        desc="The text contained in the message user had sent"
    )
    user_images: Optional[list[dspy.Image]] = dspy.InputField(
        desc="A set of pictures the user had sent"
    )
    conversation_list: list[tuple[int, str]] = dspy.InputField(
        desc="Previous history of conversations in the format (id, topic)"
    )

    category: list[QueryCategory] = dspy.OutputField()
    needs_clarification: bool = dspy.OutputField(
        desc="Return true if the question has ambiguties which needs clarification."
    )
    required_clarifications: Optional[str] = dspy.OutputField(
        desc="The topic items which need clarification."
    )

    conversation_id: Optional[int] = dspy.OutputField(
        desc="The conversation id which the user is continuing from."
    )
    is_new_conversation: bool = dspy.OutputField(
        desc="Return true if the user hasn't talked about the current matter"
    )
    conversation_topic: Optional[str] = dspy.OutputField(
        desc="A brief summary about the message which you can use as a reference in the future for the current task."
        "Generate only if the conversation is new."
    )


class InfoSummarizer(dspy.Signature):
    """Analyze the given text, image or both and summarize the data."""

    context_txt: Optional[str] = dspy.InputField()
    context_img: Optional[list[dspy.Image]] = dspy.InputField()
    summary: str = dspy.OutputField()
    reminder_txt: Optional[str] = dspy.OutputField(
        desc="Venue and other details about the event if any."
    )
    remind_at: Optional[Date] = dspy.OutputField(desc="Date and time of the event.")


class AnswerQuestion(dspy.Signature):
    """Answer a user's question based on provided context."""

    classification: QueryCategory = dspy.InputField(
        desc="The category to which the question relates to. This indicates which tools to use"
    )
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ClarifyQuery(dspy.Signature):
    """Formulate a clarifying question to get more details from the user."""

    unclear_query: str = dspy.InputField()
    required_clarifications: str = dspy.InputField()
    clarifying_question: str = dspy.OutputField()


class GenerateResponse(dspy.Signature):
    """Rewrite the given response into a polite and helpful customer support response. The response should be concise and easy to understand.
    The response must be in point by point format if there are multiple points to be addressed.

    If the category is INFORMATION_DUMP, acknowledge that the information has been noted.
    If the category is SCHEDULING, include the reminder details in the response.
    If the category is GENERAL_INQUIRY, provide a concise and accurate answer to the user's question by using the proposed answer.
    If the category is OTHER, politely inform the user that their query is outside the scope of support and suggest alternative resources or contacts for assistance.
    """

    user_query: str = dspy.InputField()
    category: QueryCategory = dspy.InputField()
    proposed_answer: str = dspy.InputField()
    response: str = dspy.OutputField()


# Admin Agent Signatures
class Analyzer(dspy.Signature):
    """Analyze the given image, text or both and extract main information from it."""

    context_img: Optional[dspy.Image] = dspy.InputField()
    context_txt: Optional[str] = dspy.InputField(
        desc="The text which was provided with image."
    )
    info: str = dspy.OutputField(desc="The extracted information in concise form.")
