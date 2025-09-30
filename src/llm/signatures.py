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
    # NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    INFORMATION = "INFORMATION"
    SCHEDULE = "SCHEDULE"
    OTHER = "OTHER"


# Customer Agent Signatures
class ClassifyQuery(dspy.Signature):
    """Classify the user's message into one of the following support categories.
    Categories:
        INFORMATION: The user is asking to provide new information or update existing information. If the user is providing
            information, it'll also be considered as INFORMATION. An image with no context, or a text message with no question
            is also considered as INFORMATION. The user may also be asking about schedule details of an event, in which case
            the category is still INFORMATION.

        SCHEDULE: The user is requesting to be reminded about a specific matter or attempting to set up an event. 
            Examples: "When is the meeting?", "Remind me to call John at 3 PM," "Schedule a dinner for next Tuesday."
        
        OTHER: The message content does not clearly fit into any of the above defined categories.
    """
        # NEEDS_CLARIFICATION: The user's message is ambiguous, lacks sufficient detail, or is too broad to provide a specific answer.

    user_text: str = dspy.InputField(
        desc="The text contained in the message user had sent"
    )
    user_images: Optional[list[dspy.Image]] = dspy.InputField(
        desc="A set of pictures the user had sent"
    )

    category: QueryCategory = dspy.OutputField()
    # required_clarifications: Optional[str] = dspy.OutputField(
    #     desc="The topic items which need clarification."
    # )


class InfoAgent(dspy.Signature):
    """You are an information manager. Your task is to use the tools provided to store and retrieve information as needed
    after a thorough analysis of the user's input. 
    
    - If the user provides new information, summarize it concisely and return while setting the 'response' field to the 
      summary and 'is_data_dump' to True.
    - If the user provided information that has details regarding an event, extract the event details and set 
      'has_event_data' to True.
    - If the user asks a question or requests information, retrieve relevant information using the tools provided.
    """

    context_txt: Optional[str] = dspy.InputField()
    context_img: Optional[list[dspy.Image]] = dspy.InputField()
    user_id: str = dspy.InputField()

    response: Optional[str] = dspy.OutputField()
    has_event_data: bool = dspy.OutputField()
    is_data_dump: bool = dspy.OutputField()

class ScheduleAgent(dspy.Signature):
    """You are a scheduling assistant. Your task is to use the tools provided to manage and schedule events based on user requests."""

    user_id: str = dspy.InputField()
    content_txt: Optional[str] = dspy.InputField()
    content_img: Optional[list[dspy.Image]] = dspy.InputField()
    response: str = dspy.OutputField()


class AnswerQuestion(dspy.Signature):
    """Answer a user's question based on provided context and use the tools at hand to fulfill the request."""

    classification: QueryCategory = dspy.InputField(
        desc="The category to which the question relates to. This indicates which tools to use"
    )
    user_id: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ClarifyQuery(dspy.Signature):
    """Formulate a clarifying question to get more details from the user."""

    unclear_query: str = dspy.InputField()
    required_clarifications: str = dspy.InputField()
    clarifying_question: str = dspy.OutputField()


class GenerateResponse(dspy.Signature):
    """STRICTLY ONLY Rewrite the given response into a polite and helpful customer support response.
    The response should be concise and easy to understand.
    The response must be in point by point format if there are multiple points to be addressed.

    If the category is INFORMATION_DUMP, acknowledge that the information has been noted.
    If the category is SCHEDULING, include the reminder details in the response.
    If the category is GENERAL_INQUIRY, provide a concise and accurate answer to the user's question by using the proposed answer.
    If the category is OTHER, politely inform the user that their query is outside the scope of support and suggest
        alternative resources or contacts for assistance.
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
    summary: str = dspy.OutputField(desc="The extracted information in concise form.")
