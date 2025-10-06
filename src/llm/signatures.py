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

    user_text: str = dspy.InputField(
        desc="The text contained in the message user had sent"
    )
    user_images: Optional[list[dspy.Image]] = dspy.InputField(
        desc="A set of pictures the user had sent"
    )

    category: QueryCategory = dspy.OutputField()


class InfoAgent(dspy.Signature):
    """You are an information manager. Your task is to use the tools provided to store and retrieve information as needed
    after a thorough analysis of the user's input.

    - If the user provides new information, summarize it concisely and return while setting the 'response' field to the
      summary and 'is_data_dump' to True and if it also has details regarding an event, extract the event details and set
      'set_event_reminder' as a prompt to another agent.

    - If a users wants to know about their schedule or an event that is bound to happen, use the tools to fetch the schedule
        and information about the event.

    - If the user asks a question or requests information, retrieve relevant information using the tools provided.
        Make sure to populate the 'source_documents' field with the msg_ids of the relevant documents.

    - If the user is EXPLICITLY asking to retrieve a document, use the tools to fetch the msg_id of the document and return
        it in the 'source_documents' field and set 'is_hard_retrieval' to True. DO NOT attempt to summarize the document, your
        task is only to fetch the document. THE SOURCE DOCUMENTS MUST ONLY CONTAIN THE MSG_IDs OF THE DOCUMENTS, NOT THE CONTENT OF THE MESSAGE.
    """

    context_txt: Optional[str] = dspy.InputField()
    context_img: Optional[list[dspy.Image]] = dspy.InputField()
    user_id: str = dspy.InputField()

    response: Optional[str] = dspy.OutputField()
    set_event_reminder: str = dspy.OutputField()
    is_data_dump: bool = dspy.OutputField()
    source_documents: Optional[list[str]] = dspy.OutputField(
        desc="The msg_ids of the messages you used as source which you retrieved using tools."
    )
    is_hard_retrieval: bool = dspy.OutputField(
        desc="Set this to True if the user is explicitly asking to retrieve a document."
    )

class NotionAgent(dspy.Signature):
    """You are a Notion assistant. Your task is to use the tools provided to create and manage Notion pages based on user requests."""

    query: str = dspy.InputField()
    response: str = dspy.OutputField()


class ScheduleAgent(dspy.Signature):
    """You are a scheduling assistant. Your task is to use the tools provided to manage and schedule events based on user requests."""

    user_id: str = dspy.InputField()
    content_txt: Optional[str] = dspy.InputField()
    content_img: Optional[list[dspy.Image]] = dspy.InputField()
    response: str = dspy.OutputField()


class ClarifyQuery(dspy.Signature):
    """Formulate a clarifying question to get more details from the user."""

    unclear_query: str = dspy.InputField()
    required_clarifications: str = dspy.InputField()
    clarifying_question: str = dspy.OutputField()


class GenerateResponse(dspy.Signature):
    """STRICTLY ONLY Rewrite the given response into a polite and helpful format.
    The response should be concise, easy to understand and as short as possible.
    The response must be in point by point format if there are multiple points to be addressed.
    If the proposed answer is answer to a question, make sure to indicate the sources.

    If is_hard_retrieval is True, do not attempt to answer the question, just inform the user that the document has 
    been found and will be provided. Set the document_ids in the response from the proposed answer
    """

    user_query: str = dspy.InputField()
    category: QueryCategory = dspy.InputField()
    proposed_answer: str = dspy.InputField()
    is_hard_retrieval: bool = dspy.InputField()
    document_ids: Optional[list[str]] = dspy.InputField()

    document_ids_o: Optional[list[str]] = dspy.OutputField()
    is_hard_retrieval_o: bool = dspy.OutputField()
    response: str = dspy.OutputField()


class Analyzer(dspy.Signature):
    """Analyze the given image, text or both and extract main information from it."""

    context_img: Optional[dspy.Image] = dspy.InputField()
    context_txt: Optional[str] = dspy.InputField(
        desc="The text which was provided with image."
    )
    summary: str = dspy.OutputField(desc="The extracted information in concise form.")
