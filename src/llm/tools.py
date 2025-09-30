import base64
import logging
from chromadb import HttpClient as ChromaClient
import dspy

chroma_client = ChromaClient()
chroma_client.get_or_create_collection(name="bot-infostore")


def insert_info_embedding(summary: str, user_id: str, info_id: int, msg_id: int):
    collection = chroma_client.get_or_create_collection(name="bot-infostore")
    logging.info(f"Inserting embedding for info_id: {info_id}")

    collection.add(
        ids=[str(info_id)],
        documents=[summary],
        metadatas=[{"user_id": user_id, "info_id": info_id, "msg_id": msg_id}],
    )


def insert_message_embedding(content: str, user_id: str, is_llm: bool, msg_id: int):
    collection = chroma_client.get_or_create_collection(name="bot-msgstore")
    logging.info(f"Inserting embedding for msg_id: {msg_id}")

    collection.add(
        ids=[str(msg_id)],
        documents=[content],
        metadatas=[{"user_id": user_id, "msg_id": msg_id, "is_llm": is_llm}],
    )


def retrieve_relevant_info(query: str, user_id: str):
    """Retrieve relevant information for a given query and user.
    Returns:
        [(Id, Distance, Document)]: The id points to the source of the information stored in the database
            and Document is the summary of the information.
    """
    collection = chroma_client.get_or_create_collection(name="bot-infostore")
    logging.info(f"Retrieving relevant info for user: {user_id} with query: {query}")
    results = collection.query(
        query_texts=[query],
        n_results=10,
        include=["documents", "metadatas", "distances"],
        where={"user_id": user_id},
    )

    if not results["ids"]:
        return

    data = sorted(
        [
            (_id, dist, res)
            for _id, dist, res in zip(
                results["ids"][0],
                results["distances"][0], # type: ignore
                results["documents"][0],  # type: ignore
            )
        ],
        key=lambda x: x[1],
        reverse=True
    )
    logging.info("Retrieved data: %s", data)  # noqa: F821
    
    return data 


def retrieve_relevant_messages(query: str, user_id: str):
    """Retrieve relevant past messages for a given query and user.
    Returns:
        [Document]: The past messages sent by the user.
    """
    collection = chroma_client.get_or_create_collection(name="bot-msgstore")
    results = collection.query(
        query_texts=[query],
        n_results=10,
        where={"user_id": user_id, "is_llm": False},
    )
    if not results["ids"]:
        return
    logging.info("Retrieved Messages: %s", results["documents"])  # noqa: F821
    return [res for res in results["documents"]]  # type: ignore


if __name__ == "__main__":
    chroma_client.get_or_create_collection(name="bot-infostore")


def convert_image(file: bytes) -> dspy.Image:
    return dspy.Image.from_file(
        "data:image/png;base64," + base64.b64encode(file).decode("utf-8")
    )
