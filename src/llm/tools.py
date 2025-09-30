import logging
from chromadb import HttpClient as ChromaClient

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

def insert_message_embedding(content: str, user_id: str, is_llm: bool,msg_id: int):
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
        [(Id, Document)]: The id points to the source of the information stored in the database.
    """
    collection = chroma_client.get_or_create_collection(name="bot-infostore")
    logging.info(f"Retrieving relevant info for user: {user_id} with query: {query}")
    results = collection.query(
        query_texts=[query],
        n_results=10,
        where={"user_id": user_id},
    )
    if not results["ids"]:
        return
    logging.info("Retrieved: %s", results["documents"])  # noqa: F821
    return [(_id, res) for _id, res in zip(results["ids"], results["documents"])]  # type: ignore

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
