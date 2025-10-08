import base64
import logging
from os import getenv
from chromadb import AsyncHttpClient as ChromaClient
from chromadb.api import AsyncClientAPI
import dspy


class EmbeddingStore:
    client: AsyncClientAPI

    @classmethod
    async def create(cls):
        self = EmbeddingStore()
        ssl = False
        headers = {}
        if getenv("CHROMA_KEY"):
            headers = {"x-chroma-token": getenv("CHROMA_KEY")}
            ssl = True
        self.client = await ChromaClient(
            host=getenv("CHROMA_HOST"),  # type: ignore
            port=int(getenv("CHROMA_PORT")),  # type: ignore
            tenant=getenv("CHROMA_TENANT"),  # type: ignore
            headers=headers,  # type: ignore
            ssl=ssl,
            database=getenv("CHROMA_DB"),  # type: ignore
        )

        try:  # Just initialise collections on a fresh instance
            await self.client.get_or_create_collection(
                name="bot-infostore", configuration={"hnsw": {"space": "cosine"}}
            )
            await self.client.get_or_create_collection(
                name="bot-msgstore", configuration={"hnsw": {"space": "cosine"}}
            )
        except Exception as e:
            logging.exception(f"Error setting up Chroma collections: {e}")

        return self

    async def insert_info_embedding(
        self, summary: str, user_id: int, info_id: int, msg_id: int, has_img: bool
    ):
        collection = await self.client.get_collection(name="bot-infostore")
        logging.info(f"Inserting embedding for info_id: {info_id}")

        await collection.add(
            ids=[str(info_id)],
            documents=[summary],
            metadatas=[
                {
                    "user_id": user_id,
                    "info_id": info_id,
                    "msg_id": msg_id,
                    "has_img": has_img,
                }
            ],
        )

    async def insert_message_embedding(
        self, content: str, user_id: int, is_llm: bool, msg_id: int
    ):
        collection = await self.client.get_collection(name="bot-msgstore")
        logging.info(f"Inserting embedding for msg_id: {msg_id}")

        await collection.add(
            ids=[str(msg_id)],
            documents=[content],
            metadatas=[{"user_id": user_id, "msg_id": msg_id, "is_llm": is_llm}],
        )

    async def retrieve_relevant_info(self, query: str, user_id: int):
        """Retrieve relevant information for a given query and user.
        Returns:
            [(msg_id, Distance, Document)]: The id points to the source of the information stored in the database
                and Document is the summary of the information.
        """
        collection = await self.client.get_collection(name="bot-infostore")
        logging.info(
            f"Retrieving relevant info for user: {user_id} with query: {query}"
        )
        results = await collection.query(
            query_texts=[query],
            n_results=10,
            include=["documents", "metadatas", "distances"],
            where={"user_id": user_id},
        )

        if not results["ids"]:
            return

        data = sorted(
            [
                (meta["msg_id"], dist, res)
                for meta, dist, res in zip(
                    results["metadatas"][0],  # type: ignore
                    results["distances"][0],  # type: ignore
                    results["documents"][0],  # type: ignore
                )
            ],
            key=lambda x: x[1],
        )
        logging.info("Retrieved data from msg_ids: %s", [m_id[0] for m_id in data])  # noqa: F821

        return data

    async def retrieve_relevant_messages(self, query: str, user_id: str):
        """Retrieve relevant past messages for a given query and user.
        Returns:
            [Document]: The past messages sent by the user.
        """
        collection = await self.client.get_collection(name="bot-msgstore")
        results = await collection.query(
            query_texts=[query],
            n_results=10,
            where={"user_id": user_id, "is_llm": False},
        )
        if not results["ids"]:
            return
        logging.info("Retrieved Messages: %s", results["documents"])  # noqa: F821
        return [res for res in results["documents"]]  # type: ignore


def convert_image(file: bytes) -> dspy.Image:
    return dspy.Image.from_file(
        "data:image/png;base64," + base64.b64encode(file).decode("utf-8")
    )
