import os

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")


def get_connection(connection: str, embedding_model: Embeddings, collection_name: str) -> PGVector:
    """Get a connection to the vector store. If connection does not exist, by default it will create one.

    Args:
        connection: The connection string for the vector store.
        embedding_model: The embedding model to use.
        collection_name: The name of the collection to use. By default, it is "lstm_memory".

    Returns:
        PGVector: A connection to the vector store.

    """
    if not collection_name:
        msg = "Collection Name can not be empty"
        raise ValueError(msg)

    return PGVector(
        embeddings=embedding_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )


def similarity_search(query: str, user_id: str, kb: PGVector, score: float = 0.6, k: int = 10) -> list[Document]:
    """Search for similar documents in the vector store.

    Args:
        query: The query to search for.
        user_id: The user id to search for.
        kb: The vector store to search in.
        score: Optional, the bar to filter out the results, default is 0.6.
        k: Optional, the number of results to return, default is 10.

    Returns:
        list[Document]: The list of documents found.

    """
    if not kb:
        msg = "KB can not be None"
        raise ValueError(msg)

    custom_filter = {"user_id": user_id}
    res_unfiltered: list[tuple[Document, float]] = kb.similarity_search_with_score(
        query=query,
        filter=custom_filter,
        k=k,
    )
    if score is not None:
        return [doc for doc, relevancy in res_unfiltered if relevancy >= score]

    return [doc for doc, _ in res_unfiltered]

