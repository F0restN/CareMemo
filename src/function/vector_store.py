import os

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from classes.Memory import MemoryItem
from utils.contradictory_detect import ContradictionDetector
from utils.embedding import get_nomic_embedding

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")


def get_connection(connection: str, embedding_model: Embeddings, collection_name: str) -> PGVector:
    """Get a connection to the vector store. If connection does not exist, by default it will create one

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


def add_to_memory(memory: MemoryItem, kb: PGVector = None) -> str:
    """Add a memory to the vector store.

    Args:
        memory: MemoryItem
        kb: PGVector, by default using the lstm_memory collection

    Returns:
        str: result of the add_documents method

    """
    if kb is None:
        kb = get_connection(
            connection=PGVECTOR_CONN,
            embedding_model=get_nomic_embedding(),
            collection_name="lstm_memory",
        )

    doc = Document(
        page_content=memory.content,
        metadata=memory.metadata,
    )

    return kb.add_documents([doc])


def recall_memory(query: str, user_id: int, kb: PGVector, score: float = 0.6, k: int = 10) -> list[Document]:
    """Recall the memory from the vector store. Return document above cut off score if score is provided. Otherwise return all documents using default threshold.

    Args:
        query: The query to search for.
        user_id: The user id to search for.
        kb: PGVector, if None provided,by default using the lstm_memory collection
        score: Optional, the bar to filter out the results, default is 0.6.
        k: Optional, the number of results to return, default is 10.

    Returns:
        list[Document]: The list of documents found.

    """
    return similarity_search(query=query, user_id=user_id, kb=kb, score=score, k=k)


def delete_memory(memory_id: str | list[str], kb: PGVector = None):
    """Delete a memory from the vector store.

    Args:
        memory_id: The id of the memory to delete.
        kb: PGVector, if None provided,by default using the lstm_memory collection

    Returns:
        str: result of the delete_documents method

    """
    if kb is None:
        raise ValueError("KB can not be None")

    if memory_id is None:
        raise ValueError("Memory id can not be None")

    target_ids: list[str] = []
    if isinstance(memory_id, list):
        target_ids = memory_id
    else:
        target_ids.append(memory_id)

    return kb.delete(ids=target_ids, collection_only=True)


def update_memory(premise_memory: Document, new_memory: MemoryItem, kb: PGVector = None):
    """Update a memory from the vector store.

    Args:
        premise_memory: The memory to update.
        new_memory: The new memory to add.
        kb: PGVector, if None provided,by default using the lstm_memory collection

    Returns:
        str: result of the delete_documents method

    """
    if kb is None:
        msg = "KB can not be None"
        raise ValueError(msg)

    # Delete the previous one and add the new one.
    kb.delete(ids=[premise_memory.id], kb=kb)

    # Add the new memory to the vector store.
    add_to_memory(new_memory, kb=kb)
