import os

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from pydantic import BaseModel, Field

from classes.Memory import MemoryItem
from function.vector_store import get_connection
from utils.embedding import get_nomic_embedding


class MemoryOperation(BaseModel):
    """A Pydantic-based class to handle memory operations using PGVector for vector storage and retrieval.

    Args:
        connection_string (str): The connection string for the vector store. Default is the environment variable PGVECTOR_CONN.
        collection_name (str): The name of the collection to store the memories. Default is "lstm_memory".
        embedding_model (Embeddings): The embedding model to use for the memories. Default is the embedding model nomic embedding model from local ollama.
        kb (PGVector): The vector store to use for the memories. If not provided, it will be initialized with the connection_string, collection_name, and embedding_model.

    """

    connection_string: str = Field(default_factory=lambda: os.environ.get("PGVECTOR_CONN"))
    collection_name: str = Field(default="lstm_memory")
    embedding_model: Embeddings | None = Field(default_factory=get_nomic_embedding)
    kb: PGVector | None = Field(default=None)

    class Config:
        """Config for the MemoryOperation class."""

        arbitrary_types_allowed = True

    def __init__(self, **data: dict) -> None:
        """Initialize the MemoryOperation class with vector store connection.

        Args:
            **data: Optional configuration data including connection_string,
                   collection_name, and embedding_model

        """
        super().__init__(**data)
        if self.kb is None:
            try:
                self.kb = get_connection(self.connection_string, self.embedding_model, self.collection_name)
            except ValueError as e:
                raise ValueError(f"Invalid connection string or collection name: {e}")

    def add_to_memory(self, memory: MemoryItem) -> str:
        """Add a single memory item to the vector store.

        Args:
            memory (MemoryItem, required): The memory item containing content and metadata to store.
                                         Must have 'content' and 'metadata' attributes.

        Returns:
            str: The document ID or confirmation message from the vector store add operation.

        Raises:
            ValueError: If memory lacks required 'content' or 'metadata' attributes.

        """
        if not hasattr(memory, "content") or not memory.content:
            raise ValueError("Memory must have 'content' attribute and it cannot be empty")
        if not hasattr(memory, "metadata") or not memory.metadata:
            raise ValueError("Memory must have 'metadata' attribute and it cannot be empty")

        doc = Document(
            page_content=memory.content,
            metadata=memory.metadata,
        )

        return self.kb.add_documents([doc])

    def recall_memory(self, query: str, user_id: int, score: float = 0.6, k: int = 10) -> list[Document]:
        """Search and retrieve memories from the vector store based on semantic similarity.

        Args:
            query (str, required): The search query text for semantic similarity matching.
            user_id (int, required): The user ID to filter memories (only returns memories for this user).
            score (float, optional): Minimum similarity score threshold for results.
                                   Default is 0.6. Range: 0.0 to 1.0, where 1.0 is exact match.
            k (int, optional): Maximum number of results to return. Default is 10.

        Returns:
            List[Document]: List of Document objects matching the query, ordered by similarity score.
                           Each Document contains page_content and metadata fields.
                           Returns empty list if no matches above threshold found.

        """
        return self.kb.similarity_search(query=query, filter={"user_id": user_id}, k=k, score_threshold=score)

    def delete_memory(self, memory_id: str | list[str]) -> str:
        """Delete one or multiple memories from the vector store by their IDs.

        Args:
            memory_id (str or List[str], required): Single memory ID as string, or list of memory IDs to delete.
                                                   Cannot be None or empty.

        Returns:
            str: Confirmation message or result status from the delete operation.

        """
        if memory_id is None:
            raise ValueError("Memory id cannot be None")

        target_ids: list[str] = []

        if isinstance(memory_id, list):
            if not memory_id:
                raise ValueError("Memory ID list cannot be empty")
            target_ids = memory_id
        else:
            if not memory_id or not memory_id.strip():
                raise ValueError("Memory ID cannot be empty string")
            target_ids.append(memory_id)

        return self.kb.delete(ids=target_ids, collection_only=True)

    def update_memory(self, premise_memory: Document, new_memory: MemoryItem) -> str:
        """Update an existing memory by replacing it with new content and metadata.

        This operation performs an add-then-delete sequence to update the memory,
        which ensures data consistency by only removing the old memory after
        successfully adding the new one.

        Args:
            premise_memory (Document, required): The existing memory document to be updated.
                                               Must have a valid 'id' attribute.
            new_memory (MemoryItem, required): The new memory item containing updated content and metadata.
                                             Must have 'content' and 'metadata' attributes.

        Returns:
            str: The document ID or confirmation message from the add operation for the new memory.

        Note:
            This operation adds the new memory first, then deletes the old one to ensure
            data consistency. If deletion fails, both memories will exist temporarily.

        """
        if not hasattr(premise_memory, "id") or not premise_memory.id:
            raise AttributeError("premise_memory must have a valid 'id' attribute")

        # First, add the new memory to ensure it's successfully stored
        try:
            new_memory_result = self.add_to_memory(new_memory)
        except Exception as e:
            raise ConnectionError(f"Failed to add updated memory: {e}")

        # Then, delete the old memory
        try:
            self.kb.delete(ids=[premise_memory.id], collection_only=True)
        except KeyError as _:
            # If old memory doesn't exist, that's actually fine - The new memory is already added successfully
            pass
        except Exception as e:
            # If deletion fails, we have both memories but new one is added
            raise ConnectionError(f"New memory added successfully but failed to delete old memory: {e}")

        return new_memory_result
