import os

from classes.Memory import MemoryItem, BaseMemory
from function.vector_store import get_connection, add_to_memory, recall_memory
from utils.embedding import get_nomic_embedding
from langchain_core.documents import Document


PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")

embedding_model = get_nomic_embedding()

# Demo workflow

# Get knowledge collection first, by default it will create one if it doesn't exist
kb = get_connection(PGVECTOR_CONN, embedding_model, "test_collection")


# ----------------| Determinationa and Extraction |----------------
from function.mem_proc import memory_extract_decision, summarize_lstm_from_query

example_query = "please remember my name is Jay"

extract_decision: bool = memory_extract_decision(example_query)

if extract_decision:
    mem: BaseMemory = summarize_lstm_from_query(example_query)

# ----------------| Add memory, must be a instance of MemoryItem |----------------
mi: MemoryItem = MemoryItem(
    user_id="Jay Hanks",
    source="QUERY",
    **mem.model_dump()
)

add_res: str = add_to_memory(mi, kb)
print(add_res)

# ----------------| recall |----------------

example_recall_query = "what is my name ?"

res: list[Document] = recall_memory(example_recall_query, "Jay Hanks", score=0.6, kb=kb)

print(res)





