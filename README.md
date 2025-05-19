![banner](public/banner.png)

Welcome to CareMemo. This is an intelligent memory system designed to support caregivers of individuals with Alzheimer's Disease and Related Dementias (ADRD). By leveraging AI capabilities to maintain contextual memory across conversations, CareMemo delivers personalized and empathetic assistance to alleviate caregiver burden and enhance care quality.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Features](#features)
- [System Architecture](#system-architecture)

## Getting Started

### Prerequisites

Python 3.10.4 or higher
PostgreSQL with pgvector extension (for vector storage)
Access to LLM services (DeepSeek, OpenAI, or Ollama)-

### Environment Setup

Create a `.env` file in the project root with the following variables:

```
PGVECTOR_CONN=postgresql://username:password@localhost:5432/carememo
DEEPSEEK_API=your_api

You also need to locally install ollama and nomic embedding model to use this repo.
```

## Features

CareMemo utilizes a sophisticated memory architecture that includes working memory, short-term memory (STM), and long-term memory (LTM) components. This enables the system to remember important information about the user and their care recipient, facilitating more personalized and context-aware interactions over time.

**Memory Classification System**: Automatically categorizes information as either long-term or short-term memory

**Memory Extraction**: Extracts relevant information from user conversations

**Conversational Memory**: Maintains context across multiple interactions

**Vector-Based Retrieval**: Efficiently retrieves relevant memories using semantic search

**Multi-Category Support**: Organizes memories across categories like ADRD info, caregiving, bio info, social connections, and preferences

**Personalized Responses**: Generates context-aware responses that incorporate previously stored memories

## System Architecture

CareMemo's architecture consists of several key components:

**Memory Classes**: Defines the structure for memory items with attributes like level, category, type, and content

**Memory Processing**: Determines when to store information and how to categorize it

**Vector Storage**: Persists long-term memories in a vector database for efficient retrieval

**Answer Generation**: Creates contextually appropriate responses using retrieved memories and conversation history

## Example

```python
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
```