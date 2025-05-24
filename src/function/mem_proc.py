import os
from typing import TYPE_CHECKING

from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector

from classes.Memory import BaseEpisodicMemory, BaseMemory, MemoryItem
from function.vector_store import (
    add_to_memory,
    similarity_search,
    update_memory,
)
from utils.contradictory_detect import ContradictionDetector
from utils.llm_manager import _get_deepseek
from utils.PROMPT import (
    EPISODIC_MEMORY_PROMPT_TEMPLATE,
    MEMORY_DETERMINATION_PROMPT_TEMPLATE,
    MEMORY_SUMMARIZATION_PROMPT,
)

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")

def format_conversation(chat_history: list[object]) -> str:
    """Format converation into clear and concise conversation list."""
    conversation = [
        f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history]
    return "\n".join(conversation)


def format_conversation_pipeline(chat_history: list[object]) -> str:
    """Format conversation into clear and concise conversation list.

    Args:
        chat_history: list[{
            id: str
            role: str
            content: str
            timestamp: str
        }]

    Returns:
        str: formatted conversation

    """
    conversation = [
        f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history
    ]
    return "\n".join(conversation)


def memory_extract_decision(query: str) -> bool:
    """Decide whether the user's query is about the life situation that is factual and not gonna change for a while."""
    llm = _get_deepseek("deepseek-chat", temperature=0.0)

    prompt = PromptTemplate.from_template(MEMORY_DETERMINATION_PROMPT_TEMPLATE)
    chain = prompt | llm

    res: AIMessage = chain.invoke({"query": query})

    try:
        return res.content.upper() == "YES"
    except Exception as e:
        return False


def summarize_lstm_from_query(query: str) -> BaseMemory:
    """Summarize the user's query into a memory item."""
    llm = _get_deepseek("deepseek-chat", temperature=0.0)
    prompt = PromptTemplate.from_template(MEMORY_SUMMARIZATION_PROMPT)
    structured_llm = prompt | llm.with_structured_output(
        schema=BaseMemory, method="function_calling", include_raw=False)

    res: BaseMemory = structured_llm.invoke({"query": query})

    return res


def summarize_episodicM_from_conversation(conversation: str) -> BaseEpisodicMemory:  # noqa: N802
    """Summarize the conversation into a episodic memory."""
    llm = _get_deepseek("deepseek-chat", temperature=0.0)
    prompt = PromptTemplate.from_template(EPISODIC_MEMORY_PROMPT_TEMPLATE)
    structured_llm = prompt | llm.with_structured_output(
        schema=BaseEpisodicMemory, method="function_calling", include_raw=False)
    return structured_llm.invoke({"conversation": conversation})


def classify_operation(query: str, user_id: str, new_memory: MemoryItem, kb: PGVector = None) -> str | MemoryItem:
    """Classify the operation of the user's query."""
    if new_memory.metadata["level"] == "STM":
        return "APPEND"

    premise_memory: list[Document] = similarity_search(query=query, user_id=user_id, kb=kb, score=0.2, k=1)

    print(f"<AI Internal>: Premise Memories are {premise_memory}")

    if premise_memory is not None and len(premise_memory) > 0:
        contradiction_detector = ContradictionDetector(model_name="nomic-ai/nomic-embed-text-v2-moe")
        if contradiction_detector.detect_contradiction_contrast(new_memory.convert_to_sentence(), premise_memory.page_content)["is_contradiction"]:
            update_memory(
                premise_memory=premise_memory[0],
                new_memory=new_memory,
                kb=kb,
            )
            return "UPDATED"

    add_to_memory(
        memory=new_memory,
        kb=kb,
    )
    return "ADDED"


if __name__ == "__main__":

    user_query = "Can you recommend activities that are suitable for someonewith dementia to engage in and enjoy?"

    res: BaseMemory = summarize_lstm_from_query(user_query)

    print(res)
