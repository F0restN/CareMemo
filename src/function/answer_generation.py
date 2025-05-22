import time

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.prompts import PromptTemplate

from utils.llm_manager import _get_deepseek, _get_llm
from utils.PROMPT import BASIC_PROMPT, CONTRADICTION_CHECK_PROMPT


def generate_answer_with_memory(
    question: str,
    ltm_context: list[Document] | None = None,
    stm_context: list[Document] | None = None,
    conversation_history: list[AnyMessage] | None = None,
    temperature: float = 0.6,
) -> AIMessage:
    """Generate answer from context documents using LLM.

    Args:
        question: User's question
        ltm_context: List of Langchain Document objects
        stm_context: List of Langchain Document objects
        conversation_history: User conversation history
        model: Name of the Ollama model
        temperature: Model temperature

    """
    if not conversation_history:
        conversation_history = []
    if not ltm_context:
        ltm_context = []
    if not stm_context:
        stm_context = []
    if not question:
        msg = "Question required"
        raise ValueError(msg)

    llm = _get_deepseek("deepseek-chat", temperature)
    prompt = PromptTemplate(
        input_variables=["question", "ltm_context",
                         "stm_context", "conversation_history"],
        template=BASIC_PROMPT,
    )

    chain = prompt | llm

    return chain.invoke({
        "question": question,
        "ltm_context": ltm_context,
        "stm_context": stm_context,
        "conversation_history": conversation_history,
    })

def contradiction_check(sentence1: str, sentence2: str) -> bool:
    """Check if two sentences are contradictory.
    
    Args:
        sentence1: The first sentence
        sentence2: The second sentence
    """
    
    # Load the model
    # llm = _get_llm("deepseek-r1:14b", temperature=0.3)
    llm = _get_deepseek("deepseek-chat", temperature=0.3)
    
    prompt = PromptTemplate(
        input_variables=["sentence1", "sentence2"],
        template=CONTRADICTION_CHECK_PROMPT,
    )
    
    res = llm.invoke(prompt.format(sentence1=sentence1, sentence2=sentence2))
    
    return res.content.upper() == "YES"


if __name__ == "__main__":
    print(contradiction_check("My dad seems forgetting things more frequently than before.", "My dad is getting better so far."))
