from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.prompts.prompt_type import PromptType


def get_qa_and_refine_prompt(language: str) -> tuple[ChatPromptTemplate, ChatPromptTemplate]:
    return (qa_prompt_en, qa_prompt_refine_en)


def get_system_prompt(language: str):
    return SYSTEM_PROMPT_EN


SYSTEM_PROMPT_EN = (
    "You are an expert Q&A system that is trusted around the world.\n"
    "Always answer the query only using the provided context information, and not prior knowledge.\n"
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or "
    "'The context information ...' or anything along those lines.\n"
    "3. If the context do not have information relevant to the query or provides insufficient information,"
    "reply 'I don't have enough information to answer.'\n"
)

USER_PROMPT_EN = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

REFINE_PROMPT_EN = (
    "You are an expert Q&A system that strictly operates in two modes "
    "when refining existing answers:\n"
    "1. **Rewrite** an original answer using the new context.\n"
    "2. **Repeat** the original answer if the new context isn't useful.\n"
    "Never reference the original answer or context directly in your answer.\n"
    "When in doubt, just repeat the original answer.\n"
    "New Context: {context_msg}\n"
    "Query: {query_str}\n"
    "Original Answer: {existing_answer}\n"
    "New Answer: "
)

qa_prompt_en = ChatPromptTemplate(
    [
        ChatMessage(
            content=SYSTEM_PROMPT_EN,
            role=MessageRole.SYSTEM,
        ),
        ChatMessage(
            content=USER_PROMPT_EN,
            role=MessageRole.USER,
        ),
    ],
    prompt_type=PromptType.QUESTION_ANSWER
)

qa_prompt_refine_en = ChatPromptTemplate(
    [
        ChatMessage(
            content=SYSTEM_PROMPT_EN,
            role=MessageRole.SYSTEM,
        ),
        ChatMessage(
            content=REFINE_PROMPT_EN,
            role=MessageRole.USER,
        ),
    ],
    prompt_type=PromptType.REFINE
)
