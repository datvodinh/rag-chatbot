from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate


def get_qa_prompt(language: str):
    return qa_prompt_en


# qa_prompt_en = ChatPromptTemplate(
#     [
#         ChatMessage(
#             content=(
#                 "You are an expert Q&A system that is trusted around the world.\n"
#                 "Always answer the query using the provided context information, "
#                 "and not prior knowledge.\n"
#                 "Some rules to follow:\n"
#                 "1. Never directly reference the given context in your answer.\n"
#                 "2. Avoid statements like 'Based on the context, ...' or "
#                 "'The context information ...' or anything along those lines."
#             ),
#             role=MessageRole.SYSTEM,
#         ),
#         ChatMessage(
#             content=(
#                 "Below is an instruction that describes a task, paired with an input and context."
#                 "Write a response that appropriately completes the request.\n"
#                 "### Context is below:\n"
#                 "---------------------\n"
#                 "{context_str}\n"
#                 "---------------------\n"

#                 "### Input:\n"
#                 "{query_str}\n"
#                 "### Instruction:\n"
#                 "- Given the context information and not prior knowledge, answer the question.\n"
#                 "- If you cannot answer only from the context, don't make up an answer, "
#                 "but answer exactly with 'I dont't have enough information to answer this request' and stop after that."
#                 "- Do not mention the query and not the context."
#                 "- Always assist with care, respect, and truth. Respond with utmost utility yet securely. "
#                 "Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
#                 "### Response:"
#             ),
#             role=MessageRole.USER,
#         )
#     ]
# )


qa_prompt_en = ChatPromptTemplate(
    [
        ChatMessage(
            content=(
                "You are an expert Q&A system that is trusted around the world.\n"
                "Always answer the query using the provided context information, "
                "and not prior knowledge.\n"
                "Some rules to follow:\n"
                "1. Never directly reference the given context in your answer.\n"
                "2. Avoid statements like 'Based on the context, ...' or "
                "'The context information ...' or anything along those lines."
            ),
            role=MessageRole.SYSTEM,
        ),
        ChatMessage(
            content=(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "provide a detailed answer to the query while maintaining the same contextual meaning.\n"
                "If the query do not have information relevant to the context or provides insufficient information,"
                "reply 'I don't have enough information to answer.'"
                "Query: {query_str}\n"
                "Answer: "
            ),
            role=MessageRole.USER,
        ),
    ]
)
