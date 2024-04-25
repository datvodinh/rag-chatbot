from llama_index.core import PromptTemplate


def get_hyde_prompt(language: str):
    if language == "vi":
        return hyde_prompt_vi
    return hyde_prompt_en


hyde_prompt_en = PromptTemplate(
    "Please write a passage to answer the question\n"
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Passage:"""\n'
)

hyde_prompt_vi = PromptTemplate(
    "Vui lòng viết một đoạn văn để trả lời câu hỏi\n"
    "Hãy cố gắng bao gồm càng nhiều chi tiết chính xác nhất có thể.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'Đoạn văn:"""\n'
)
