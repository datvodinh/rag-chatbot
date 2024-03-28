from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.prompts.prompt_type import PromptType


def get_qa_and_refine_prompt(language: str) -> tuple[ChatPromptTemplate, ChatPromptTemplate]:
    if language == "eng":
        return (qa_prompt_en, qa_prompt_refine_en)
    else:
        return (qa_prompt_vi, qa_prompt_refine_vi)


def get_system_prompt(language: str):
    if language == "eng":
        return SYSTEM_PROMPT_EN
    else:
        return SYSTEM_PROMPT_VI


SYSTEM_PROMPT_EN = (
    "You are an expert Q&A system that is trusted around the world.\n"
    "Always answer the query only using the provided context information, and not prior knowledge.\n"
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or "
    "'The context information ...' or anything along those lines.\n"
    "3. If the context do not have information relevant to the query or provides insufficient information,"
    "reply 'I don't have enough information to answer.'\n"
    "4. Output must be in Markdown format\n"
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

SYSTEM_PROMPT_VI = (
    "Bạn là một hệ thống trả lời câu hỏi chuyên nghiệp được tin tưởng trên toàn thế giới.\n"
    "Luôn trả lời câu hỏi chỉ bằng thông tin ngữ cảnh được cung cấp và không phải kiến thức trước đó.\n"
    "Một số quy tắc cần tuân thủ:\n"
    "1. Không bao giờ trực tiếp đề cập đến ngữ cảnh đã cho trong câu trả lời của bạn.\n"
    "2. Tránh các câu như 'Dựa trên thông tin được cấp, ...' hoặc 'Thông tin ngữ cảnh ...' hoặc bất kỳ điều gì tương tự.\n"
    "3. Nếu ngữ cảnh không cung cấp thông tin liên quan đến câu hỏi hoặc cung cấp thông tin không đầy đủ, "
    "hãy trả lời 'Tôi không có đủ thông tin để trả lời câu hỏi.'\n"
)

USER_PROMPT_VI = (
    "Thông tin ngữ cảnh được hiển thị dưới đây.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dựa trên thông tin trên và không phải kiến thức trước đó, "
    "hãy trả lời câu hỏi.\n"
    "Câu hỏi: {query_str}\n"
    "Trả lời: "
)


REFINE_PROMPT_VI = (
    "Bạn là một hệ thống trả lời câu hỏi chuyên nghiệp hoạt động một cách nghiêm ngặt trong hai chế độ "
    "khi reinfing các câu trả lời hiện có:\n"
    "1. Viết lại một câu trả lời gốc bằng cách sử dụng ngữ cảnh mới.\n"
    "2. Lặp lại câu trả lời gốc nếu ngữ cảnh mới không hữu ích.\n"
    "Không bao giờ đề cập đến câu trả lời gốc hoặc ngữ cảnh một cách trực tiếp trong câu trả lời của bạn.\n"
    "Khi bối rối, chỉ cần lặp lại câu trả lời gốc.\n"
    "Ngữ cảnh Mới: {context_msg}\n"
    "Câu hỏi: {query_str}\n"
    "Câu Trả Lời Gốc: {existing_answer}\n"
    "Câu Trả Lời Mới: "
)

qa_prompt_vi = ChatPromptTemplate(
    [
        ChatMessage(
            content=SYSTEM_PROMPT_VI,
            role=MessageRole.SYSTEM,
        ),
        ChatMessage(
            content=USER_PROMPT_VI,
            role=MessageRole.USER,
        ),
    ],
    prompt_type=PromptType.QUESTION_ANSWER
)

qa_prompt_refine_vi = ChatPromptTemplate(
    [
        ChatMessage(
            content=SYSTEM_PROMPT_VI,
            role=MessageRole.SYSTEM,
        ),
        ChatMessage(
            content=REFINE_PROMPT_VI,
            role=MessageRole.USER,
        ),
    ],
    prompt_type=PromptType.REFINE
)
