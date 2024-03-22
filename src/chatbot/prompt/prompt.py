
def get_prompt_format(language: str):
    pass


default_prompt_vi = (

    "Lịch sử trò chuyện:\n"
    "---------------------\n"
    "{history}\n"
    "---------------------\n"
    "Dựa trên lịch sử và kiến thức của bạn, vui lòng cung cấp câu trả lời cho câu hỏi sau đây:\n"
    "Câu hỏi: {question}\n"
    "Trả lời: "
)

default_prompt_eng = (
    "Chat history:\n"
    "---------------------\n"
    "{history}\n"
    "---------------------\n"
    "Based solely on your prior knowledge and the chat history (if exist), please provide an answer to the following question:\n"
    "Question: {question}\n"
    "Answer: "
)

rag_prompt_vi = (
    "Lịch sử trò chuyện:\n"
    "---------------------\n"
    "{history}\n"
    "---------------------\n"
    "Thông tin cung cấp:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Dựa hoàn toàn vào thông tin được cung cấp và lịch sử trò chuyện,"
    "hãy cung cấp một câu trả lời chi tiết cho câu hỏi với việc giữ nguyên ý nghĩa ngữ cảnh.\n"
    "Nếu không có đủ thông tin, vui lòng đề cập đến điều đó.\n"
    "Câu hỏi: {question}\n"
    "Trả lời: "
)

rag_prompt_eng = (
    "Chat History:\n"
    "---------------------\n"
    "{history}\n"
    "---------------------\n"
    "Context Information:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Based solely on the provided Context Information and the Chat History,"
    "provide a detailed answer to the question while maintaining the same contextual meaning.\n"
    "If the context provides insufficient information and the question cannot be directly answered,"
    "reply 'Based on the context I cannot answer.'"
    "Question: {question}\n"
    "Answer: "
)
