from llama_index.core import PromptTemplate


def get_query_gen_prompt(language: str):
    if language == "vi":
        return query_gen_prompt_vi
    return query_gen_prompt_en


query_gen_prompt_vi = PromptTemplate(
    "Bạn là một người tạo truy vấn tìm kiếm tài năng, cam kết cung cấp các truy vấn tìm kiếm chính xác và liên quan, ngắn gọn, cụ thể và không mơ hồ.\n"
    "Tạo ra {num_queries} truy vấn tìm kiếm độc đáo và đa dạng, mỗi truy vấn trên một dòng, liên quan đến truy vấn đầu vào sau đây:\n"
    "### Truy vấn Gốc: {query}\n"
    "### Vui lòng cung cấp các truy vấn tìm kiếm mà:\n"
    "- Liên quan đến truy vấn gốc\n"
    "- Được xác định rõ ràng và cụ thể\n"
    "- Không mơ hồ và không thể hiểu sai\n"
    "- Hữu ích để lấy kết quả tìm kiếm chính xác và liên quan\n"
    "### Các Truy Vấn Được Tạo Ra:\n"
)

query_gen_prompt_en = PromptTemplate(
    "You are a skilled search query generator, dedicated to providing accurate and relevant search queries that are concise, specific, and unambiguous.\n"
    "Generate {num_queries} unique and diverse search queries, one on each line, related to the following input query:\n"
    "### Original Query: {query}\n"
    "### Please provide search queries that are:\n"
    "- Relevant to the original query\n"
    "- Well-defined and specific\n"
    "- Free of ambiguity and vagueness\n"
    "- Useful for retrieving accurate and relevant search results\n"
    "### Generated Queries:\n"
)
