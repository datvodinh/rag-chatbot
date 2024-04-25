from llama_index.core import PromptTemplate


def get_query_gen_prompt(language: str):
    if language == "vi":
        return query_gen_prompt_vi
    return query_gen_prompt_en


query_gen_prompt_vi = PromptTemplate(
    "Bạn là một trợ lý hữu ích tạo ra nhiều truy vấn tìm kiếm dựa trên một "
    "truy vấn đầu vào duy nhất. Hãy tạo ra {num_queries} truy vấn tìm kiếm, mỗi truy vấn trên một dòng, "
    "liên quan đến truy vấn đầu vào sau đây:\n"
    "Truy vấn: {query}\n"
    "Các truy vấn:\n"
)

query_gen_prompt_en = PromptTemplate(
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)
