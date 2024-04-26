from pydantic import BaseModel


class OllamaSettings(BaseModel):
    keep_alive: str = "5m"
    tfs_z: float = 1.0
    top_k: int = 40
    top_p: float = 0.9
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    request_timeout: float = 120.0


class LLMSettings(BaseModel):
    max_new_tokens: int = 256
    context_window: int = 3900
    temperature: float = 0.1


class RetrieverSettings(BaseModel):
    num_queries: int = 4
    similarity_top_k: int = 10
    top_k_rerank: int = 5
    rerank_llm: str = "cross-encoder/stsb-roberta-base"
    fusion_mode: str = "reciprocal_rerank"
    chat_token_limit: int = 3000


class IngestionSettings(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 32
    chunking_regex: str = "[^,.;。？！]+[,.;。？！]?"
    paragraph_sep: str = "\n\n\n"
    num_workers: int = 0


class StorageSettings(BaseModel):
    persist_dir_chroma: str = "data/chroma"
    persist_dir_storage: str = "data/storage"
    collection_name: str = "data"
    port: int = 8000
