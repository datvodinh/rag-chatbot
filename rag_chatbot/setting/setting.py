from pydantic import BaseModel
from typing import List


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
    num_queries: int = 6
    similarity_top_k: int = 10
    retriever_weights: List[float] = [0.4, 0.6]
    top_k_rerank: int = 5
    # rerank_llm: str = "cross-encoder/stsb-roberta-base"
    fusion_mode: str = "dist_based_score"
    chat_token_limit: int = 3000


class IngestionSettings(BaseModel):
    embed_llm: str = "intfloat/multilingual-e5-large-instruct"
    window_size: int = 5
    window_metadata_key: str = "window"
    original_text_metadata_key: str = "original_text"
    num_workers: int = 0


class StorageSettings(BaseModel):
    persist_dir_chroma: str = "data/chroma"
    persist_dir_storage: str = "data/storage"
    collection_name: str = "data"
    port: int = 8000
