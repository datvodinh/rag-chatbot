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
    port: int = 11434


class LLMSettings(BaseModel):
    context_window: int = 8000
    temperature: float = 0.1


class RetrieverSettings(BaseModel):
    num_queries: int = 5
    similarity_top_k: int = 20
    retriever_weights: List[float] = [0.4, 0.6]
    top_k_rerank: int = 4
    rerank_llm: str = "BAAI/bge-reranker-large"
    fusion_mode: str = "dist_based_score"
    chat_token_limit: int = 3000


class IngestionSettings(BaseModel):
    embed_llm: str = "BAAI/bge-large-en-v1.5"
    embed_batch_size: int = 2
    cache_folder: str = "data/huggingface"
    chunk_size: int = 1024
    chunk_overlap: int = 32
    chunking_regex: str = "[^,.;。？！]+[,.;。？！]?"
    paragraph_sep: str = "\n\n\n"
    num_workers: int = 0


class StorageSettings(BaseModel):
    persist_dir_chroma: str = "data/chroma"
    persist_dir_storage: str = "data/storage"
    collection_name: str = "data"
    port: int = 8000
