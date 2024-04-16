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