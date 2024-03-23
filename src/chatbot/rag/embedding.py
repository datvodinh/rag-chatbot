import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()


class LocalEmbedding:
    def __init__(self) -> None:
        pass

    @staticmethod
    def set(model_name: str = "text-embedding-ada-002"):
        if model_name != "text-embedding-ada-002":
            return HuggingFaceEmbedding(
                model_name,
                cache_folder=os.path.join(os.getcwd(), "data/huggingface"),
                trust_remote_code=True
            )
        else:
            return OpenAIEmbedding()
