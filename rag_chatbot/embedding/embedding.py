import os
import torch
import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv


load_dotenv()


class LocalEmbedding:
    def __init__(self) -> None:
        pass

    @staticmethod
    def set(
        model_name: str = "BAAI/bge-large-en-v1.5",
        host: str = "host.docker.internal"
    ):
        if model_name != "text-embedding-ada-002":
            return HuggingFaceEmbedding(
                model=AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                ),
                cache_folder=os.path.join(os.getcwd(), "data/huggingface"),
                trust_remote_code=True,
                embed_batch_size=16
            )
        else:
            return OpenAIEmbedding()

    @staticmethod
    def pull(host: str, model_name: str):
        payload = {
            "name": model_name
        }
        return requests.post(f"http://{host}:11434/api/pull", json=payload, stream=True)

    @staticmethod
    def check_model_exist(host: str, model_name: str) -> bool:
        data = requests.get(f"http://{host}:11434/api/tags").json()
        list_model = [d["name"] for d in data["models"]]
        if model_name in list_model:
            return True
        return False
