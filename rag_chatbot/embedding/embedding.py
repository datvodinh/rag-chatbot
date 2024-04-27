import os
import torch
import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoModel, AutoTokenizer
from ..setting import IngestionSettings
from dotenv import load_dotenv


load_dotenv()
setting = IngestionSettings()


class LocalEmbedding:
    @staticmethod
    def set(**kwargs):
        model_name = setting.embed_llm
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
    def pull(host: str, **kwargs):
        payload = {
            "name": setting.embed_llm
        }
        return requests.post(f"http://{host}:11434/api/pull", json=payload, stream=True)

    @staticmethod
    def check_model_exist(host: str, **kwargs) -> bool:
        data = requests.get(f"http://{host}:11434/api/tags").json()
        list_model = [d["name"] for d in data["models"]]
        if setting.embed_llm in list_model:
            return True
        return False
