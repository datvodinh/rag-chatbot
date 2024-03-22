from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()


class LocalRAGModel:
    def __init__(self) -> None:
        pass

    @staticmethod
    def set(
        model_name: str = "gpt-3.5-turbo",
        host: str = "host.docker.internal"
    ):
        if model_name in ["gpt-3.5-turbo", "gpt-4"]:
            return OpenAI(
                model=model_name
            )
        else:
            return Ollama(
                model=model_name,
                base_url=f"http://{host}:11434"
            )

    @staticmethod
    def pull(host: str, model_name: str):
        payload = {
            "name": model_name
        }
        return requests.post(f"{host}:11434/api/pull", json=payload, stream=True)

    @staticmethod
    def check_model_exist(host: str, model_name: str) -> bool:
        data = requests.get(f"{host}:11434/api/tags").json()
        list_model = [d["name"] for d in data["models"]]
        if model_name in list_model:
            return True
        return False
