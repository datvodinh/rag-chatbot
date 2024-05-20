from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from ...setting import RAGSettings
from dotenv import load_dotenv
import requests

load_dotenv()


class LocalRAGModel:
    def __init__(self) -> None:
        pass

    @staticmethod
    def set(
        model_name: str = "llama3:8b-instruct-q8_0",
        system_prompt: str | None = None,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None
    ):
        setting = setting or RAGSettings()
        if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo"]:
            return OpenAI(
                model=model_name,
                temperature=setting.ollama.temperature
            )
        else:
            settings_kwargs = {
                "tfs_z": setting.ollama.tfs_z,
                "top_k": setting.ollama.top_k,
                "top_p": setting.ollama.top_p,
                "repeat_last_n": setting.ollama.repeat_last_n,
                "repeat_penalty": setting.ollama.repeat_penalty,
            }
            return Ollama(
                model=model_name,
                system_prompt=system_prompt,
                base_url=f"http://{host}:{setting.ollama.port}",
                temperature=setting.ollama.temperature,
                context_window=setting.ollama.context_window,
                request_timeout=setting.ollama.request_timeout,
                additional_kwargs=settings_kwargs
            )

    @staticmethod
    def pull(host: str, model_name: str):
        setting = RAGSettings()
        payload = {
            "name": model_name
        }
        return requests.post(
            f"http://{host}:{setting.ollama.port}/api/pull",
            json=payload, stream=True
        )

    @staticmethod
    def check_model_exist(host: str, model_name: str) -> bool:
        setting = RAGSettings()
        data = requests.get(
            f"http://{host}:{setting.ollama.port}/api/tags"
        ).json()
        if data["models"] is None:
            return False
        list_model = [d["name"] for d in data["models"]]
        if model_name in list_model:
            return True
        return False
