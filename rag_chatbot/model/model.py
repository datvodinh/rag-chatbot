from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from rag_chatbot.setting import OllamaSettings, LLMSettings
from dotenv import load_dotenv
import requests

load_dotenv()

llm_settings = LLMSettings()
ollama_settings = OllamaSettings()


class LocalRAGModel:
    def __init__(self) -> None:
        pass

    @staticmethod
    def set(
        model_name: str = "llama3:8b-instruct-q8_0",
        system_prompt: str | None = None,
        host: str = "host.docker.internal"
    ):

        if model_name in ["gpt-3.5-turbo", "gpt-4"]:
            return OpenAI(
                model=model_name,
                temperature=llm_settings.temperature
            )
        else:
            settings_kwargs = {
                "tfs_z": ollama_settings.tfs_z,
                "top_k": ollama_settings.top_k,
                "top_p": ollama_settings.top_p,
                "repeat_last_n": ollama_settings.repeat_last_n,
                "repeat_penalty": ollama_settings.repeat_penalty,
            }
            return Ollama(
                model=model_name,
                system_prompt=system_prompt,
                base_url=f"http://{host}:{ollama_settings.port}",
                temperature=llm_settings.temperature,
                context_window=llm_settings.context_window,
                request_timeout=ollama_settings.request_timeout,
                additional_kwargs=settings_kwargs
            )

    @staticmethod
    def pull(host: str, model_name: str):
        payload = {
            "name": model_name
        }
        return requests.post(
            f"http://{host}:{ollama_settings.port}/api/pull",
            json=payload, stream=True
        )

    @staticmethod
    def check_model_exist(host: str, model_name: str) -> bool:
        data = requests.get(
            f"http://{host}:{ollama_settings.port}/api/tags"
        ).json()
        list_model = [d["name"] for d in data["models"]]
        if model_name in list_model:
            return True
        return False
