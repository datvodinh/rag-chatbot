import requests
import json

BASE_URL = "http://host.docker.internal:11434"


class LocalLLMModel:
    def __init__(self) -> None:
        pass

    def check_model_exist(self, model_name: str) -> bool:
        data = requests.get(f"{BASE_URL}/api/tags").json()
        list_model = [d["name"] for d in data["models"]]
        if model_name in list_model:
            return True
        return False

    def pull_model(self, model_name: str):
        payload = {
            "name": model_name
        }
        return requests.post(f"{BASE_URL}/api/pull", json=payload, stream=True)

    def history_to_messages(self, history: list[list[str, str]]) -> list[dict[str, str]]:
        if len(history) == 0:
            return []
        else:
            messages = []
            for user_mess, bot_mess in history:
                messages.append({"role": "user", "content": user_mess})
                messages.append({"role": "assistant", "content": bot_mess})
            return messages

    def get_response(
        self,
        model: str,
        message: str,
        history: list,
        temperature: float,
        top_k: int,
        top_p: float,
        freq_penalty: float
    ):
        url = f"{BASE_URL}/api/chat"

        user_message = {
            "role": "user",
            "content": message
        }
        print(user_message)
        messages = self.history_to_messages(history)
        messages.append(user_message)
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "seed": 42,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "frequency_penalty": freq_penalty,
            }
        }
        response = requests.post(url, json=payload, stream=True)
        text = []
        if response.status_code == 200:
            for data in response.iter_lines(chunk_size=1, decode_unicode=True):
                text.append(json.loads(data.decode())["message"]["content"])
                yield "", history + [[message, "".join(text)]]
        else:
            return "", history
