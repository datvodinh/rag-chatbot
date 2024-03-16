import gradio as gr
import requests
import json


def send_message(message: str, history: str = None):
    url = "http://host.docker.internal:11434/api/generate"
    payload = {
        "model": "phi:2.7b",
        "prompt": message
    }
    response = requests.post(url, json=payload, stream=True)
    text = []
    for data in response.iter_lines(chunk_size=1, decode_unicode=True):
        text.append(json.loads(data.decode()).get("response", ""))
        yield "".join(text)


demo = gr.ChatInterface(fn=send_message, examples=["Why is the sky blue?"],
                        title="LLM + RAG", theme=gr.themes.Soft(primary_hue="green"))
demo.launch(share=False, server_name="0.0.0.0")
