import gradio as gr
import requests
import json

# TODO: implemented RAG


def clear_chat(message, history):
    return "", []


def undo_chat(message, history):
    if len(history) > 0:
        history.pop(-1)
        return "", history
    return "", []


def history_to_messages(history: list[list[str, str]]) -> list[dict[str, str]]:
    if len(history) == 0:
        return []
    else:
        messages = []
        for user_mess, bot_mess in history:
            messages.append({"role": "user", "content": user_mess})
            messages.append({"role": "assistant", "content": bot_mess})
        return messages


def messages_to_history(messages: list[dict[str, str]]) -> list[list[str, str]]:
    if len(messages) == 0:
        return []
    else:
        history = []
        holder = []
        for data in messages:
            holder.append(data['content'])
            if len(holder) == 2:
                history.append(holder)
                holder = []
        return history


def get_response(
    model: str,
    message: str,
    history: list,
    temperature: float,
    top_k: int,
    top_p: float,
    freq_penalty: float
):
    url = "http://host.docker.internal:11434/api/chat"

    user_message = {
        "role": "user",
        "content": message
    }
    messages = history_to_messages(history)
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


with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("# LLM Chatbot + RAG")
    with gr.Row(variant='compact'):
        with gr.Column(variant='compact'):
            pdf_file = gr.File()
            model = gr.Dropdown(label="Model", choices=["llama2:chat",
                                "mistral:instruct"], value="llama2:chat", interactive=True)
            with gr.Accordion("Hyperparameters"):
                temp = gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.8, step=0.05)
                top_k = gr.Slider(label="Top_k", minimum=5, maximum=100, value=20, step=5)
                top_p = gr.Slider(label="Top_p", minimum=0.8, maximum=1, value=0.9, step=0.05)
                freq_penalty = gr.Slider(label="Frequency Penalty", value=1.05, minimum=-2, maximum=2, step=0.05)

        with gr.Column(scale=2, variant="panel"):
            chatbot = gr.Chatbot(layout='bubble', value=[], scale=2)
            with gr.Row(variant='panel'):
                message = gr.Textbox(label="Enter Prompt:", scale=3, lines=1)
                send_btn = gr.Button(value="Send", scale=1)
            with gr.Row():
                clear_btn = gr.Button(value="Clear")
                undo_btn = gr.Button(value="Undo")

    # EVENT
    send_btn.click(get_response, inputs=[model, message, chatbot, temp,
                   top_k, top_p, freq_penalty], outputs=[message, chatbot])
    message.submit(get_response, inputs=[model, message, chatbot, temp,
                   top_k, top_p, freq_penalty], outputs=[message, chatbot])
    clear_btn.click(clear_chat, inputs=[message, chatbot], outputs=[message, chatbot])
    model.change(clear_chat, inputs=[message, chatbot], outputs=[message, chatbot])
    undo_btn.click(undo_chat, inputs=[message, chatbot], outputs=[message, chatbot])


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")
