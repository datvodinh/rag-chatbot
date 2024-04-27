import os
import shutil
import json
import argparse
import sys
import socket
import gradio as gr
import llama_index
from rag_chatbot import LocalRAGPipeline, run_ollama_server, Logger

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

LOG_FILE = "logging.log"
llama_index.core.set_global_handler("simple")
logger = Logger(LOG_FILE)
logger.reset_logs()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="host.docker.internal",
    help="Set host to local or in docker container"
)
parser.add_argument(
    "--share", action='store_true',
    help="Share gradio app"
)
args = parser.parse_args()


INPUT_DIR = os.path.join(os.getcwd(), "data/data")
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
rag_pipeline = LocalRAGPipeline(host=args.host)

if args.host != "host.docker.internal":
    def is_port_open(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(('localhost', port))
                return True
            except ConnectionRefusedError:
                return False
    port_number = 11434
    if not is_port_open(port_number):
        run_ollama_server()

with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate"), js=js_func) as demo:
    gr.Markdown("# Chat with Multiple PDFs 🤖")
    with gr.Tab("Chatbot Interface"):
        with gr.Row(variant='panel', equal_height=False):
            with gr.Column(variant='panel', scale=10):
                with gr.Column():
                    chat_mode = gr.Radio(
                        label="Mode",
                        choices=["chat", "compact"],
                        value="compact",
                        interactive=True
                    )
                    language = gr.Radio(
                        label="Language",
                        choices=["vi", "eng"],
                        value="eng",
                        interactive=True
                    )
                    model = gr.Dropdown(
                        label="Set Model",
                        choices=[
                            "starling-lm:7b-beta-q6_K",
                            "llama3:8b-instruct-q6_K",
                            "nous-hermes2:10.7b-solar-q4_K_M",
                            "codeqwen:7b-chat-v1.5-q5_1",
                        ],
                        value=None,
                        interactive=True,
                        allow_custom_value=True
                    )
                    with gr.Row():
                        pull_btn = gr.Button("Pull Model", visible=False, min_width=50)
                        cancel_btn = gr.Button("Cancel", visible=False, min_width=50)

                doc_progress = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                )

                documents = gr.Files(
                    label="Add Documents",
                    value=[],
                    file_types=[".txt", ".pdf", ".csv"],
                    file_count="multiple",
                    height=150
                )

            with gr.Column(scale=30, variant="panel"):
                chatbot = gr.Chatbot(layout='bubble', value=[], height=500, scale=2)
                with gr.Row(variant='panel'):
                    message = gr.Textbox(label="Enter Prompt:", scale=5, lines=1)
                with gr.Row(variant='panel'):
                    reset_btn = gr.Button(value="Reset")
                    clear_btn = gr.Button(value="Clear")
                    undo_btn = gr.Button(value="Undo")

    with gr.Tab("Retrieval Process"):
        with gr.Row(variant="panel"):
            log = gr.Code(label="", language="markdown", interactive=False, lines=30)
            # log = gr.TextArea(interactive=False, lines=30, max_lines=30, show_copy_button=True)
        demo.load(logger.read_logs, None, log, every=1, show_progress="hidden", scroll_to_output=True)

    # @send_btn.click(inputs=[message, chatbot, chat_mode], outputs=[message, chatbot])
    @message.submit(inputs=[model, message, chatbot, chat_mode], outputs=[message, chatbot])
    def get_respone(model, message, chatbot, mode, progress=gr.Progress(track_tqdm=True)):
        if model in [None, ""]:
            gr.Warning("You need to set model first!")
            return "", []
        else:
            console = sys.stdout
            sys.stdout = Logger(LOG_FILE)
            user_mess = message
            all_text = []
            for text in rag_pipeline.query(user_mess, mode):
                all_text.append(text)
                yield "", chatbot + [[user_mess, "".join(all_text)]]

            sys.stdout = console

    @clear_btn.click(outputs=[message, chatbot])
    @model.change(outputs=[message, chatbot])
    @chat_mode.change(outputs=[message, chatbot])
    def clear_chat():
        return "", []

    @model.change(inputs=[model], outputs=[pull_btn, cancel_btn])
    def get_confirm_pull_model(model):
        if (model in ["gpt-3.5-turbo", "gpt-4"]) \
                or (rag_pipeline.check_exist(model)) \
                or (model in [None, ""]):
            if model not in [None, ""]:
                set_model(model)
            return gr.update(visible=False), gr.update(visible=False)
        return gr.update(visible=True), gr.update(visible=True)

    @pull_btn.click(outputs=[pull_btn, cancel_btn])
    @cancel_btn.click(outputs=[pull_btn, cancel_btn])
    def hide_model_button():
        return gr.update(visible=False), gr.update(visible=False)

    @cancel_btn.click(outputs=[model])
    def clear_model_choie():
        return None

    @undo_btn.click(inputs=[message, chatbot], outputs=[message, chatbot])
    def undo_chat(message, history):
        if len(history) > 0:
            history.pop(-1)
            return "", history
        return "", []

    @reset_btn.click(outputs=[message, chatbot, documents])
    def reset_chat():
        rag_pipeline.reset_conversation()
        return "", [], None

    @pull_btn.click(inputs=[model], outputs=[message, chatbot])
    def set_model(model, progress=gr.Progress(track_tqdm=True)):
        if model in ["gpt-3.5-turbo", "gpt-4"]:
            rag_pipeline.set_model(model)
        elif not rag_pipeline.check_exist(model):
            response = rag_pipeline.pull_model(model)
            if response.status_code == 200:
                gr.Info(f"Pulling {model}!")
                for data in response.iter_lines(chunk_size=1):
                    data = json.loads(data)
                    if 'completed' in data.keys() and 'total' in data.keys():
                        progress(data['completed'] / data['total'], desc="Downloading")
                    else:
                        progress(0.)
            else:
                gr.Warning(f"Model {model} doesn't exist!")
            rag_pipeline.set_model(model)
        else:
            rag_pipeline.set_model(model)
        gr.Info(f"Model {model} is ready!")
        return "", []

    @chat_mode.change(inputs=[documents, language, chat_mode], outputs=[doc_progress])
    @documents.change(inputs=[documents, language, chat_mode], outputs=[doc_progress])
    def processing_document(
        document, language, mode,
        progress=gr.Progress(track_tqdm=True)
    ):
        if document is not None:
            gr.Info("Processing Document!")
            if args.host == "host.docker.internal":
                for file_path in document:
                    shutil.move(src=file_path, dst=os.path.join(INPUT_DIR, file_path.split("/")[-1]))
                nodes = rag_pipeline.get_nodes_from_file(input_dir=INPUT_DIR)
            else:
                nodes = rag_pipeline.get_nodes_from_file(input_files=document)
            gr.Info("Indexing!")
            rag_pipeline.store_nodes(nodes)
            rag_pipeline.set_engine(language, mode)
            gr.Info("Processing Completed!")
            return "Completed!"
        else:
            return "Empty Documents"

    @language.change(inputs=[language])
    def change_language(language):
        gr.Info(f"Change language to {language}")


demo.launch(share=args.share, server_name="0.0.0.0", debug=False, show_api=False)
