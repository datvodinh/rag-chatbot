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
    with gr.Tab("Interface"):
        with gr.Row(variant='panel', equal_height=False):
            with gr.Column(variant='panel', scale=10) as setting:
                with gr.Column():
                    status = gr.Textbox(
                        label="Status",
                        value="Ready!",
                        interactive=False,
                    )

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
                        label="Enter model or choose below",
                        choices=[
                            "llama3:8b-instruct-q8_0",
                            "starling-lm:7b-beta-q8_0",
                            "mixtral:instruct",
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

                    documents = gr.Files(
                        label="Add Documents",
                        value=[],
                        file_types=[".txt", ".pdf", ".csv"],
                        file_count="multiple",
                        height=150
                    )

            with gr.Column(scale=30, variant="panel"):
                chatbot = gr.Chatbot(
                    layout='bubble', likeable=True,
                    value=[], height=500, scale=2,

                    show_copy_button=True,
                    bubble_full_width=False,
                    avatar_images=["./assets/user.png", "./assets/bot.png"]
                )
                with gr.Row(variant='panel'):
                    message = gr.Textbox(label="Enter Query:", scale=5, lines=1)
                with gr.Row(variant='panel'):
                    ui_btn = gr.Button(value="Show/Hide Setting", min_width=20)
                    undo_btn = gr.Button(value="Undo", min_width=20)
                    clear_btn = gr.Button(value="Clear", min_width=20)
                    reset_btn = gr.Button(value="Reset", min_width=20)
                    sidebar_state = gr.State(False)

                    @ui_btn.click(inputs=[sidebar_state], outputs=[setting, sidebar_state])
                    def show_hide_setting(state):
                        state = not state
                        return gr.update(visible=state), state

    with gr.Tab("Output"):
        with gr.Row(variant="panel"):
            log = gr.Code(label="", language="markdown", interactive=False, lines=30)
            # log = gr.TextArea(interactive=False, lines=30, max_lines=30, show_copy_button=True)
        demo.load(logger.read_logs, None, log, every=1, show_progress="hidden", scroll_to_output=True)

    with gr.Tab("Setting"):
        with gr.Row(variant='panel', equal_height=False):
            with gr.Column():
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=rag_pipeline.get_system_prompt(),
                    interactive=True,
                    lines=20,
                    max_lines=50
                )
                sys_prompt_btn = gr.Button(value="Set System Prompt")

    @message.submit(inputs=[model, message, chatbot, chat_mode], outputs=[message, chatbot, status])
    def get_respone(model, message, chatbot, mode, progress=gr.Progress(track_tqdm=True)):
        if model in [None, ""]:
            gr.Warning("You need to set model first!")
            return "", [], "Ready!"
        else:
            console = sys.stdout
            sys.stdout = Logger(LOG_FILE)
            user_mess = message
            all_text = []
            for text in rag_pipeline.query(user_mess, mode):
                all_text.append(text)
                yield "", chatbot + [[user_mess, "".join(all_text)]], "Answering!"
            yield "", chatbot + [[user_mess, "".join(all_text)]], "Completed!"
            sys.stdout = console

    @clear_btn.click(outputs=[message, chatbot, status])
    @model.change(outputs=[message, chatbot, status])
    @chat_mode.change(outputs=[message, chatbot])
    def clear_chat():
        return "", [], "Ready!"

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

    @reset_btn.click(outputs=[message, chatbot, documents, status])
    def reset_chat():
        rag_pipeline.reset_conversation()
        return "", [], None, "Ready!"

    @pull_btn.click(inputs=[model], outputs=[message, chatbot, status])
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
        return "", [], "Ready!"

    @chat_mode.change(inputs=[documents, model, language, chat_mode], outputs=[status, system_prompt])
    @documents.change(inputs=[documents, model, language, chat_mode], outputs=[status, system_prompt])
    def processing_document(
        document, model, language, mode,
        progress=gr.Progress(track_tqdm=True)
    ):
        if document not in [None, []]:
            gr.Info("Processing Document!")
            if args.host == "host.docker.internal":
                for file_path in document:
                    shutil.move(src=file_path, dst=os.path.join(INPUT_DIR, file_path.split("/")[-1]))
                nodes = rag_pipeline.get_nodes_from_file(input_dir=INPUT_DIR)
            else:
                nodes = rag_pipeline.get_nodes_from_file(input_files=document)
            gr.Info("Indexing!")
            rag_pipeline.store_nodes(nodes)
            rag_pipeline.set_language(language)
            rag_pipeline.set_model(model)
            rag_pipeline.set_engine(mode)
            gr.Info("Processing Completed!")
            return "Completed!", rag_pipeline.get_system_prompt()
        else:
            return "Empty Documents!", rag_pipeline.get_system_prompt()

    @language.change(inputs=[model, language, chat_mode])
    def change_language(model, language, mode):
        using_rag = rag_pipeline.check_nodes_exist()
        rag_pipeline.set_language(language)
        rag_pipeline.set_system_prompt_by_lang(language, using_rag)
        rag_pipeline.set_model(model)
        if using_rag:
            rag_pipeline.set_engine(mode)
        gr.Info(f"Change language to {language}")

    @sys_prompt_btn.click(inputs=[model, system_prompt, chat_mode])
    def change_system_prompt(model, sys_prompt, mode):
        rag_pipeline.set_system_prompt(sys_prompt)
        rag_pipeline.set_model(model)
        if rag_pipeline.check_nodes_exist():
            rag_pipeline.set_engine(mode)
        gr.Info("System prompt updated!")


demo.launch(share=args.share, server_name="0.0.0.0", debug=False, show_api=False)
