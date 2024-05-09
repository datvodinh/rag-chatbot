import os
import shutil
import json
import sys
import gradio as gr
from .theme import JS_LIGHT_THEME, CSS
from ..pipeline import LocalRAGPipeline
from ..logger import Logger


class LocalChatbotUI:
    def __init__(
        self,
        pipeline: LocalRAGPipeline,
        logger: Logger,
        host: str = "host.docker.internal",
        data_dir: str = "data/data",
        avatar_images: list[str] = ["./assets/user.png", "./assets/bot.png"]
    ):
        self._pipeline = pipeline
        self._logger = logger
        self._host = host
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        self._avatar_images = [os.path.join(os.getcwd(), image) for image in avatar_images]

    def _get_respone(
        self,
        chat_mode: str,
        message: str,
        chatbot: list[list[str, str]],
        progress=gr.Progress(track_tqdm=True)
    ):
        if self._pipeline.get_model_name() in [None, ""]:
            gr.Warning("You need to set model first!")
            return "", [], "Ready!"
        elif message in [None, ""]:
            gr.Warning("You need to enter message!")
            return "", [], "Ready!"
        else:
            console = sys.stdout
            sys.stdout = self._logger
            answer = []
            response = self._pipeline.query(chat_mode, message, chatbot)
            for text in response.response_gen:
                answer.append(text)
                yield "", chatbot + [[message, "".join(answer)]], "Answering!"
            yield "", chatbot + [[message, "".join(answer)]], "Completed!"
            sys.stdout = console

    def _get_confirm_pull_model(self, model: str):
        if (model in ["gpt-3.5-turbo", "gpt-4"]) or (self._pipeline.check_exist(model)):
            self._change_model(model)
            return gr.update(visible=False), gr.update(visible=False), "Ready!"
        return gr.update(visible=True), gr.update(visible=True), "Confirm Pull Model!"

    def _pull_model(self, model: str, progress=gr.Progress(track_tqdm=True)):
        if (model not in ["gpt-3.5-turbo", "gpt-4"]) and not (self._pipeline.check_exist(model)):
            response = self._pipeline.pull_model(model)
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
                return "", [], "Pull Fail!", ""

        return "", [], "Pull Completed!", model

    def _change_model(self, model: str):
        if model not in [None, ""]:
            self._pipeline.set_model_name(model)
            self._pipeline.set_model()
            self._pipeline.set_engine()
            gr.Info(f"Change model to {model}!")
        return "Ready!"

    def _processing_document(
        self,
        document: list[str],
        progress=gr.Progress(track_tqdm=True)
    ):
        if document not in [None, []]:
            gr.Info("Processing Document!")
            if self._host == "host.docker.internal":
                for file_path in document:
                    shutil.move(src=file_path, dst=os.path.join(
                        self._data_dir, file_path.split("/")[-1]
                    ))
                nodes = self._pipeline.get_nodes_from_file(data_dir=self._data_dir)
            else:
                nodes = self._pipeline.get_nodes_from_file(input_files=document)
            gr.Info("Indexing!")
            self._pipeline.store_nodes(nodes)
            self._pipeline.set_chat_mode()
            gr.Info("Processing Completed!")
            return self._pipeline.get_system_prompt(), "Completed!"
        else:
            return self._pipeline.get_system_prompt(), "Empty Documents!"

    def _change_system_prompt(self, sys_prompt: str):
        self._pipeline.set_system_prompt(sys_prompt)
        self._pipeline.set_chat_mode()
        gr.Info("System prompt updated!")

    def _change_language(self, language: str):
        self._pipeline.set_language(language)
        self._pipeline.set_chat_mode()
        gr.Info(f"Change language to {language}")

    def _undo_chat(self, history: list[list[str, str]]):
        if len(history) > 0:
            history.pop(-1)
            return history
        return []

    def _reset_chat(self):
        self._pipeline.reset_conversation()
        gr.Info("Reset chat!")
        return "", [], None, "Ready!"

    def _clear_chat(self):
        self._pipeline.clear_conversation()
        gr.Info("Clear chat!")
        return "", [], "Ready!"

    def _show_hide_setting(self, state):
        state = not state
        if state:
            return "Hide Setting", gr.update(visible=state), state
        return "Show Setting", gr.update(visible=state), state

    def build(self):
        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue="slate"),
            js=JS_LIGHT_THEME,
            css=CSS,
        ) as demo:
            gr.Markdown("## Local RAG Chatbot 🤖")
            with gr.Tab("Interface"):
                with gr.Row(variant='panel', equal_height=False):
                    with gr.Column(variant='panel', scale=10) as setting:
                        with gr.Column():
                            status = gr.Textbox(
                                label="Status",
                                value="Ready!",
                                interactive=False
                            )
                            language = gr.Radio(
                                label="Language",
                                choices=["vi", "eng"],
                                value="eng",
                                interactive=True
                            )
                            model = gr.Dropdown(
                                label="Choose Model:",
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
                                height=150,
                                interactive=True
                            )

                    with gr.Column(scale=30, variant="panel"):
                        chatbot = gr.Chatbot(
                            layout='bubble', likeable=True,
                            value=[], height=550, scale=2,

                            show_copy_button=True,
                            bubble_full_width=False,
                            avatar_images=self._avatar_images
                        )
                        with gr.Row(variant='panel'):
                            chat_mode = gr.Dropdown(
                                choices=["chat", "QA"],
                                value="chat",
                                min_width=50,
                                show_label=False,
                                interactive=True,
                                allow_custom_value=False
                            )
                            message = gr.Textbox(placeholder="Enter you message:", show_label=False, scale=6, lines=1)
                            submit_btn = gr.Button(value="Submit", min_width=20, visible=True, elem_classes=["btn"])
                        with gr.Row(variant='panel'):
                            ui_btn = gr.Button(value="Hide Setting", min_width=20)
                            undo_btn = gr.Button(value="Undo", min_width=20)
                            clear_btn = gr.Button(value="Clear", min_width=20)
                            reset_btn = gr.Button(value="Reset", min_width=20)
                            sidebar_state = gr.State(True)

            with gr.Tab("Setting"):
                with gr.Row(variant='panel', equal_height=False):
                    with gr.Column():
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value=self._pipeline.get_system_prompt(),
                            interactive=True,
                            lines=20,
                            max_lines=50
                        )
                        sys_prompt_btn = gr.Button(value="Set System Prompt")

            with gr.Tab("Output"):
                with gr.Row(variant="panel"):
                    log = gr.Code(label="", language="markdown", interactive=False, lines=30)
                    demo.load(
                        self._logger.read_logs, outputs=[log], every=1,
                        show_progress="hidden", scroll_to_output=True
                    )

            clear_btn.click(
                self._clear_chat,
                outputs=[message, chatbot, status]
            )
            cancel_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False), None),
                outputs=[pull_btn, cancel_btn, model]
            )
            undo_btn.click(
                self._undo_chat,
                inputs=[chatbot],
                outputs=[chatbot]
            )
            reset_btn.click(
                self._reset_chat,
                outputs=[message, chatbot, documents, status]
            )
            pull_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False)), outputs=[pull_btn, cancel_btn]
            ).then(self._pull_model, inputs=[model], outputs=[message, chatbot, status, model]
                   ).then(self._change_model, inputs=[model], outputs=[status])
            submit_btn.click(
                self._get_respone,
                inputs=[chat_mode, message, chatbot],
                outputs=[message, chatbot, status]
            )
            message.submit(
                self._get_respone,
                inputs=[chat_mode, message, chatbot],
                outputs=[message, chatbot, status]
            )
            language.change(
                self._change_language,
                inputs=[language]
            )
            model.change(
                self._get_confirm_pull_model,
                inputs=[model],
                outputs=[pull_btn, cancel_btn, status]
            )
            documents.change(
                self._processing_document,
                inputs=[documents],
                outputs=[system_prompt, status]
            )

            sys_prompt_btn.click(
                self._change_system_prompt,
                inputs=[system_prompt]
            )
            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state],
                outputs=[ui_btn, setting, sidebar_state]
            )

        return demo
