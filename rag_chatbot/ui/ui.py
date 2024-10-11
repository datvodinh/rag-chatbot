import os
import shutil
import json
import sys
import time
import gradio as gr
from dataclasses import dataclass
from typing import ClassVar
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from .theme import JS_LIGHT_THEME, CSS
from ..pipeline import LocalRAGPipeline
from ..logger import Logger


@dataclass
class DefaultElement:
    DEFAULT_MESSAGE: ClassVar[dict] = {"text": ""}
    DEFAULT_MODEL: str = ""
    DEFAULT_HISTORY: ClassVar[list] = []
    DEFAULT_DOCUMENT: ClassVar[list] = []

    HELLO_MESSAGE: str = "Hi 👋, how can I help you today?"
    SET_MODEL_MESSAGE: str = "You need to choose LLM model 🤖 first!"
    EMPTY_MESSAGE: str = "You need to enter your message!"
    DEFAULT_STATUS: str = "Ready!"
    CONFIRM_PULL_MODEL_STATUS: str = "Confirm Pull Model!"
    PULL_MODEL_SCUCCESS_STATUS: str = "Pulling model 🤖 completed!"
    PULL_MODEL_FAIL_STATUS: str = "Pulling model 🤖 failed!"
    MODEL_NOT_EXIST_STATUS: str = "Model doesn't exist!"
    PROCESS_DOCUMENT_SUCCESS_STATUS: str = "Processing documents 📄 completed!"
    PROCESS_DOCUMENT_EMPTY_STATUS: str = "Empty documents!"
    ANSWERING_STATUS: str = "Answering!"
    COMPLETED_STATUS: str = "Completed!"


class LLMResponse:
    def __init__(self) -> None:
        pass

    def _yield_string(self, message: str):
        for i in range(len(message)):
            time.sleep(0.01)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                [[None, message[: i + 1]]],
                DefaultElement.DEFAULT_STATUS,
            )

    def welcome(self):
        yield from self._yield_string(DefaultElement.HELLO_MESSAGE)

    def set_model(self):
        yield from self._yield_string(DefaultElement.SET_MODEL_MESSAGE)

    def empty_message(self):
        yield from self._yield_string(DefaultElement.EMPTY_MESSAGE)

    def stream_response(
        self,
        message: str,
        history: list[list[str]],
        response: StreamingAgentChatResponse,
    ):
        answer = []
        for text in response.response_gen:
            answer.append(text)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                history + [[message, "".join(answer)]],
                DefaultElement.ANSWERING_STATUS,
            )
        yield (
            DefaultElement.DEFAULT_MESSAGE,
            history + [[message, "".join(answer)]],
            DefaultElement.COMPLETED_STATUS,
        )


class LocalChatbotUI:
    def __init__(
        self,
        pipeline: LocalRAGPipeline,
        logger: Logger,
        host: str = "host.docker.internal",
        data_dir: str = "data/data",
        avatar_images: list[str] = ["./assets/user.png", "./assets/bot.png"],
    ):
        self._pipeline = pipeline
        self._logger = logger
        self._host = host
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self._variant = "panel"
        self._llm_response = LLMResponse()

    def _get_respone(
        self,
        chat_mode: str,
        message: dict[str, str],
        chatbot: list[list[str, str]],
        progress=gr.Progress(track_tqdm=True),
    ):
        if self._pipeline.get_model_name() in [None, ""]:
            for m in self._llm_response.set_model():
                yield m
        elif message["text"] in [None, ""]:
            for m in self._llm_response.empty_message():
                yield m
        else:
            console = sys.stdout
            sys.stdout = self._logger
            response = self._pipeline.query(chat_mode, message["text"], chatbot)
            for m in self._llm_response.stream_response(
                message["text"], chatbot, response
            ):
                yield m
            sys.stdout = console

    def _get_confirm_pull_model(self, model: str):
        if (model in ["gpt-3.5-turbo", "gpt-4"]) or (self._pipeline.check_exist(model)):
            self._change_model(model)
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                DefaultElement.DEFAULT_STATUS,
            )
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            DefaultElement.CONFIRM_PULL_MODEL_STATUS,
        )

    def _pull_model(self, model: str, progress=gr.Progress(track_tqdm=True)):
        if (model not in ["gpt-3.5-turbo", "gpt-4"]) and not (
            self._pipeline.check_exist(model)
        ):
            response = self._pipeline.pull_model(model)
            if response.status_code == 200:
                gr.Info(f"Pulling {model}!")
                for data in response.iter_lines(chunk_size=1):
                    data = json.loads(data)
                    if "completed" in data.keys() and "total" in data.keys():
                        progress(data["completed"] / data["total"], desc="Downloading")
                    else:
                        progress(0.0)
            else:
                gr.Warning(f"Model {model} doesn't exist!")
                return (
                    DefaultElement.DEFAULT_MESSAGE,
                    DefaultElement.DEFAULT_HISTORY,
                    DefaultElement.PULL_MODEL_FAIL_STATUS,
                    DefaultElement.DEFAULT_MODEL,
                )

        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.PULL_MODEL_SCUCCESS_STATUS,
            model,
        )

    def _change_model(self, model: str):
        if model not in [None, ""]:
            self._pipeline.set_model_name(model)
            self._pipeline.set_model()
            self._pipeline.set_engine()
            gr.Info(f"Change model to {model}!")
        return DefaultElement.DEFAULT_STATUS

    def _upload_document(self, document: list[str], list_files: list[str] | dict):
        if document in [None, []]:
            if isinstance(list_files, list):
                return (list_files, DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return list_files.get("files")
                return document
        else:
            if isinstance(list_files, list):
                return (document + list_files, DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return document + list_files.get("files")
                return document

    def _reset_document(self):
        self._pipeline.reset_documents()
        gr.Info("Reset all documents!")
        return (
            DefaultElement.DEFAULT_DOCUMENT,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _show_document_btn(self, document: list[str]):
        visible = False if document in [None, []] else True
        return (gr.update(visible=visible), gr.update(visible=visible))

    def _processing_document(
        self, document: list[str], progress=gr.Progress(track_tqdm=True)
    ):
        document = document or []
        if self._host == "host.docker.internal":
            input_files = []
            for file_path in document:
                dest = os.path.join(self._data_dir, file_path.split("/")[-1])
                shutil.move(src=file_path, dst=dest)
                input_files.append(dest)
            self._pipeline.store_nodes(input_files=input_files)
        else:
            self._pipeline.store_nodes(input_files=document)
        self._pipeline.set_chat_mode()
        gr.Info("Processing Completed!")
        return (self._pipeline.get_system_prompt(), DefaultElement.COMPLETED_STATUS)

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
        return DefaultElement.DEFAULT_HISTORY

    def _reset_chat(self):
        self._pipeline.reset_conversation()
        gr.Info("Reset chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_DOCUMENT,
            DefaultElement.DEFAULT_STATUS,
        )

    def _clear_chat(self):
        self._pipeline.clear_conversation()
        gr.Info("Clear chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
        )

    def _show_hide_setting(self, state):
        state = not state
        label = "Hide Setting" if state else "Show Setting"
        return (label, gr.update(visible=state), state)

    def _welcome(self):
        for m in self._llm_response.welcome():
            yield m

    def build(self):
        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue="slate"),
            js=JS_LIGHT_THEME,
            css=CSS,
        ) as demo:
            gr.Markdown("## Local RAG Chatbot 🤖")
            with gr.Tab("Interface"):
                sidebar_state = gr.State(True)
                with gr.Row(variant=self._variant, equal_height=False):
                    with gr.Column(
                        variant=self._variant, scale=10, visible=sidebar_state.value
                    ) as setting:
                        with gr.Column():
                            status = gr.Textbox(
                                label="Status", value="Ready!", interactive=False
                            )
                            language = gr.Radio(
                                label="Language",
                                choices=["vi", "eng"],
                                value="eng",
                                interactive=True,
                            )
                            model = gr.Dropdown(
                                label="Choose Model:",
                                choices=[
                                    "llama3.1:8b-instruct-q8_0",
                                ],
                                value=None,
                                interactive=True,
                                allow_custom_value=True,
                            )
                            with gr.Row():
                                pull_btn = gr.Button(
                                    value="Pull Model", visible=False, min_width=50
                                )
                                cancel_btn = gr.Button(
                                    value="Cancel", visible=False, min_width=50
                                )

                            documents = gr.Files(
                                label="Add Documents",
                                value=[],
                                file_types=[".txt", ".pdf", ".csv"],
                                file_count="multiple",
                                height=150,
                                interactive=True,
                            )
                            with gr.Row():
                                upload_doc_btn = gr.UploadButton(
                                    label="Upload",
                                    value=[],
                                    file_types=[".txt", ".pdf", ".csv"],
                                    file_count="multiple",
                                    min_width=20,
                                    visible=False,
                                )
                                reset_doc_btn = gr.Button(
                                    "Reset", min_width=20, visible=False
                                )

                    with gr.Column(scale=30, variant=self._variant):
                        chatbot = gr.Chatbot(
                            layout="bubble",
                            value=[],
                            height=550,
                            scale=2,
                            show_copy_button=True,
                            bubble_full_width=False,
                            avatar_images=self._avatar_images,
                        )

                        with gr.Row(variant=self._variant):
                            chat_mode = gr.Dropdown(
                                choices=["chat", "QA"],
                                value="QA",
                                min_width=50,
                                show_label=False,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            message = gr.MultimodalTextbox(
                                value=DefaultElement.DEFAULT_MESSAGE,
                                placeholder="Enter you message:",
                                file_types=[".txt", ".pdf", ".csv"],
                                show_label=False,
                                scale=6,
                                lines=1,
                            )
                        with gr.Row(variant=self._variant):
                            ui_btn = gr.Button(
                                value="Hide Setting"
                                if sidebar_state.value
                                else "Show Setting",
                                min_width=20,
                            )
                            undo_btn = gr.Button(value="Undo", min_width=20)
                            clear_btn = gr.Button(value="Clear", min_width=20)
                            reset_btn = gr.Button(value="Reset", min_width=20)

            with gr.Tab("Setting"):
                with gr.Row(variant=self._variant, equal_height=False):
                    with gr.Column():
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value=self._pipeline.get_system_prompt(),
                            interactive=True,
                            lines=10,
                            max_lines=50,
                        )
                        sys_prompt_btn = gr.Button(value="Set System Prompt")

            with gr.Tab("Output"):
                with gr.Row(variant=self._variant):
                    log = gr.Code(
                        label="", language="markdown", interactive=False, lines=30
                    )
                    demo.load(
                        self._logger.read_logs,
                        outputs=[log],
                        every=1,
                        show_progress="hidden",
                        # scroll_to_output=True,
                    )

            clear_btn.click(self._clear_chat, outputs=[message, chatbot, status])
            cancel_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False), None),
                outputs=[pull_btn, cancel_btn, model],
            )
            undo_btn.click(self._undo_chat, inputs=[chatbot], outputs=[chatbot])
            reset_btn.click(
                self._reset_chat, outputs=[message, chatbot, documents, status]
            )
            pull_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False)),
                outputs=[pull_btn, cancel_btn],
            ).then(
                self._pull_model,
                inputs=[model],
                outputs=[message, chatbot, status, model],
            ).then(self._change_model, inputs=[model], outputs=[status])
            message.submit(
                self._upload_document, inputs=[documents, message], outputs=[documents]
            ).then(
                self._get_respone,
                inputs=[chat_mode, message, chatbot],
                outputs=[message, chatbot, status],
            )
            language.change(self._change_language, inputs=[language])
            model.change(
                self._get_confirm_pull_model,
                inputs=[model],
                outputs=[pull_btn, cancel_btn, status],
            )
            documents.change(
                self._processing_document,
                inputs=[documents],
                outputs=[system_prompt, status],
            ).then(
                self._show_document_btn,
                inputs=[documents],
                outputs=[upload_doc_btn, reset_doc_btn],
            )

            sys_prompt_btn.click(self._change_system_prompt, inputs=[system_prompt])
            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state],
                outputs=[ui_btn, setting, sidebar_state],
            )
            upload_doc_btn.upload(
                self._upload_document,
                inputs=[documents, upload_doc_btn],
                outputs=[documents, upload_doc_btn],
            )
            reset_doc_btn.click(
                self._reset_document, outputs=[documents, upload_doc_btn, reset_doc_btn]
            )
            demo.load(self._welcome, outputs=[message, chatbot, status])

        return demo
