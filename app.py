import os
import shutil
import json
import argparse
import gradio as gr
from rag_chatbot import RAGPipeline
from ollama.ollama import run_ollama_server


def main(host="host.docker.internal", share=False):

    INPUT_DIR = os.path.join(os.getcwd(), "data/data")
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    rag_pipeline = RAGPipeline(host=host)

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
        gr.Markdown("# LLM Chatbot + RAG")
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(variant='panel'):
                gr.Markdown("### Step 1: Set Model")
                model = gr.Dropdown(
                    label="Model",
                    choices=[

                        "mistral:7b-instruct-v0.2-q6_K",
                        "mistral:7b-instruct-v0.2-q4_0",
                        "zephyr:7b-beta-q6_K",
                        "gpt-3.5-turbo"
                    ],
                    value="mistral:7b-instruct-v0.2-q6_K",
                    interactive=True,
                    allow_custom_value=True
                )
                embed_model = gr.Dropdown(
                    label="Embedding Model",
                    choices=[
                        "BAAI/bge-base-en-v1.5",
                        "intfloat/multilingual-e5-large-instruct"
                    ],
                    value="BAAI/bge-base-en-v1.5",
                    interactive=True,
                )
                language = gr.Dropdown(
                    label="Language",
                    choices=["vi", "eng"],
                    value="eng",
                    interactive=True
                )
                pull_btn = gr.Button("Pull Model")
                gr.Markdown("### Step 2: Add Documents")
                documents = gr.Files(
                    label="Documents",
                    file_types=[".txt", ".pdf", ".csv"],
                    file_count="multiple",
                    height=200
                )
                doc_progress = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                )
                doc_btn = gr.Button("Get Index")

            with gr.Column(scale=3, variant="panel"):
                chatbot = gr.Chatbot(layout='bubble', value=[], scale=3)
                with gr.Row(variant='panel'):
                    message = gr.Textbox(label="Enter Prompt:", scale=5, lines=1)
                    send_btn = gr.Button(value="Send", scale=1, size='sm')
                with gr.Row(variant='panel'):
                    clear_btn = gr.Button(value="Clear")
                    undo_btn = gr.Button(value="Undo")

        @send_btn.click(inputs=[message, chatbot], outputs=[message, chatbot])
        @message.submit(inputs=[message, chatbot], outputs=[message, chatbot])
        def get_respone(message, chatbot, progress=gr.Progress(track_tqdm=True)):
            gr.Info("Generating Answer!")
            user_mess = message
            all_text = []
            for text in rag_pipeline.query(user_mess):
                all_text.append(text)
                yield "", chatbot + [[user_mess, "".join(all_text)]]
            gr.Info("Generating Completed!")

        @clear_btn.click(outputs=[message, chatbot])
        @model.change(outputs=[message, chatbot])
        def clear_chat():
            return "", []

        @undo_btn.click(inputs=[message, chatbot], outputs=[message, chatbot])
        def undo_chat(message, history):
            if len(history) > 0:
                history.pop(-1)
                return "", history
            return "", []

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
                            progress(data['completed'] / data['total'])
                        else:
                            progress(0.)
                else:
                    gr.Warning(f"Model {model} doesn't exist!")
                rag_pipeline.set_model(model)
                gr.Info(f"Model {model} is ready!")
            else:
                rag_pipeline.set_model(model)
                gr.Info(f"Model {model} is ready!")
            return "", []

        @pull_btn.click(inputs=[embed_model], outputs=[message, chatbot])
        def set_embed_model(embed_model, progress=gr.Progress(track_tqdm=True)):
            gr.Info(f"Pulling {embed_model}!")
            rag_pipeline.set_embed_model(embed_model)
            gr.Info(f"Embedding model {embed_model} is ready!")
            return "", []

        @doc_btn.click(inputs=[documents, language], outputs=[doc_progress])
        def processing_document(document, language, progress=gr.Progress(track_tqdm=True)):
            gr.Info("Processing Document!")
            if host == "host.docker.internal":
                for file_path in documents:
                    shutil.move(src=file_path, dst=os.path.join(INPUT_DIR, file_path.split("/")[-1]))
                documents = rag_pipeline.get_documents(input_dir=INPUT_DIR)
            else:
                documents = rag_pipeline.get_documents(input_files=document)
            gr.Info("Indexing!")
            rag_pipeline.query_engine = rag_pipeline.get_query_engine(documents, language)
            gr.Info("Processing Completed!")
            return "Completed!"

        @documents.change(inputs=[documents])
        def change(documents):
            print(documents)

        @language.change(inputs=[language])
        def change_language(language):
            gr.Info(f"Change language to {language}")

    demo.launch(share=share, server_name="0.0.0.0")


if __name__ == "__main__":
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

    if args.host != "host.docker.internal":
        run_ollama_server()

    main(host=args.host, share=args.share)
