import os
import json
import gradio as gr
from chatbot import LocalLLMModel, RAGPipeline


with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    llm_model = LocalLLMModel()
    rag_pipeline = RAGPipeline()
    gr.Markdown("# LLM Chatbot + RAG")
    with gr.Row(variant='panel', equal_height=True):
        with gr.Column(variant='panel'):
            documents = gr.Files(label="Documents")
            doc_btn = gr.Button("Process Docs")
            model = gr.Dropdown(label="Model",
                                choices=["gemma:2b-instruct", "llama2:chat", "mistral:instruct",
                                         "mistral:7b-instruct-v0.2-q4_0"],
                                value="llama2:chat", interactive=True, allow_custom_value=True
                                )
            embed_model = gr.Dropdown(label="Embedding Model",
                                      choices=["intfloat/e5-large-v2"],
                                      value="intfloat/e5-large-v2", interactive=True, allow_custom_value=True
                                      )
            language = gr.Dropdown(label="Language", choices=["vi", "eng"], value="vi", interactive=True)
            pull_btn = gr.Button("Pull Model")

        with gr.Column(scale=3, variant="panel"):
            chatbot = gr.Chatbot(layout='bubble', value=[], scale=3)
            with gr.Row(variant='panel'):
                message = gr.Textbox(label="Enter Prompt:", scale=5, lines=1)
                send_btn = gr.Button(value="Send", scale=1, size='sm')
            with gr.Row(variant='panel'):
                clear_btn = gr.Button(value="Clear")
                undo_btn = gr.Button(value="Undo")

    with gr.Accordion("Hyperparameters", open=False):
        with gr.Row():
            with gr.Column(variant='panel'):
                gr.Markdown(value="## Model")
                temp = gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.8, step=0.05)
                top_k = gr.Slider(label="Top_k", minimum=5, maximum=100, value=20, step=5)
                top_p = gr.Slider(label="Top_p", minimum=0.8, maximum=1, value=0.9, step=0.05)
                freq_penalty = gr.Slider(label="Frequency Penalty", value=1.05, minimum=-2, maximum=2, step=0.05)
            with gr.Column(variant='panel'):
                gr.Markdown("## RAG")
                chunk_size = gr.Radio(choices=[256, 512, 1024, 2048],)
    # EVENT
    get_response_kwargs = {
        "inputs": [model, message, chatbot, temp, top_k, top_p, freq_penalty],
        "outputs": [message, chatbot]
    }

    @send_btn.click(**get_response_kwargs)
    @message.submit(**get_response_kwargs)
    def get_respone(model, message, chatbot, temp, top_k, top_p, freq_penalty):
        print(chatbot)
        if rag_pipeline.has_docs:
            gr.Info("Querying Database!")
            message = rag_pipeline.query(message)
            gr.Info("Querying Completed!")
            print(message)
        gr.Info("Generating Answer!")
        for mess, his in llm_model.get_response(
            model, message, chatbot, temp, top_k, top_p, freq_penalty
        ):
            yield mess, his
        gr.Info("Generating Completed!")

    @clear_btn.click(inputs=[message, chatbot], outputs=[message, chatbot])
    @model.change(inputs=[message, chatbot], outputs=[message, chatbot])
    def clear_chat(message, history):
        return "", []

    @undo_btn.click(inputs=[message, chatbot], outputs=[message, chatbot])
    def undo_chat(message, history):
        if len(history) > 0:
            history.pop(-1)
            return "", history
        return "", []

    @pull_btn.click(inputs=[model], outputs=[message, chatbot])
    def check_and_download_model(model, progress=gr.Progress(track_tqdm=True)):

        if not llm_model.check_model_exist(model):
            response = llm_model.pull_model(model)
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

        else:
            gr.Info(f"Model {model} is ready!")

        gr.Info("Pulling Embedding Model!")
        rag_pipeline.pull_embed_model()
        gr.Info("Pulling Completed!")
        return "", []

    @doc_btn.click(inputs=[documents])
    def processing_document(documents):
        gr.Info("Processing Document!")
        for d in documents:
            with open(d, "r") as f:
                data = f.read()
            if not os.path.exists(os.path.join(os.getcwd(),"data/data/")):
                os.makedirs(os.path.join(os.getcwd(),"data/data/"))
            data_dir = os.path.join(os.getcwd(),f"data/data/{d.replace("/","")}")
            with open(data_dir, "w") as f:
                f.write(data)
        rag_pipeline.add_document()
        gr.Info("Processing Completed!")

    @documents.change(inputs=[documents])
    def change(documents):
        print(documents)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")
