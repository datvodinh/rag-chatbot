from rag_chatbot.embedding import LocalEmbedding
from rag_chatbot.model import LocalRAGModel
from rag_chatbot.ingest import DataIngestion
from rag_chatbot.engine import (
    LocalChatEngine,
    LocalRetrieveEngine,
    LocalSummaryEngine
)
from llama_index.core import Settings


class RAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self.host = host
        self.chat_engine = LocalChatEngine()
        self.retrieve_engine = LocalRetrieveEngine(host=host)
        self.summary_engine = LocalSummaryEngine()
        Settings.chunk_size = 512
        Settings.chunk_overlap = 64
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)

    def set_model(self, model_name: str):
        Settings.llm = LocalRAGModel.set(model_name, host=self.host)

    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name, self.host)

    def pull_model(self, model_name: str):
        return LocalRAGModel.pull(self.host, model_name)

    def pull_embed_model(self, model_name: str):
        return LocalEmbedding.pull(self.host, model_name)

    def check_exist(self, model_name: str):
        return LocalRAGModel.check_model_exist(self.host, model_name)

    def check_exist_embed(self, model_name: str):
        return LocalEmbedding.check_model_exist(self.host, model_name)

    def get_documents(self, input_dir: str = None, input_files: list = None):
        return DataIngestion.get_documents(input_dir, input_files)

    def set_engine(self, documents, language, mode):
        if mode == "chat":
            self.query_engine = self.chat_engine.from_documents(documents, language)
        elif mode == "summary":
            pass
        else:
            self.query_engine = self.retrieve_engine.from_documents(documents, language)

    def query(self, queries: str):
        response = self.query_engine.stream_chat(queries)
        for text in response.response_gen:
            yield text
