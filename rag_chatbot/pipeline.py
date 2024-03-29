from rag_chatbot.embedding import LocalEmbedding
from rag_chatbot.model import LocalRAGModel
from rag_chatbot.ingest import DataIngestion
from rag_chatbot.vector_store import LocalVectorStore
from rag_chatbot.engine import (
    LocalChatEngine,
    LocalRetrieveEngine,
    LocalSummaryEngine,
)
from llama_index.core import Settings


class RAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host
        self._chat_engine = LocalChatEngine()
        self._retrieve_engine = LocalRetrieveEngine(host=host)
        self._summary_engine = LocalSummaryEngine()
        Settings.chunk_size = 512
        Settings.chunk_overlap = 64
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)
        self.vector_index = None
        self.summary_index = None

    def set_model(self, model_name: str):
        Settings.llm = LocalRAGModel.set(model_name, host=self._host)

    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name, self._host)

    def pull_model(self, model_name: str):
        return LocalRAGModel.pull(self._host, model_name)

    def pull_embed_model(self, model_name: str):
        return LocalEmbedding.pull(self._host, model_name)

    def check_exist(self, model_name: str):
        return LocalRAGModel.check_model_exist(self._host, model_name)

    def check_exist_embed(self, model_name: str):
        return LocalEmbedding.check_model_exist(self._host, model_name)

    def get_documents(self, input_dir: str = None, input_files: list = None):
        return DataIngestion.get_documents(input_dir, input_files)

    def set_engine(
        self,
        documents=None,
        language: str = "eng",
        mode: str = "chat"
    ):
        if self.vector_index is None and mode != "summary":
            self.vector_index = LocalVectorStore.get_index(documents, mode)
        elif self.summary_index is None and mode == "summary":
            self.summary_index = LocalVectorStore.get_index(documents, mode)
        if mode == "chat":
            self.query_engine = self._chat_engine.from_index(self.vector_index, language)
        elif mode == "summary":
            self.query_engine = self._summary_engine.from_index(self.summary_index, language)
        else:
            self.query_engine = self._retrieve_engine.from_index(self.vector_index, language)

    def query(self, queries: str, mode):
        if mode == "chat":
            response = self.query_engine.stream_chat(queries)
        else:
            response = self.query_engine.query(queries)
        for text in response.response_gen:
            yield text
