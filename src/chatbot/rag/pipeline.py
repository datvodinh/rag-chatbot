from .embedding import LocalEmbedding
from .model import LocalRAGModel
from .vector_store import LocalVectorStore

from llama_index.core import Settings


class RAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self.host = host
        Settings.chunk_size = 512
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set()
        self.vector_store = LocalVectorStore(llm=Settings.llm, host=host)

    def set_model(self, model_name: str):
        Settings.llm = LocalRAGModel.set(model_name, host=self.host)
        self.vector_store.set_llm(Settings.llm)

    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name)

    def pull_model(self, model_name: str):
        return LocalRAGModel.pull(self.host, model_name)

    def check_exist(self, model_name: str):
        return LocalRAGModel.check_model_exist(self.host, model_name)

    def get_documents(self, input_dir: str):
        return self.vector_store.get_documents(input_dir)

    def get_nodes(self, documents):
        return self.vector_store.get_nodes(documents)

    def get_index(self, documents):
        return self.vector_store.get_index(documents)

    def get_query_engine(self, index, language):
        return self.vector_store.get_query_engine(index, language)

    def update_query_engine(
        self,
        input_dir: str,
        language: str
    ):
        documents = self.get_documents(input_dir)
        index = self.get_index(documents)
        self.query_engine = self.get_query_engine(index, language)

    def query(self, query: str):
        return self.query_engine.query(query)
