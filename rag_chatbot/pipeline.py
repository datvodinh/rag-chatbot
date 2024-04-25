from .embedding import LocalEmbedding
from .model import LocalRAGModel
from .ingestion import LocalDataIngestion
from .vector_store import LocalVectorStore
from .engine import LocalChatEngine, LocalCompactEngine
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage


class LocalRAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host
        self._engine = {
            "chat": LocalChatEngine(host=host),
            "compact": LocalCompactEngine(host=host)
        }
        self._default_model = LocalRAGModel.set(host=host)
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        self._vector_store = LocalVectorStore(host=host)
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)

    def set_model(self, model_name: str):
        Settings.llm = LocalRAGModel.set(model_name, host=self._host)
        self._default_model = LocalRAGModel.set(model_name, host=self._host)

    def reset_engine(self):
        self._query_engine = None

    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name, self._host)

    def pull_model(self, model_name: str):
        return LocalRAGModel.pull(self._host, model_name)

    def pull_embed_model(self, model_name: str):
        return LocalEmbedding.pull(self._host, model_name)

    def check_exist(self, model_name: str) -> bool:
        return LocalRAGModel.check_model_exist(self._host, model_name)

    def check_exist_embed(self, model_name: str) -> bool:
        return LocalEmbedding.check_model_exist(self._host, model_name)

    def get_nodes_from_file(
            self,
            input_dir: str = None,
            input_files: list[str] = None
    ) -> None:
        nodes = self._ingestion.get_nodes_from_file(
            input_dir=input_dir,
            input_files=input_files
        )
        return nodes

    def set_engine(
        self,
        nodes,
        language: str = "eng",
        mode: str = "chat",
    ):
        index = self._vector_store.get_index(nodes)
        self._query_engine = self._engine[mode].from_index(
            index, language=language
        )

    def query(self, queries: str, mode):
        if self._query_engine is not None:
            if mode == "chat":
                response = self._query_engine.stream_chat(queries)
            else:
                response = self._query_engine.query(queries)
            for text in response.response_gen:
                yield text
        else:
            response = self._default_model.stream_chat(
                [
                    ChatMessage(role="user", content=queries)
                ]
            )
            for r in response:
                yield r.delta
