from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    LocalRAGModel,
    LocalEmbedding,
    LocalVectorStore,
    get_system_prompt
)
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse


class LocalRAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host
        self._language = "eng"
        self._mode = "chat"
        self._model_name = ""
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)
        self._engine = LocalChatEngine(host=host)
        self._default_model = LocalRAGModel.set(self._model_name, host=host)
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        self._vector_store = LocalVectorStore(host=host)
        self._nodes = []
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)

    def get_model_name(self):
        return self._model_name

    def set_model_name(self, model_name: str):
        self._model_name = model_name

    def get_language(self):
        return self._language

    def set_language(self, language: str):
        self._language = language

    def get_mode(self):
        return self._mode

    def set_mode(self, mode: str):
        self._mode = mode

    def get_system_prompt(self):
        return self._system_prompt

    def set_system_prompt(
        self,
        system_prompt: str | None = None
    ):
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language,
            is_rag_prompt=self.check_nodes_exist()
        )

    def set_model(self):
        Settings.llm = LocalRAGModel.set(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            host=self._host
        )
        self._default_model = Settings.llm

    def reset_engine(self):
        self._query_engine = self._engine.set_engine(self._default_model, self._system_prompt)

    def check_nodes_exist(self):
        return len(self._nodes) > 0

    def reset_nodes(self):
        self._nodes = []

    def reset_conversation(self):
        self.reset_engine()
        self.reset_nodes()
        self.set_system_prompt(
            get_system_prompt(language=self._language, is_rag_prompt=False)
        )

    def store_nodes(self, nodes):
        self._nodes.extend(nodes)

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

    def set_chat_mode(
        self,
        system_prompt: str | None = None
    ):
        self.set_language(self._language)
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine()

    def set_engine(self):
        self.set_mode(self._mode)
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=self._nodes,
            language=self._language
        )

    def query(self, message: str) -> StreamingAgentChatResponse:
        return self._query_engine.stream_chat(message)
