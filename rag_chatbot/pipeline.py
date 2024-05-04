from .embedding import LocalEmbedding
from .model import LocalRAGModel
from .ingestion import LocalDataIngestion
from .vector_store import LocalVectorStore
from .engine import LocalChatEngine, LocalCompactEngine
from .prompt import get_system_prompt
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
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        self._vector_store = LocalVectorStore(host=host)
        self._nodes = []
        self._language = "eng"
        self._mode = "chat"
        Settings.llm = LocalRAGModel.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)

    def set_model(self, model_name: str):
        Settings.llm = LocalRAGModel.set(
            model_name=model_name,
            system_prompt=self._system_prompt,
            host=self._host
        )
        self._default_model = Settings.llm

    def set_language(self, language: str):
        self._language = language

    def set_mode(self, mode: str):
        self._mode = mode

    def get_system_prompt(self):
        return self._system_prompt

    def set_system_prompt(
        self,
        system_prompt: str | None = None
    ):
        self._system_prompt = system_prompt

    def set_system_prompt_by_lang(
        self,
        language: str,
        is_rag_prompt: bool
    ):
        self._system_prompt = get_system_prompt(language, is_rag_prompt)

    def reset_engine(self):
        self._query_engine = None

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

    def set_engine(
        self,
        mode: str = "chat",
    ):
        self.set_mode(mode)
        self.set_system_prompt(get_system_prompt(language=self._language, is_rag_prompt=True))
        index = self._vector_store.get_index(self._nodes)
        self._query_engine = self._engine[mode].from_index(
            llm=self._default_model, vector_index=index, language=self._language
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
                    ChatMessage(role="system", content=self._system_prompt),
                    ChatMessage(role="user", content=queries)
                ]
            )
            for r in response:
                yield r.delta
