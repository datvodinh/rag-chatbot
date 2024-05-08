from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from ...setting import RAGSettings
from .retriever import LocalRetriever


class LocalChatEngine:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._retriever = LocalRetriever(self._setting)
        self._host = host

    def set_engine(
        self,
        llm,
        system_prompt: str,
        vector_index: VectorStoreIndex | None = None,
        language: str = "eng",
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:

        # Normal chat engine
        if vector_index is None:
            return SimpleChatEngine.from_defaults(
                llm=llm,
                system_prompt=system_prompt,
                memory=ChatMemoryBuffer(
                    token_limit=self._setting.ollama.chat_token_limit
                )
            )

        # Chat engine with documents
        retriever = self._retriever.get_retrievers(
            vector_index=vector_index,
            language=language
        )
        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            system_prompt=system_prompt,
            memory=ChatMemoryBuffer(
                token_limit=self._setting.ollama.chat_token_limit
            )
        )
