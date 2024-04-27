from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from rag_chatbot.prompt import get_system_prompt
from ..setting import RetrieverSettings
from .retriever import LocalRetriever


class LocalChatEngine:
    def __init__(
        self,
        setting: RetrieverSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RetrieverSettings()
        self._retriever = LocalRetriever(self._setting)
        self._host = host

    def from_index(
        self,
        llm,
        vector_index: VectorStoreIndex,
        language: str,
    ) -> CondensePlusContextChatEngine:
        retriever = self._retriever.get_retrievers(
            vector_index=vector_index,
            language=language
        )
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=ChatMemoryBuffer(token_limit=self._setting.chat_token_limit),
            system_prompt=get_system_prompt(language),
            # node_postprocessors=[
            #     SentenceTransformerRerank(
            #         top_n=self._setting.top_k_rerank,
            #         model=self._setting.rerank_llm
            #     )
            # ]
        )

        return chat_engine
