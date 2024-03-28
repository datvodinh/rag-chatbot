from llama_index.core import Document, VectorStoreIndex, Settings
from .base import LocalBaseEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from rag_chatbot.prompt import get_query_gen_prompt, get_system_prompt


class LocalChatEngine(LocalBaseEngine):
    def __init__(self, host: str = "host.docker.internal"):
        super().__init__()
        self._num_queries = 6
        self._similarity_top_k = 5
        self._similarity_cutoff = 0.7
        self._host = host

    def _from_documents(self, documents: Document, language: str):
        # GET INDEX
        index = VectorStoreIndex.from_documents(
            documents=documents,
            # storage_context=storage_context,
            show_progress=True
        )

        return self._from_index(index, language)

    def _from_index(self, index: VectorStoreIndex, language: str):
        # VECTOR INDEX RETRIEVER
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self._similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True
        )

        # FUSION RETRIEVER
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever],
            llm=Settings.llm,
            query_gen_prompt=get_query_gen_prompt(language),
            similarity_top_k=5,
            num_queries=self._num_queries,
            mode="reciprocal_rerank",
            verbose=True
        )
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=fusion_retriever,
            llm=Settings.llm,
            memory=ChatMemoryBuffer(token_limit=6000),
            system_prompt=get_system_prompt(language)
        )

        return chat_engine
