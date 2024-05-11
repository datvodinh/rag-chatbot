from typing import List
from dotenv import load_dotenv
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    VectorIndexRetriever,
    RouterRetriever
)
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.schema import BaseNode
from llama_index.core.llms.llm import LLM
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings, VectorStoreIndex
from ..prompt import get_query_gen_prompt
from ...setting import RAGSettings

load_dotenv()


class TwoStageRetriever:
    def __init__(self) -> None:
        pass


class LocalRetriever:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._host = host

    def _get_two_stage_retriever(
        self,
        llm: LLM,
        vector_index: VectorStoreIndex,
        language: str,
    ):
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self._setting.retriever.similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True
        )

    def _get_fusion_retriever(
        self,
        llm: LLM,
        vector_index: VectorStoreIndex,
        language: str,
    ):
        # VECTOR INDEX RETRIEVER
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self._setting.retriever.similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True
        )

        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=self._setting.retriever.similarity_top_k,
            verbose=True
        )

        # FUSION RETRIEVER
        fusion_retriever = QueryFusionRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            retriever_weights=self._setting.retriever.retriever_weights,
            llm=llm,
            query_gen_prompt=get_query_gen_prompt(language),
            similarity_top_k=self._setting.retriever.top_k_rerank,
            num_queries=self._setting.retriever.num_queries,
            mode=self._setting.retriever.fusion_mode,
            verbose=True
        )

        return fusion_retriever

    def get_retrievers(
        self,
        llm: LLM,
        language: str,
        nodes: List[BaseNode],
    ):
        vector_index = VectorStoreIndex(nodes=nodes)
        if len(nodes) > self._setting.retriever.top_k_rerank:
            retriever = self._get_fusion_retriever(llm, vector_index, language)
        else:
            retriever = VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=self._setting.retriever.top_k_rerank,
                verbose=True
            )

        return retriever

# TODO: new router retriever
# Ambigous query: vector + bm25 + query fusion
# Good query: vector + bm25 + rerank
