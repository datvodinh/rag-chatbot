from typing import List
from dotenv import load_dotenv
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever
)
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import IndexNode, QueryBundle, BaseNode
from llama_index.core.llms.llm import LLM
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings, VectorStoreIndex
from ..prompt import get_query_gen_prompt
from ...setting import RAGSettings

load_dotenv()


class NewQueryFusionRetriever(QueryFusionRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        llm: str | None = None,
        query_gen_prompt: str | None = None,
        mode: FUSION_MODES = FUSION_MODES.SIMPLE,
        similarity_top_k: int = ...,
        num_queries: int = 4,
        use_async: bool = True,
        verbose: bool = False,
        callback_manager: CallbackManager | None = None,
        objects: List[IndexNode] | None = None,
        object_map: dict | None = None,
        retriever_weights: List[float] | None = None
    ) -> None:
        super().__init__(retrievers, llm, query_gen_prompt, mode, similarity_top_k, num_queries,
                         use_async, verbose, callback_manager, objects, object_map, retriever_weights)

    def _get_queries(self, original_query: str) -> List[QueryBundle]:
        prompt_str = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )
        response = self._llm.complete(prompt_str)

        # assume LLM proper put each query on a newline
        queries = response.text.split("\n")
        queries = [q.strip() for q in queries if q.strip()]

        if self._verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")

        # The LLM often returns more queries than we asked for, so trim the list.
        return [QueryBundle(q) for q in queries[: self.num_queries - 1]]


class LocalRetriever:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._host = host

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
        fusion_retriever = NewQueryFusionRetriever(
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
