from typing import List
from dotenv import load_dotenv
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
    RouterRetriever
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, IndexNode
from llama_index.core.llms.llm import LLM
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings, VectorStoreIndex
from ..prompt import get_query_gen_prompt
from ...setting import RAGSettings

load_dotenv()


class TwoStageRetriever(QueryFusionRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        setting: RAGSettings | None = None,
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
        super().__init__(
            retrievers, llm, query_gen_prompt, mode, similarity_top_k, num_queries,
            use_async, verbose, callback_manager, objects, object_map, retriever_weights
        )
        self._setting = setting or RAGSettings()
        self._rerank_model = SentenceTransformerRerank(
            top_n=self._setting.retriever.top_k_rerank,
            model=self._setting.retriever.rerank_llm,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)
        results = self._simple_fusion(results)
        return self._rerank_model.postprocess_nodes(results, query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)
        results = self._simple_fusion(results)
        return self._rerank_model.postprocess_nodes(results, query_bundle)


class LocalRetriever:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._host = host

    def _get_normal_retriever(
        self,
        vector_index: VectorStoreIndex,
        llm: LLM | None = None,
        language: str = "eng",
    ):
        llm = llm or Settings.llm
        return VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self._setting.retriever.similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True
        )

    def _get_hybrid_retriever(
        self,
        vector_index: VectorStoreIndex,
        llm: LLM | None = None,
        language: str = "eng",
        gen_query: bool = True
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
        if gen_query:
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                retriever_weights=self._setting.retriever.retriever_weights,
                llm=llm,
                query_gen_prompt=get_query_gen_prompt(language),
                similarity_top_k=self._setting.retriever.top_k_rerank,
                num_queries=self._setting.retriever.num_queries,
                mode=self._setting.retriever.fusion_mode,
                verbose=True
            )
        else:
            hybrid_retriever = TwoStageRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                retriever_weights=self._setting.retriever.retriever_weights,
                llm=llm,
                query_gen_prompt=None,
                similarity_top_k=self._setting.retriever.similarity_top_k,
                num_queries=1,
                mode=self._setting.retriever.fusion_mode,
                verbose=True
            )

        return hybrid_retriever

    def _get_router_retriever(
        self,
        vector_index: VectorStoreIndex,
        llm: LLM | None = None,
        language: str = "eng",
    ):
        fusion_tool = RetrieverTool.from_defaults(
            retriever=self._get_hybrid_retriever(
                vector_index, llm, language, gen_query=True
            ),
            description="Use this tool when the user's query is ambiguous or unclear.",
            name="Fusion Retriever with BM25 and Vector Retriever and LLM Query Generation."
        )
        two_stage_tool = RetrieverTool.from_defaults(
            retriever=self._get_hybrid_retriever(
                vector_index, llm, language, gen_query=False
            ),
            description="Use this tool when the user's query is clear and unambiguous.",
            name="Two Stage Retriever with BM25 and Vector Retriever and LLM Rerank."
        )

        return RouterRetriever.from_defaults(
            selector=LLMSingleSelector.from_defaults(llm=llm),
            retriever_tools=[fusion_tool, two_stage_tool],
            llm=llm
        )

    def get_retrievers(
        self,
        nodes: List[BaseNode],
        llm: LLM | None = None,
        language: str = "eng",

    ):
        vector_index = VectorStoreIndex(nodes=nodes)
        if len(nodes) > self._setting.retriever.top_k_rerank:
            retriever = self._get_router_retriever(vector_index, llm, language)
        else:
            retriever = self._get_normal_retriever(vector_index, llm, language)

        return retriever
