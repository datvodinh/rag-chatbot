from typing import Dict, List
from dotenv import load_dotenv
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,
    TransformRetriever,

)
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from ..prompt import get_query_gen_prompt, get_hyde_prompt
from ..setting import RetrieverSettings

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


class EnsembleRetriever(BaseRetriever):
    def __init__(
            self,
            retrievers: List[BaseRetriever],
            callback_manager: CallbackManager | None = None,
            object_map: Dict | None = None, objects: List[IndexNode] | None = None,
            verbose: bool = False
    ) -> None:
        super().__init__(callback_manager, object_map, objects, verbose)
        self.retrievers = retrievers

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        list_nodes = []
        for retriever in self.retrievers:
            list_nodes.extend(retriever.retrieve(query_bundle))
        return list_nodes


class LocalRetriever:
    def __init__(
        self,
        setting: RetrieverSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RetrieverSettings()
        self._host = host

    def get_retrievers(
        self,
        vector_index: VectorStoreIndex,
        language: str,
    ):
        # VECTOR INDEX RETRIEVER
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self._setting.similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True
        )

        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=self._setting.similarity_top_k,
            verbose=True
        )

        # FUSION RETRIEVER
        fusion_retriever = NewQueryFusionRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            retriever_weights=self._setting.retriever_weights,
            llm=Settings.llm,
            query_gen_prompt=get_query_gen_prompt(language),
            similarity_top_k=self._setting.top_k_rerank,
            num_queries=self._setting.num_queries,
            mode=self._setting.fusion_mode,
            verbose=True
        )

        # HYDE RETRIEVER
        # hyde_retriever = TransformRetriever(
        #     retriever=vector_retriever,
        #     query_transform=HyDEQueryTransform(
        #         llm=Settings.llm,
        #         hyde_prompt=get_hyde_prompt(language),
        #         include_original=False
        #     ),
        #     verbose=True
        # )

        # ensemble_retriever = EnsembleRetriever(
        #     retrievers=[fusion_retriever, hyde_retriever],
        #     verbose=True
        # )

        return fusion_retriever
