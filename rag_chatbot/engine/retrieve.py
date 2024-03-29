from llama_index.core import (
    Settings,
    VectorStoreIndex,
    get_response_synthesizer,
    Document,
)
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    VectorIndexRetriever
)
from rag_chatbot.prompt import (
    get_qa_and_refine_prompt,
    get_query_gen_prompt
)
from llama_index.core.query_engine import RetrieverQueryEngine
from dotenv import load_dotenv
from .base import LocalBaseEngine

load_dotenv()


class LocalRetrieveEngine(LocalBaseEngine):
    def __init__(self, host: str = "host.docker.internal"):
        super().__init__()
        self._num_queries = 6
        self._similarity_top_k = 5
        self._similarity_cutoff = 0.7
        self._host = host

    def _from_documents(
        self,
        documents: Document,
        language: str,
    ):
        # GET INDEX
        index = VectorStoreIndex.from_documents(
            documents=documents,
            # storage_context=storage_context,
            show_progress=True
        )

        return self._from_index(index, language)

    def _from_index(
        self,
        index: VectorStoreIndex,
        language: str,
    ):
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

        qa_template, refine_template = get_qa_and_refine_prompt(language)

        query_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            response_synthesizer=get_response_synthesizer(
                llm=Settings.llm,
                text_qa_template=qa_template,
                refine_template=refine_template,
                response_mode="compact",
                streaming=True,
                verbose=True
            )
        )

        return query_engine

    def _query(self, queries: str):
        return self.fusion_query_engine.query(queries)
