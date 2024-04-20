from dotenv import load_dotenv
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    VectorIndexRetriever,
    RouterRetriever,
    SummaryIndexRetriever,
)
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    get_response_synthesizer,
    Document,
)
from .base import LocalBaseEngine
from ..prompt import (
    get_qa_and_refine_prompt,
    get_query_gen_prompt,
    get_single_select_prompt
)
from ..setting import GlobalSettings
from ..ingestion import LocalVectorStore

load_dotenv()


class LocalCompactEngine(LocalBaseEngine):
    def __init__(self, host: str = "host.docker.internal"):
        super().__init__()
        settings = GlobalSettings()
        self._num_queries = settings.num_queries
        self._similarity_top_k = settings.similarity_top_k
        self._top_k_rerank = settings.top_k_rerank
        self._similarity_cutoff = settings.similarity_cutoff
        self._host = host

    def _from_documents(
        self,
        documents: Document,
        language: str,
    ):
        # GET INDEX
        vector_index = LocalVectorStore.get_index(documents=documents, mode="vector")
        summary_index = LocalVectorStore.get_index(documents=documents, mode="summary")
        return self._from_index(vector_index, summary_index, language)

    def _from_index(
        self,
        vector_index: VectorStoreIndex,
        summary_index: SummaryIndex,
        language: str,
    ) -> RetrieverQueryEngine:
        # VECTOR INDEX RETRIEVER
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self._similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True
        )

        # FUSION RETRIEVER
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever],
            llm=Settings.llm,
            query_gen_prompt=get_query_gen_prompt(language),
            similarity_top_k=self._similarity_top_k,
            num_queries=self._num_queries,
            mode="reciprocal_rerank",
            verbose=True
        )

        summary_retriever = SummaryIndexRetriever(
            index=summary_index,
            verbose=True
        )

        fusion_tool = RetrieverTool.from_defaults(
            retriever=fusion_retriever,
            description=(
                "Optimal for resolving ambiguous queries by "
                "generating additional queries to refine search results."
            )
        )

        summary_tool = RetrieverTool.from_defaults(
            retriever=summary_retriever,
            description=(
                "Efficient tool for summarizing queries, "
                "allowing for concise extraction of key information."
            )
        )

        qa_template, refine_template = get_qa_and_refine_prompt(language)

        router_retriever = RouterRetriever(
            selector=LLMSingleSelector.from_defaults(
                llm=Settings.llm,
                prompt_template_str=get_single_select_prompt(language)
            ),
            retriever_tools=[
                summary_tool,
                fusion_tool
            ],
            verbose=True

        )

        qa_template, refine_template = get_qa_and_refine_prompt(language)

        query_engine = RetrieverQueryEngine.from_args(
            retriever=router_retriever,
            response_synthesizer=get_response_synthesizer(
                llm=Settings.llm,
                text_qa_template=qa_template,
                refine_template=refine_template,
                response_mode="tree_summarize",
                streaming=True,
                verbose=True
            ),
        )

        return query_engine
