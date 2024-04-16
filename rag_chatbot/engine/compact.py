from llama_index.core.node_parser import SentenceSplitter
from .base import LocalBaseEngine
from dotenv import load_dotenv
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever, RouterRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    get_response_synthesizer,
    Document,
)
from rag_chatbot.prompt import (
    get_qa_and_refine_prompt,
    get_query_gen_prompt,
    get_single_select_prompt
)


load_dotenv()


class LocalCompactEngine(LocalBaseEngine):
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
        vector_index = VectorStoreIndex.from_documents(
            documents=documents,
            transformations=[
                SentenceSplitter(
                    chunk_size=Settings.chunk_size,
                    chunk_overlap=Settings.chunk_overlap
                )
            ],
            show_progress=True
        )

        summary_index = SummaryIndex.from_documents(
            documents=documents,
            transformations=[
                SentenceSplitter(
                    chunk_size=Settings.chunk_size,
                    chunk_overlap=Settings.chunk_overlap
                )
            ],
            show_progress=True
        )

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

        fusion_tool = RetrieverTool.from_defaults(
            retriever=fusion_retriever,
            description=(
                "Useful when the query is ambiguous "
                "and you want to generate more query in order to retrieve context."
            )
        )

        summary_tool = RetrieverTool.from_defaults(
            retriever=summary_index.as_retriever(
                retriever_mode="llm",
                llm=Settings.llm
            ),
            description=(
                "Useful for summarization query "
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
        # FUSION QUERY ENGINE
        query_engine = RetrieverQueryEngine.from_args(
            retriever=router_retriever,
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
