from .base import LocalBaseEngine
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    get_response_synthesizer,
    DocumentSummaryIndex,

)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.document_summary import DocumentSummaryIndexLLMRetriever
from rag_chatbot.prompt import get_qa_and_refine_prompt


class LocalSummaryEngine(LocalBaseEngine):
    def __init__(self, host: str = "host.docker.internal"):
        super().__init__()
        self._num_queries = 6
        self._similarity_top_k = 5
        self._similarity_cutoff = 0.7
        self._host = host

    def _from_documents(self, documents: Document, language: str):
        doc_summary_index = DocumentSummaryIndex.from_documents(
            documents=documents,
            llm=Settings,
            transformations=[
                SentenceSplitter(
                    chunk_size=Settings.chunk_size,
                    chunk_overlap=Settings.chunk_overlap
                )
            ]
        )
        return self._from_index(doc_summary_index, language)

    def _from_index(self, index: VectorStoreIndex, language: str):
        summary_retriever = DocumentSummaryIndexLLMRetriever(
            index=index,
            choice_batch_size=10,
            choice_top_k=1,
            llm=Settings.llm,
        )

        qa_template, refine_template = get_qa_and_refine_prompt(language)

        query_engine = RetrieverQueryEngine.from_args(
            retriever=summary_retriever,
            response_synthesizer=get_response_synthesizer(
                llm=Settings.llm,
                text_qa_template=qa_template,
                refine_template=refine_template,
                use_async=True,
                response_mode="compact",
                streaming=True,
                verbose=True
            )
        )

        return query_engine
