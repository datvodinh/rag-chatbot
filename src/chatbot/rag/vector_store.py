import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from chatbot.prompt import get_query_gen_format


class LocalVectorStore:
    def __init__(
        self,
        llm,
        host: str = "host.docker.internal",
        num_queries: int = 5,
        similarity_top_k: int = 5,
        similarity_cutoff: float = 0.7
    ) -> None:
        self._llm = llm
        self._num_queries = num_queries
        self._similarity_top_k = similarity_top_k
        self._similarity_cutoff = similarity_cutoff
        remote_db = chromadb.HttpClient(host=host, port=8000)
        chroma_collection = remote_db.get_or_create_collection("collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.storage_context.persist(os.path.join(os.getcwd(), "data/storage"))

    def set_llm(self, llm):
        self._llm = llm

    def get_documents(self, input_dir: str):
        documents = SimpleDirectoryReader(
            input_dir=input_dir,
            filename_as_id=True
        ).load_data(show_progress=True)
        return documents

    def get_nodes(self, documents: Document):
        splitter = SentenceSplitter(chunk_size=512)
        nodes = splitter.get_nodes_from_documents(documents)
        return nodes

    def get_index(self, documents=None, nodes=None):
        assert (documents is not None) or (nodes is not None), "atleast documents or nodes is not None"
        if documents is not None:
            return VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                transformations=[
                    SentenceSplitter(chunk_size=512)
                ]
            )
        elif nodes is not None:
            return VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                transformations=[
                    SentenceSplitter(chunk_size=512)
                ]
            )

    def get_query_engine(
        self,
        index: VectorStoreIndex,
        nodes,
        language: str = "eng"
    ):
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self._similarity_top_k
        )
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self._similarity_top_k
        )
        fusion_retriever = QueryFusionRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            query_gen_prompt=get_query_gen_format(language),
            similarity_top_k=self._similarity_top_k,
            num_queries=self._num_queries,
            mode="reciprocal_rerank",
            verbose=True
        )
        query_engine = RetrieverQueryEngine(
            retriever=fusion_retriever,
            response_synthesizer=get_response_synthesizer(
                llm=self._llm,
                streaming=True,
                verbose=True
            ),
            node_postprocessors=[
                SimilarityPostprocessor(
                    similarity_cutoff=self._similarity_cutoff
                )
            ]
        )
        return query_engine
