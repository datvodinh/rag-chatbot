import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document,
    get_response_synthesizer,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    VectorIndexRetriever
)
from chatbot.prompt import get_query_gen_format
from dotenv import load_dotenv

load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        num_queries: int = 5,
        similarity_top_k: int = 5,
        similarity_cutoff: float = 0.7
    ) -> None:
        self._num_queries = num_queries
        self._similarity_top_k = similarity_top_k
        self._similarity_cutoff = similarity_cutoff
        self._host = host

    def set_llm(self, llm):
        Settings.llm = llm

    def set_embed_model(self, embed_model):
        Settings.embed_model = embed_model

    def get_documents(self, input_dir: str):
        documents = SimpleDirectoryReader(
            input_dir=input_dir,
            filename_as_id=True
        ).load_data(show_progress=True)
        return documents

    def get_index(self, documents: Document):
        if self._host == "host.docker.internal":
            remote_db = chromadb.HttpClient(host=self._host, port=8000)
        else:
            remote_db = chromadb.EphemeralClient()
        chroma_collection = remote_db.get_or_create_collection(name="collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            transformations=[
                SentenceSplitter(chunk_size=512)
            ]
        )

    def get_query_engine(
        self,
        index: VectorStoreIndex,
        language: str = "eng"
    ):
        vector_retriever = index.as_retriever(
            similarity_top_k=self._similarity_top_k
        )
        # bm25_retriever = BM25Retriever.from_defaults(
        #     nodes=nodes,
        #     similarity_top_k=self._similarity_top_k
        # )
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever],
            llm=Settings.llm,
            similarity_top_k=self._similarity_top_k,
            num_queries=self._num_queries,
            mode="simple",
            verbose=True
        )

        index.as_retriever()

        query_engine = RetrieverQueryEngine.from_args(
            llm=Settings.llm,
            retriever=fusion_retriever,
            response_synthesizer=get_response_synthesizer(
                llm=Settings.llm,
                response_mode="tree_summarize",
                streaming=True,
                verbose=True
            ),
            # node_postprocessors=[
            #     SimilarityPostprocessor(
            #         similarity_cutoff=self._similarity_cutoff
            #     )
            # ]
        )
        return query_engine
