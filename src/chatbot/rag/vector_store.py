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
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from chatbot.prompt import (
    get_prompt_format,
    get_query_gen_format,
    get_rerank_prompt
)
from dotenv import load_dotenv

load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        num_queries: int = 6,
        similarity_top_k: int = 10,
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

    def get_documents(self, input_dir: str = None, input_files: list = None):
        return SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
            filename_as_id=True
        ).load_data(show_progress=True)

    def get_index(self, documents: Document):
        if self._host == "host.docker.internal":
            remote_db = chromadb.HttpClient(host=self._host, port=8000)
        else:
            remote_db = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "data/chroma"))
        chroma_collection = remote_db.get_or_create_collection(name="collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self._storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        return VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=self._storage_context,
            transformations=[
                SentenceSplitter(
                    chunk_size=Settings.chunk_size,
                    chunk_overlap=Settings.chunk_overlap
                )
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
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever],
            llm=Settings.llm,
            similarity_top_k=self._similarity_top_k,
            num_queries=self._num_queries,
            mode="reciprocal_rerank",
            verbose=True
        )

        index.as_retriever()

        query_engine = RetrieverQueryEngine.from_args(
            llm=Settings.llm,
            retriever=fusion_retriever,
            response_synthesizer=get_response_synthesizer(
                llm=Settings.llm,
                response_mode="compact",
                streaming=True,
                verbose=True
            ),
            # node_postprocessors=[
            #     SentenceTransformerRerank(
            #         top_n=5,
            #         model="sentence-transformers/all-MiniLM-L6-v2"
            #     )
            # ]
        )
        return query_engine
