import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    get_response_synthesizer,
    SimpleDirectoryReader,
    StorageContext,
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
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv


load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        persist_dir: str = os.path.join(os.getcwd(), "data/chroma"),
        num_queries: int = 6,
        similarity_top_k: int = 5,
        similarity_cutoff: float = 0.7
    ) -> None:
        self._num_queries = num_queries
        self._similarity_top_k = similarity_top_k
        self._similarity_cutoff = similarity_cutoff
        self._host = host
        self._persist_dir = persist_dir

    def set_llm(self, llm):
        Settings.llm = llm

    def set_embed_model(self, embed_model):
        Settings.embed_model = embed_model

    def _exclude_metadata(self, documents: list[Document]):
        for doc in documents:
            doc.excluded_embed_metadata_keys = ["doc_id"]
            doc.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
        return documents

    def get_documents(self, input_dir: str = None, input_files: list = None):
        documents = SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
            filename_as_id=True
        ).load_data(show_progress=True)
        documents = self._exclude_metadata(documents)
        return documents

    def get_query_engine(
        self,
        documents: Document,
        language: str,
    ):
        # CHROMA VECTOR STORE
        if self._host == "host.docker.internal":
            remote_db = chromadb.HttpClient(host=self._host, port=8000)
        else:
            remote_db = chromadb.EphemeralClient()
        chroma_collection = remote_db.get_or_create_collection(name="collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        splitter = SentenceSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap
        )
        # GET NODES FROM DOCUMENTS
        nodes = splitter.get_nodes_from_documents(documents)
        storage_context.docstore.add_documents(nodes)
        # GET INDEX
        vector_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True
        )

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
            similarity_top_k=5,
            num_queries=self._num_queries,
            mode="reciprocal_rerank",
            verbose=True
        )

        qa_template, refine_template = get_qa_and_refine_prompt(language)

        fusion_query_engine = RetrieverQueryEngine.from_args(
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

        return fusion_query_engine
