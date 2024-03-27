import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Document
from dotenv import load_dotenv

load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        persist_dir: str = os.path.join(os.getcwd(), "data/chroma"),
    ) -> None:
        # CHROMA VECTOR STORE
        if host == "host.docker.internal":
            remote_db = chromadb.HttpClient(host=self._host, port=8000)
        else:
            remote_db = chromadb.PersistentClient(persist_dir)
        chroma_collection = remote_db.get_or_create_collection(name="collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

    def store_index(self):
        # TODO
        pass

    def store_documents(self):
        # TODO
        pass

    def _set_engine(
        self,
        documents: Document,
        language: str,
    ):

        nodes = self.get_nodes(documents)
        self.storage_context.docstore.add_documents(nodes)
        # GET INDEX
        vector_index = VectorStoreIndex.from_documents(
            documents=documents,
            # storage_context=storage_context,
            show_progress=True
        )
