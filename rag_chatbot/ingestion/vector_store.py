import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    Document,
    SummaryIndex
)
from dotenv import load_dotenv

load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        persist_dir: str = os.path.join(os.getcwd(), "data/chroma"),
    ) -> None:
        pass
        # TODO
        # CHROMA VECTOR STORE
        # if host == "host.docker.internal":
        #     remote_db = chromadb.HttpClient(host=self._host, port=8000)
        # else:
        #     remote_db = chromadb.PersistentClient(persist_dir)
        # chroma_collection = remote_db.get_or_create_collection(name="collection")
        # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        # self.storage_context = StorageContext.from_defaults(
        #     vector_store=vector_store
        # )

    def store_index(self):
        # TODO
        pass

    def store_documents(self):
        # TODO
        pass

    @staticmethod
    def get_index(
        documents: Document,
        mode: str = "summary"
    ):

        if mode == "summary":
            return SummaryIndex.from_documents(
                documents=documents,
                transformations=[
                    SemanticSplitterNodeParser.from_defaults(
                        embed_model=Settings.embed_model,
                        breakpoint_percentile_threshold=95,
                        buffer_size=1
                    )
                ],
                show_progress=True
            )

        else:
            return VectorStoreIndex.from_documents(
                documents=documents,
                transformations=[
                    SemanticSplitterNodeParser.from_defaults(
                        embed_model=Settings.embed_model,
                        breakpoint_percentile_threshold=95,
                        buffer_size=1
                    )
                ],
                show_progress=True
            )
