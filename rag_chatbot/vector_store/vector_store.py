import os
import chromadb
from typing import List
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseNode
from llama_index.core import VectorStoreIndex, StorageContext
from dotenv import load_dotenv
from ..setting import StorageSettings

load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        setting: StorageSettings | None = None,
    ) -> None:
        # TODO
        # CHROMA VECTOR STORE
        self._setting = setting or StorageSettings()
        # self._persist_dir_chroma = os.path.join(os.getcwd(), self._setting.persist_dir_chroma)
        # self._persist_dir_storage = os.path.join(os.getcwd(), self._setting.persist_dir_storage)
        # if host == "host.docker.internal":
        #     remote_db = chromadb.HttpClient(
        #         host=self._host,
        #         port=self._setting.port
        #     )
        # else:
        #     remote_db = chromadb.PersistentClient(
        #         path=self._persist_dir_chroma
        #     )
        # chroma_collection = remote_db.get_or_create_collection(
        #     name=self._setting.collection_name
        # )
        
        # self._vector_store = ChromaVectorStore.from_collection(
        #     collection=chroma_collection
        # )
        # self._storage_context = StorageContext.from_defaults(
        #     persist_dir=self._persist_dir_storage
        #     if os.path.exists(self._persist_dir_storage) else None,
        #     vector_store=self._vector_store
        # )

    # def store_nodes(self, nodes: List[BaseNode]):
    #     self._vector_store.add(nodes)
    #     self._storage_context = StorageContext.from_defaults()
    #     self._storage_context.persist(persist_dir=self._persist_dir_storage)

    def get_index(self, nodes):
        index = VectorStoreIndex(nodes=nodes)
        return index
