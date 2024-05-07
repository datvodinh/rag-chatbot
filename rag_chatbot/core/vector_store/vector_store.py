from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from ..setting import RAGSettings

load_dotenv()


class LocalVectorStore:
    def __init__(
        self,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ) -> None:
        # TODO
        # CHROMA VECTOR STORE
        self._setting = setting or RAGSettings()

    def get_index(self, nodes):
        index = VectorStoreIndex(nodes=nodes)
        return index
