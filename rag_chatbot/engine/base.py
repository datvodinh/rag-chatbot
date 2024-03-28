from abc import abstractmethod
from llama_index.core import Document, VectorStoreIndex


class LocalBaseEngine:
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _from_documents(self, documents: Document, language: str):
        pass

    @abstractmethod
    def _from_index(self, index: VectorStoreIndex, language: str):
        pass

    def from_documents(self, documents: Document, language: str):
        return self._from_documents(documents, language)

    def from_index(self, index: VectorStoreIndex, language: str):
        return self._from_index(index, language)

    @abstractmethod
    def _query(self, queries: str):
        pass

    def query(self, queries: str):
        return self._query(queries)
