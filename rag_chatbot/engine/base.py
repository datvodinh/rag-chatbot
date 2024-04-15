from abc import abstractmethod
from llama_index.core import Document, VectorStoreIndex


class LocalBaseEngine:
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _from_documents(self, documents: Document, language: str):
        raise NotImplementedError

    @abstractmethod
    def _from_index(self, index: VectorStoreIndex, language: str):
        raise NotImplementedError

    def from_documents(self, documents: Document, language: str):
        return self._from_documents(documents, language)

    def from_index(self, index: VectorStoreIndex, language: str):
        return self._from_index(index, language)
