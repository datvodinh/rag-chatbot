from abc import abstractmethod
from llama_index.core import Document


class LocalBaseEngine:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _set_engine(self, documents: Document, language: str):
        pass

    def set_engine(self, documents: Document, language: str,):
        return self._set_engine(documents, language)

    @abstractmethod
    def _query(self, queries: str):
        pass

    def query(self, queries: str):
        return self._query(queries)
