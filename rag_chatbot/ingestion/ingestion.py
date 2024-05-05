import re
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from dotenv import load_dotenv
from typing import List
from ..setting import RAGSettings

load_dotenv()


class LocalDataIngestion:
    def __init__(self, setting: RAGSettings | None = None) -> None:
        self._setting = setting or RAGSettings()

    def get_nodes_from_file(
        self,
        input_dir: str = None,
        input_files: list = None
    ) -> List[BaseNode]:
        documents = SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
            filename_as_id=True
        ).load_data(show_progress=True)

        splitter = SentenceSplitter.from_defaults(
            chunk_size=self._setting.ingestion.chunk_size,
            chunk_overlap=self._setting.ingestion.chunk_overlap,
            paragraph_separator=self._setting.ingestion.paragraph_sep,
            secondary_chunking_regex=self._setting.ingestion.chunking_regex
        )

        for doc in documents:
            doc.text = re.sub(r'\s+', ' ', doc.text.strip())
            doc.excluded_embed_metadata_keys = [
                "file_name", "doc_id", "file_path", "page_label",
                "file_type", "file_size", "creation_date", "last_modified_date"
            ]
            doc.excluded_llm_metadata_keys = [
                "file_name", "doc_id", "file_path", "page_label",
                "file_type", "file_size", "creation_date", "last_modified_date"
            ]

        documents = [doc for doc in documents if doc.text.strip()]

        pipeline = IngestionPipeline(
            transformations=[
                splitter,
                Settings.embed_model
            ]
        )
        return pipeline.run(
            documents=documents,
            show_progress=True,
            num_workers=self._setting.ingestion.num_workers
        )
