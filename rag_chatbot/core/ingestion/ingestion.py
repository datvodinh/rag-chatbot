import re
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from typing import Any, List
from tqdm import tqdm
from ...setting import RAGSettings

load_dotenv()


class LocalDataIngestion:
    def __init__(self, setting: RAGSettings | None = None) -> None:
        self._setting = setting or RAGSettings()
        self._node_store = {}
        self._ingested_file = []

    def store_nodes(
        self,
        input_files: list[str],
        embed_nodes: bool = True,
        embed_model: Any | None = None
    ) -> List[BaseNode]:
        return_nodes = []
        self._ingested_file = []
        if len(input_files) == 0:
            return return_nodes
        splitter = SentenceSplitter.from_defaults(
            chunk_size=self._setting.ingestion.chunk_size,
            chunk_overlap=self._setting.ingestion.chunk_overlap,
            paragraph_separator=self._setting.ingestion.paragraph_sep,
            secondary_chunking_regex=self._setting.ingestion.chunking_regex
        )
        excluded_keys = [
            "doc_id", "file_path", "file_type", "page_label", "file_name",
            "file_size", "creation_date", "last_modified_date"
        ]
        if embed_nodes:
            Settings.embed_model = embed_model or Settings.embed_model
        for input_file in tqdm(input_files, desc="Ingesting data"):
            file_name = input_file.strip().split('/')[-1]
            self._ingested_file.append(file_name)
            if file_name in self._node_store:
                return_nodes.extend(self._node_store[file_name])
            else:
                document = SimpleDirectoryReader(
                    input_files=[input_file],
                    filename_as_id=True
                ).load_data()

                for doc in document:
                    doc.metadata['file_name'] = file_name
                    doc.text = re.sub(r'\s+', ' ', doc.text.strip())
                    doc.excluded_embed_metadata_keys = excluded_keys
                    doc.excluded_llm_metadata_keys = excluded_keys

                nodes = splitter(document, show_progress=True)
                if embed_nodes:
                    nodes = Settings.embed_model(nodes, show_progress=True)
                self._node_store[file_name] = nodes
                return_nodes.extend(nodes)
        return return_nodes

    def reset(self):
        self._node_store = {}
        self._ingested_file = []

    def check_nodes_exist(self):
        return len(self._node_store.values()) > 0

    def get_all_nodes(self):
        return_nodes = []
        for nodes in self._node_store.values():
            return_nodes.extend(nodes)
        return return_nodes

    def get_ingested_nodes(self):
        return_nodes = []
        for file in self._ingested_file:
            return_nodes.extend(self._node_store[file])
        return return_nodes
