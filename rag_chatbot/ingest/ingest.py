from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()


class DataIngestion:
    @staticmethod
    def get_documents(input_dir: str = None, input_files: list = None):
        documents = SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
            filename_as_id=True
        ).load_data(show_progress=True)
        for doc in documents:
            doc.excluded_embed_metadata_keys = ["doc_id"]
            doc.excluded_llm_metadata_keys = [
                "file_name", "doc_id", "page_label", "file_path",
                "file_type", "file_size", "creation_date", "last_modified_date"
            ]
        return documents

    @staticmethod
    def get_nodes(
        documents: list[Document],
        chunk_size: int,
        chunk_overlap: int
    ):
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # GET NODES FROM DOCUMENTS
        nodes = splitter.get_nodes_from_documents(documents)
        return nodes
