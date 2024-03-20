import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)


class RAGPipeline:
    def __init__(self) -> None:
        Settings.chunk_size = 512
        Settings.llm = None
        self.has_docs = False
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/e5-large-v2", max_length=512, cache_folder="./data/huggingface/"
        )

    def pull_embed_model(self):
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/e5-large-v2", max_length=512, cache_folder="./data/huggingface/"
        )

    def add_document(self):
        self.has_docs = True
        PERSIST_DIR = os.path.join(os.getcwd(), "data/storage")
        # if not os.path.exists(PERSIST_DIR):
            # load the documents and create the index
        documents = SimpleDirectoryReader(os.path.join(os.getcwd(), "data/data")).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
            # store it for later
            # index.storage_context.persist(persist_dir=PERSIST_DIR)
        # else:
        #     # load the existing index
        #     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        #     index = load_index_from_storage(storage_context)
        self.query_engine = index.as_query_engine(llm=None)

    def query(self, question: str):
        return str(self.query_engine.query(question))

    def get_rag_prompt():
        pass
