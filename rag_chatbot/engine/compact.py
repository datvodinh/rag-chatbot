from dotenv import load_dotenv
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from ..prompt import get_qa_and_refine_prompt
from ..setting import RetrieverSettings
from .retriever import LocalRetriever

load_dotenv()


class LocalCompactEngine:
    def __init__(
        self,
        setting: RetrieverSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RetrieverSettings()
        self._retriever = LocalRetriever(self._setting)
        self._host = host

    def from_index(
        self,
        llm,
        vector_index: VectorStoreIndex,
        language: str,
    ) -> RetrieverQueryEngine:
        retriever = self._retriever.get_retrievers(
            vector_index=vector_index,
            language=language
        )
        qa_template, refine_template = get_qa_and_refine_prompt(language)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(
                llm=llm,
                text_qa_template=qa_template,
                refine_template=refine_template,
                response_mode="compact",
                streaming=True,
                verbose=True
            ),
            node_postprocessors=[
                MetadataReplacementPostProcessor(
                    target_metadata_key="window"
                )
            ]
        )
        return query_engine
