import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    ResponseEvaluator,
)
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.storage.docstore import DocumentStore
from ..core.engine import LocalChatEngine, LocalRetriever
from ..core.model import LocalRAGModel

load_dotenv()


class RAGPipelineEvaluator:
    def __init__(
        self,
        llm: str | None = None,
        host: str = "host.docker.internal",
        eval_dir: str = "val_dataset"
    ) -> None:
        self._llm = LocalRAGModel.set(model_name=llm or "llama3:8b-instruct-q8_0")
        self._engine = LocalChatEngine(host=host)
        docstore = DocumentStore.from_persist_path(os.path.join(eval_dir, "docstore.json"))
        nodes = list(docstore.docs.values())
        index = VectorStoreIndex(nodes=nodes)
        self._base_retriever = VectorIndexRetriever(index=index)
        self._router_retriever = LocalRetriever(host=host).get_retrievers(
            llm=self._llm,
            nodes=nodes
        )

        # dataset
        self._dataset = EmbeddingQAFinetuneDataset.from_json(
            os.path.join(eval_dir, "dataset.json")
        )

        # evaluator
        self._base_retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=self._base_retriever
        )
        self._router_retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=self._router_retriever
        )

    async def get_accuracy(self):
        base_result = await self._base_retriever_evaluator.aevaluate_dataset(
            self._dataset, show_progress=True
        )
        router_result = await self._router_retriever_evaluator.aevaluate_dataset(
            self._dataset, show_progress=True
        )
        return base_result, router_result

    def get_correctness():
        pass  # TODO

    def get_faithfulness():
        pass  # TODO

    def get_relevancy():
        pass  # TODO

    def get_similarity():
        pass  # TODO

    def display_results(self, name, eval_results):
        """Display results from evaluate."""

        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()
        columns = {
            "retrievers": [name],
            "hit_rate": [hit_rate],
            "mrr": [mrr]
        }

        metric_df = pd.DataFrame(columns)

        return metric_df
