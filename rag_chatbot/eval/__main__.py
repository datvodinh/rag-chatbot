import os
import asyncio
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
)
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.storage.docstore import DocumentStore
from ..core.engine import LocalChatEngine, LocalRetriever
from ..core.model import LocalRAGModel
from ..core.embedding import LocalEmbedding
from ..setting import RAGSettings
from ..ollama import is_port_open, run_ollama_server

load_dotenv()


class RAGPipelineEvaluator:
    def __init__(
        self,
        llm: str | None = None,
        teacher: str | None = None,
        host: str = "host.docker.internal",
        dataset_path: str = "val_dataset/dataset.json",
        docstore_path: str = "val_dataset/docstore.json",
    ) -> None:
        self._setting = RAGSettings()
        top_k = self._setting.retriever.top_k_rerank
        self._llm = LocalRAGModel.set(model_name=llm, host=host)
        self._teacher = LocalRAGModel.set(model_name=teacher, host=host)
        self._engine = LocalChatEngine(host=host)
        Settings.llm = self._llm
        Settings.embed_model = LocalEmbedding.set()

        # dataset
        docstore = DocumentStore.from_persist_path(docstore_path)
        nodes = list(docstore.docs.values())
        index = VectorStoreIndex(nodes=nodes)
        self._dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)

        self._retriever = {
            "base": VectorIndexRetriever(
                index=index, similarity_top_k=top_k, verbose=True
            ),
            "bm25": BM25Retriever.from_defaults(
                index=index, similarity_top_k=top_k, verbose=True
            ),
            "router": LocalRetriever(host=host).get_retrievers(
                llm=self._llm, nodes=nodes
            ),
        }

        self._query_engine = {
            "base": RetrieverQueryEngine.from_args(
                retriever=self._retriever["base"],
                llm=self._llm,
            ),
            "bm25": RetrieverQueryEngine.from_args(
                retriever=self._retriever["bm25"],
                llm=self._llm,
            ),
            "router": RetrieverQueryEngine.from_args(
                retriever=self._retriever["router"],
                llm=self._llm,
            ),
        }

        self._retriever_evaluator = {
            "base": RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=self._retriever["base"]
            ),
            "bm25": RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=self._retriever["bm25"]
            ),
            "router": RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=self._retriever["router"]
            ),
        }

        self._faithfulness_evaluator = FaithfulnessEvaluator(
            llm=self._teacher,
        )

    async def eval_retriever(self):
        result = {}
        for name, retriever in self._retriever.items():
            print(f"Running {name} retriever")
            result[name] = await self._retriever_evaluator[name].aevaluate_dataset(
                self._dataset, show_progress=True
            )

        return result

    def display_results(self, name, eval_results):
        """Display results from evaluate."""

        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()
        metrics = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}

        return metrics


if __name__ == "__main__":
    # OLLAMA SERVER
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
        default="llama3:8b-instruct-q8_0",
        help="Set LLM model",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="gpt-4o",
        help="Set teacher model",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Set host to local or in docker container",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="harry_potter_dataset/dataset.json",
        help="Set dataset path",
    )
    parser.add_argument(
        "--docstore",
        type=str,
        default="harry_potter_dataset/docstore.json",
        help="Set docstore path",
    )
    args = parser.parse_args()
    if args.host != "host.docker.internal":
        port_number = 11434
        if not is_port_open(port_number):
            run_ollama_server()

    evaluator = RAGPipelineEvaluator(
        llm=args.llm,
        teacher=args.teacher,
        host=args.host,
        dataset_path=args.dataset,
        docstore_path=args.docstore,
    )

    retriever_result = asyncio.run(evaluator.eval_retriever())
    print(retriever_result)
    # save results
    with open("retriever_result.json", "w") as f:
        json.dump(retriever_result, f)
