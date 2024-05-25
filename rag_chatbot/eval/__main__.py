import asyncio
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.storage.docstore import DocumentStore
from ..core.engine import LocalChatEngine, LocalRetriever
from ..core.model import LocalRAGModel
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
        if llm not in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo"]:
            print("Pulling LLM model")
            LocalRAGModel.pull(host=host, model_name=llm)
            print("Pulling complete")
        self._llm = LocalRAGModel.set(model_name=llm, host=host)
        self._teacher = LocalRAGModel.set(model_name=teacher, host=host)
        self._engine = LocalChatEngine(host=host)
        Settings.llm = self._llm
        # Settings.embed_model = LocalEmbedding.set()

        # dataset
        docstore = DocumentStore.from_persist_path(docstore_path)
        nodes = list(docstore.docs.values())
        self._index = VectorStoreIndex(nodes=nodes)
        self._dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)
        self._top_k = self._setting.retriever.similarity_top_k
        self._top_k_rerank = self._setting.retriever.top_k_rerank

        self._retriever = {
            "base": VectorIndexRetriever(
                index=self._index, similarity_top_k=self._top_k_rerank, verbose=True
            ),
            "bm25": BM25Retriever.from_defaults(
                index=self._index, similarity_top_k=self._top_k_rerank, verbose=True
            ),
            "base_rerank": VectorIndexRetriever(
                index=self._index, similarity_top_k=self._top_k, verbose=True
            ),
            "bm25_rerank": BM25Retriever.from_defaults(
                index=self._index, similarity_top_k=self._top_k, verbose=True
            ),
            "router": LocalRetriever(host=host).get_retrievers(
                llm=self._llm, nodes=nodes
            ),
        }

        self._retriever_evaluator = {
            "base": RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=self._retriever["base"]
            ),
            "bm25": RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=self._retriever["bm25"]
            ),
            "base_rerank": RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=self._retriever["base_rerank"],
                node_postprocessors=[
                    SentenceTransformerRerank(
                        top_n=self._top_k_rerank,
                        model=self._setting.retriever.rerank_llm,
                    )
                ],
            ),
            "bm25_rerank": RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=self._retriever["bm25_rerank"],
                node_postprocessors=[
                    SentenceTransformerRerank(
                        top_n=self._top_k_rerank,
                        model=self._setting.retriever.rerank_llm,
                    )
                ],
            ),
            "router": RetrieverEvaluator.from_metric_names(
                ["mrr", "hit_rate"], retriever=self._retriever["router"]
            ),
        }

        self._generator_evaluator = {
            "faithfulness": FaithfulnessEvaluator(
                llm=self._teacher,
            ),
            "answer_relevancy": AnswerRelevancyEvaluator(
                llm=self._teacher,
            ),
            "context_relevancy": ContextRelevancyEvaluator(
                llm=self._teacher
            )
        }

    async def eval_retriever(self):
        result = {}
        for retriever_name in self._retriever_evaluator.keys():
            print(f"Running {retriever_name} retriever")
            result[retriever_name] = self._process_retriever_result(
                retriever_name,
                await self._retriever_evaluator[retriever_name].aevaluate_dataset(
                    self._dataset, show_progress=True
                )
            )
        return result

    async def _query_with_delay(self, query_engine, q, delay):
        await asyncio.sleep(delay)
        return await query_engine.aquery(q)

    async def eval_generator(self):
        queries = list(self._dataset.queries.values())[:3]
        context = list(self._dataset.corpus.values())[:3]
        query_engine = self._index.as_query_engine(
            llm=self._llm,
        )
        response = []
        for i in range(0, len(queries), 10):
            print(f"Running queries {i} to {i+10}")
            task = [query_engine.aquery(q) for q in queries[i:i + 10]]
            response += await tqdm_asyncio.gather(*task, desc="querying")
            await asyncio.sleep(5)

        response = [str(r) for r in response]

        faithful_task = []
        answer_relevancy_task = []
        context_relevancy_task = []
        for q, r, c in zip(queries, response, context):
            faithful_task.append(
                self._generator_evaluator["faithfulness"].aevaluate(
                    response=r, contexts=[c]
                )
            )
            answer_relevancy_task.append(
                self._generator_evaluator["answer_relevancy"].aevaluate(
                    query=q, response=r
                )
            )
            context_relevancy_task.append(
                self._generator_evaluator["context_relevancy"].aevaluate(
                    query=q, contexts=[c]
                )
            )

        faithful_result = await tqdm_asyncio.gather(
            *faithful_task, desc="faithfulness"
        )
        answer_relevancy_result = await tqdm_asyncio.gather(
            *answer_relevancy_task, desc="answer_relevancy"
        )
        context_relevancy_result = await tqdm_asyncio.gather(
            *context_relevancy_task, desc="context_relevancy"
        )

        return {
            "faithfulness": self._process_generator_result(
                "faithfulness", faithful_result
            ),
            "answer_relevancy": self._process_generator_result(
                "answer_relevancy", answer_relevancy_result
            ),
            "context_relevancy": self._process_generator_result(
                "context_relevancy", context_relevancy_result
            ),
        }

    def _process_retriever_result(self, name, eval_results):
        """Display results from evaluate."""

        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()
        metrics = {"retrievers": name, "hit_rate": hit_rate, "mrr": mrr}

        return metrics

    def _process_generator_result(self, name, eval_results):
        result = []
        for r in eval_results:
            result.append(json.loads(r.json()))
        return {"generator": name, "result": result}


if __name__ == "__main__":
    # OLLAMA SERVER
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="retriever",
        choices=["retriever", "generator"],
        help="Set type to retriever or generator",
    )
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
        if not is_port_open(port_number) and args.llm not in ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo"]:
            run_ollama_server()
    evaluator = RAGPipelineEvaluator(
        llm=args.llm,
        teacher=args.teacher,
        host=args.host,
        dataset_path=args.dataset,
        docstore_path=args.docstore,
    )

    async def eval_retriever():
        retriever_result = await evaluator.eval_retriever()
        print(retriever_result)
        # save results
        with open("retriever_result.json", "w") as f:
            json.dump(retriever_result, f)

    async def eval_generator():
        generator_result = await evaluator.eval_generator()
        # save results
        with open(f"generator_result_{args.llm}.json", "w") as f:
            json.dump(generator_result, f)

    if args.type == "retriever":
        asyncio.run(eval_retriever())
    else:
        asyncio.run(eval_generator())
