import os
import json
import random

import pandas as pd
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy
)
from datasets import Dataset

from chat_graph.graph import ChatGraph


BENCHMARK_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/benchmarks/benchmarks/benchmark.json"


def read_benchmark(sample_size: int | None = None):
    with open(BENCHMARK_PATH) as json_file:
        full_benchmark = json.load(json_file)

    if sample_size:
        return random.sample(full_benchmark, sample_size)
    else:
        return full_benchmark


def evaluate_response(question, answer, context, ground_truth):
    result = evaluate(
        dataset=Dataset.from_dict({
            "question": question,
            "answer": answer,
            "contexts": context,
            "ground_truth": ground_truth,
        }),
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=random.choice(api_keys)),
        embeddings=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    )
    return result.to_pandas()


if __name__ == "__main__":
    load_dotenv("/Users/wnowogorski/PycharmProjects/CHAT_AGH/.env")

    benchmark = read_benchmark(sample_size=1)
    print("Number of questions: {}".format(len(benchmark)))

    ff_scores = []
    ac_scores = []
    result_df = pd.DataFrame(columns=["question", "answer", "contexts", "ground_truth"])
    for b in benchmark:
        question = b["question"]
        print("\n Question: {}".format(question))
        correct_answer = b["answer"]

        graph = ChatGraph()
        response = graph.invoke(question)
        answer = response["chat_history"][-1].content
        retrieved_chunks_dict = response["retrieved_chunks"] or {}
        retrieved_chunks = []
        for url in retrieved_chunks_dict.keys():
            retrieved_chunks.extend([c.get("text") for c in retrieved_chunks_dict[url]])
        print("Number of retrieved chunks: {}".format(len(retrieved_chunks)))
        print("Answer: {}".format(answer))
        print("Correct Answer: {}".format(correct_answer))

        api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))

        df = evaluate_response(question, answer, retrieved_chunks, correct_answer)
        result_df = pd.concat([result_df, df], ignore_index=True)

    result_df.to_json("benchmark_results.json")