import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from rag.utils.logger import logger
from rag.models.google_genai_models import (
    QueryAugmentationModel,
    EnhanceSearchModel,
    AnswerGenerationModel
)
from rag.vector_store.milvus_hybrid_search import MilvusHybridSearch

ENV_PATH = ".env"
NUM_RETRIEVED_CHUNKS = 20
MAX_SEARCH_ITERATIONS = 5


def inference(query):
    load_dotenv(dotenv_path=ENV_PATH)
    logger.info("Starting inference for query: {}".format(query))

    query_augmentation_model = QueryAugmentationModel()
    augmented_query = query_augmentation_model.generate(query)
    logger.info("Query augmented: \n {} \n\n".format(augmented_query))

    vector_store = MilvusHybridSearch("chatagh")
    # vector_store = PineconeHybridSearchVectorStore(os.environ["PINECONE_API_KEY"], "chatagh")
    source_docs = vector_store.search(query, NUM_RETRIEVED_CHUNKS)
    print(source_docs)

    logger.info("Retrieved {} chunks: \n {} \n\n".format(len(source_docs), source_docs))

    enhance_search_model = EnhanceSearchModel()
    summaries = []
    for i in range(MAX_SEARCH_ITERATIONS):
        summary, questions = enhance_search_model.generate(augmented_query, context=source_docs)
        logger.info("Enhance search model response: \n Summary: {}\n Questions: \n {}".format(summary, questions))

        if not (summary and questions):
            break

        summaries.append({"text": summary})

        questions = " \n".join(questions)
        source_docs = vector_store.search(questions, k=NUM_RETRIEVED_CHUNKS)

    source_docs.extend(summaries)
    logger.info("Final retrieval result: \n {} \n\n".format(source_docs))

    answer_generation_model = AnswerGenerationModel()
    final_response = answer_generation_model.generate(augmented_query, context=source_docs)
    logger.info("Generated response: \n {} \n\n".format(final_response))

    return final_response, source_docs


if __name__ == "__main__":
    query = "Czy mogę zakwaterować sie po blokadzie kwaterowania?"
    print(inference(query))
