from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')


def bm25_similarity(text1: str, text2: str) -> float:
    """
    Compute the BM25 similarity score between two texts.

    Args:
        text1 (str): First text (treated as the document).
        text2 (str): Second text (treated as the query).

    Returns:
        float: BM25 similarity score.
    """
    doc_tokens = word_tokenize(text1.lower())
    query_tokens = word_tokenize(text2.lower())

    bm25 = BM25Okapi([doc_tokens])

    return -bm25.get_scores(query_tokens)[0]

if __name__ == '__main__':
    score = bm25_similarity("The quick brown fox jumps over the lazy dog",
                            "A fast brown animal leaped over a sleepy dog")
    print(f"BM25 similarity score: {score}")
