import os
import functools
import time
import logging

import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


logger = logging.getLogger("chat_graph_logger")
logger.setLevel(logging.INFO)

mongo_client = MongoClient(os.environ.get("MONGODB_URI"), tlsAllowInvalidCertificates=True)

embedding_model = (
    SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
    if torch.cuda.is_available()
    else SentenceTransformer("intfloat/multilingual-e5-large")
)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        fun_name = func.__name__
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        logger.info(f"[{class_name}.{fun_name}] Execution time: {end - start:.4f}s")
        return result
    return wrapper


def retry_on_exception(attempts=3, delay=1, backoff=10, exception=Exception):
    """
    A decorator to retry a function call if it raises a specified exception.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    if attempt == attempts:
                        raise
                    else:
                        print(f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff

        return wrapper

    return decorator
