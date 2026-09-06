import faiss
import numpy as np


def create_index(embeddings):
    """
    Create a FAISS index using normalized embeddings.
    Inner product is equivalent to cosine similarity
    when the embeddings are normalized.
    """
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search_index(index, query_embedding, top_k=5):
    """
    Search the FAISS index and return similarity scores
    and matching vector indices.
    """
    query_embedding = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores[0], indices[0]
