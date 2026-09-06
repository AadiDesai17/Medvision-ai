import os
import json
import faiss

from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INDEX_PATH = os.path.join(
    BASE_DIR,
    "data",
    "faiss",
    "medical.index"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "faiss",
    "metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"


class MedicalRetriever:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            INDEX_PATH
        )

        print("Loading metadata...")

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)


    def retrieve(
        self,
        condition,
        top_k=5
    ):

        query = f"""
        Medical information about {condition}.
        Definition, symptoms, risk factors,
        clinical features and general management.
        """

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        query_embedding = query_embedding.reshape(
            1,
            -1
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            result = self.metadata[index].copy()

            result["similarity_score"] = float(
                score
            )

            results.append(result)

        return results
