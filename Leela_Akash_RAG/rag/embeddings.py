from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the Sentence Transformer embedding model."""
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, texts):
    """Convert text chunks into normalized vector embeddings."""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings
