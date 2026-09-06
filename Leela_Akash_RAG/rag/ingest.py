import os
import json
import faiss

from chunking import chunk_text
from embeddings import load_embedding_model, create_embeddings
from vector_store import create_index


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWLEDGE_DIR = os.path.join(
    BASE_DIR,
    "medical_knowledge"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "faiss"
)


def load_documents():
    documents = []

    for root, dirs, files in os.walk(KNOWLEDGE_DIR):

        for filename in files:

            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(root, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()

            chunks = chunk_text(text)

            category = os.path.basename(root)

            for chunk_id, chunk in enumerate(chunks):

                documents.append({
                    "text": chunk,
                    "source": filename,
                    "path": filepath,
                    "category": category,
                    "chunk_id": chunk_id
                })

    return documents


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading medical documents...")

    documents = load_documents()

    print(
        f"Documents/chunks created: {len(documents)}"
    )

    texts = [
        item["text"]
        for item in documents
    ]

    print("\nLoading embedding model...")

    model = load_embedding_model()

    print("\nCreating embeddings...")

    embeddings = create_embeddings(
        model,
        texts
    )

    print("\nCreating FAISS index...")

    index = create_index(embeddings)

    index_path = os.path.join(
        OUTPUT_DIR,
        "medical.index"
    )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "metadata.json"
    )

    faiss.write_index(
        index,
        index_path
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n================================")
    print("RAG KNOWLEDGE BASE CREATED")
    print("================================")

    print(f"FAISS index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Total chunks: {len(documents)}")


if __name__ == "__main__":
    main()
