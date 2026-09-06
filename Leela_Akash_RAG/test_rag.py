from rag.retrieve import MedicalRetriever


def main():

    retriever = MedicalRetriever()

    condition = "Pneumonia"

    results = retriever.retrieve(
        condition,
        top_k=5
    )

    print("\n")
    print("================================")
    print("RAG SEARCH RESULTS")
    print("================================")

    print(f"\nQuery: {condition}\n")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Result {i} ---")

        print(
            f"Score: "
            f"{result['similarity_score']:.4f}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Category: "
            f"{result['category']}"
        )

        print(
            f"\n{result['text']}"
        )


if __name__ == "__main__":
    main()
