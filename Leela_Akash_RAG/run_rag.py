from rag.retrieve import MedicalRetriever


def main():

    print("================================")
    print("MEDVISION MEDICAL RAG")
    print("================================")

    retriever = MedicalRetriever()

    condition = input(
        "\nEnter medical condition: "
    ).strip()

    if not condition:
        print("Please enter a condition.")
        return

    results = retriever.retrieve(
        condition,
        top_k=5
    )

    print("\n================================")
    print("RETRIEVED MEDICAL KNOWLEDGE")
    print("================================")

    print(f"\nQuery: {condition}")

    if not results:
        print("\nNo relevant information found.")
        return

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Similarity Score: "
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
