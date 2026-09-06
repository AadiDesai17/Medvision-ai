def chunk_text(text, chunk_size=150, overlap=30):
    """
    Split medical text into overlapping word-based chunks.

    Args:
        text: Input medical document text.
        chunk_size: Maximum number of words in each chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        List of text chunks.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
