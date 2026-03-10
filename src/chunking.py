from typing import List


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap

    return chunks
