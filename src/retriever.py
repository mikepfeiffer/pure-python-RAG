import faiss
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_index():
    return faiss.read_index(str(INDEX_PATH))


def search(query_embedding, top_k=5):
    index = load_index()
    chunks = load_chunks()

    query = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        item = chunks[idx]
        item["score"] = float(score)
        results.append(item)

    return results
