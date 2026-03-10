import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.chunking import chunk_text
from src.embeddings import embed_texts

TRANSCRIPTS_DIR = Path("transcripts")
DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH = DATA_DIR / "index.faiss"


def parse_metadata(lines: list[str]) -> tuple[dict, str]:
    """Extract header key-value pairs and return (metadata, body_text)."""
    metadata = {}
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            body_start = i + 1
            break
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            metadata[key.strip().lower()] = value.strip()

    body = "\n".join(lines[body_start:]).strip()
    return metadata, body


def ingest_transcripts():
    DATA_DIR.mkdir(exist_ok=True)

    txt_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {TRANSCRIPTS_DIR}/")
        return

    print(f"Found {len(txt_files)} transcript(s)")

    all_chunks = []

    for path in tqdm(txt_files, desc="Processing transcripts"):
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata, body = parse_metadata(lines)

        video_id = path.stem
        video_title = metadata.get("title", video_id)
        url = metadata.get("videourl", "")

        chunks = chunk_text(body)

        for i, chunk_text_val in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{video_id}_{i:04d}",
                "video_id": video_id,
                "video_title": video_title,
                "source_file": path.name,
                "chunk_index": i,
                "text": chunk_text_val,
                "url": url,
            })

    print(f"Generated {len(all_chunks)} chunks total")

    texts = [c["text"] for c in all_chunks]
    print("Generating embeddings...")
    embeddings = embed_texts(texts)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index to {INDEX_PATH}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Saved chunks to {CHUNKS_PATH}")

    print("Ingestion complete.")
