from src.embeddings import embed_query
from src.retriever import search
from src.llm import answer_question


def ask(question: str, top_k: int = 5) -> dict:
    query_embedding = embed_query(question)
    chunks = search(query_embedding, top_k=top_k)
    answer = answer_question(question, chunks)

    return {
        "answer": answer,
        "sources": [
            {
                "video_title": c["video_title"],
                "chunk_index": c["chunk_index"],
                "url": c["url"],
                "score": c["score"],
                "text": c["text"],
            }
            for c in chunks
        ],
    }
