import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def answer_question(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[Video: {chunk['video_title']} | Chunk: {chunk['chunk_index']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""Answer the user's question using only the transcript excerpts below.
If the answer is not supported by the excerpts, say so.
Cite the video title when making important claims.

User question:
{question}

Transcript excerpts:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(
            f"[Video: {chunk['video_title']} | Chunk: {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)
    return f"""Answer the user's question using only the transcript excerpts below.
If the answer is not supported by the excerpts, say so.
Cite the video title when making important claims.

User question:
{question}

Transcript excerpts:
{context}
"""


def stream_answer_question(question: str, retrieved_chunks: list[dict]):
    """Yield text tokens from the LLM response as they arrive."""
    prompt = build_prompt(question, retrieved_chunks)
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
