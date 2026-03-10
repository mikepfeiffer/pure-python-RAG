# RAG 101: Understanding Retrieval-Augmented Generation

This tutorial uses the `pyrag` application as a hands-on reference. By the end, you should be able to explain what RAG is, why it exists, how each component works, and where the interesting tradeoffs live.

---

## The Problem RAG Solves

Large language models are trained on a fixed snapshot of data. Once training is complete, the model's knowledge is frozen. Ask it about something that happened after its training cutoff, or about a document it has never seen, and it has two options:

1. Admit it doesn't know
2. Make something up — confidently

Option 2 is the hallucination problem. LLMs are next-token prediction engines. When they don't have the information needed to answer a question, they don't stop and say "I don't know" — they keep predicting plausible-sounding tokens, which produces fluent but fabricated answers.

RAG is the standard technique for grounding LLM responses in a specific body of knowledge, without retraining or fine-tuning the model.

---

## What RAG Actually Is

RAG stands for **Retrieval-Augmented Generation**. The name describes the technique:

- **Retrieval** — find relevant information from a document store
- **Augmented** — add that information to the prompt
- **Generation** — let the LLM generate an answer using it

In plain terms: before asking the model a question, you look up relevant passages from your documents and hand them to the model along with the question. The model's job is then synthesis, not recall.

```
Your documents  →  [retrieve relevant chunks]  →  [inject into prompt]  →  LLM  →  answer
```

This is powerful because it separates two concerns:
- **What to look up** — handled by vector search (fast, local, inspectable)
- **How to answer** — handled by the LLM (flexible, fluent, generalizable)

---

## The RAG Pipeline, Step by Step

### Step 1 — Ingestion (offline)

Before you can answer questions, you need to prepare your documents. This is called ingestion.

**In this app:** run `python app.py ingest` or paste a YouTube URL into the chat.

The ingestion pipeline does four things:

#### 1a. Load and parse documents

Each transcript file is read and parsed. The header metadata (title, URL, speaker) is extracted, and the body text is isolated.

**Code:** [`src/ingest.py`](src/ingest.py) — `parse_metadata()`

#### 1b. Chunk the text

The full transcript is split into overlapping chunks of ~900 characters. Why chunk?

- LLMs have a finite context window — you can't stuff an entire document collection into a prompt
- Smaller chunks mean more precise retrieval — you retrieve the paragraph that's relevant, not the whole document
- Overlapping chunks prevent a relevant sentence from being split across a chunk boundary

**Code:** [`src/chunking.py`](src/chunking.py)

```
"...Azure has three types of blobs. Block blobs store..."
                         ↓  chunk  ↓
chunk_0: "Azure has three types of blobs. Block blobs store text and binary data..."
chunk_1: "...binary data. Append blobs are optimized for append operations..."
                              ↑ overlap ↑
```

The overlap (120 chars) means consecutive chunks share a window of text. This is important because a question might be answered by content that straddles a chunk boundary.

#### 1c. Embed the chunks

Each chunk is converted to a vector — a list of numbers that represents its semantic meaning. Chunks with similar meaning end up with similar vectors.

**Code:** [`src/embeddings.py`](src/embeddings.py) — uses `all-MiniLM-L6-v2` from `sentence-transformers`

This model runs entirely locally. It produces 384-dimensional vectors. The important property: the distance between two vectors corresponds to semantic similarity. "What is a subnet?" and "A subnet is a logical subdivision of a VNet" will have vectors close together in this 384-dimensional space.

#### 1d. Store chunks and build the index

The chunk text and metadata are written to `data/chunks.jsonl`. The embedding vectors are loaded into a FAISS index and saved to `data/index.faiss`.

**FAISS** (Facebook AI Similarity Search) is a library for fast nearest-neighbor search over large collections of vectors. The index is essentially a data structure that lets you say "give me the 5 vectors most similar to this query vector" very efficiently.

**Code:** [`src/ingest.py`](src/ingest.py) — `ingest_transcripts()`

---

### Step 2 — Retrieval (at query time)

When a user asks a question, the retrieval step finds the most relevant chunks.

**Code:** [`src/retriever.py`](src/retriever.py) — `search()`

#### 2a. Embed the query

The user's question is converted to a vector using the same embedding model. This is critical — the question and the chunks must live in the same vector space for the similarity comparison to be meaningful.

#### 2b. Search the FAISS index

FAISS computes the distance between the query vector and every chunk vector in the index and returns the top-k closest matches. By default this app retrieves 5 chunks.

**Code:** [`src/retriever.py`](src/retriever.py)

```python
distances, indices = index.search(query, top_k)
```

The `distances` are L2 (Euclidean) distances — smaller means more similar. The `indices` map back to positions in `chunks.jsonl`, allowing the full chunk text and metadata to be retrieved.

---

### Step 3 — Generation (at query time)

The retrieved chunks and the original question are assembled into a prompt and sent to the LLM.

**Code:** [`src/llm.py`](src/llm.py) — `build_prompt()` and `stream_answer_question()`

The prompt follows a simple but important pattern:

```
Answer the user's question using only the transcript excerpts below.
If the answer is not supported by the excerpts, say so.
Cite the video title when making important claims.

User question:
{question}

Transcript excerpts:
{context}
```

The key instruction is "using only the transcript excerpts below." This is what grounds the response. Without this constraint, the model would draw on its training data, which defeats the purpose of RAG.

The model's job is now:
1. Read the retrieved chunks
2. Synthesize an answer
3. Cite where the information came from

---

## The Two Distinct Systems in RAG

Understanding RAG requires seeing that it combines two very different technologies:

| | Vector Search | LLM |
|---|---|---|
| **Purpose** | Find relevant text | Generate a response |
| **Input** | A query vector | A prompt string |
| **Output** | Top-k chunk indices + distances | Text |
| **Runs where** | Locally (in this app) | OpenAI API |
| **Deterministic?** | Yes | No |
| **Inspectable?** | Yes (`chunks.jsonl`) | No |

This separation is why RAG is valuable:
- The retrieval step is auditable — you can inspect exactly what chunks were returned
- The LLM is only asked to do what it's good at: reading comprehension and synthesis
- You can swap out either component independently

---

## Key Concepts to Understand

### Embeddings

An embedding is a dense vector representation of text. The embedding model has learned to map text to points in a high-dimensional space such that semantically similar text maps to nearby points.

"What is a virtual network?" and "A VNet is a logically isolated network in Azure" are semantically similar — they would produce nearby embeddings — even though they share almost no words.

This is fundamentally different from keyword search (like grep), which requires exact word matches.

### Chunk Size and Overlap

These are the two most impactful parameters in a RAG system and have a direct tradeoff:

- **Smaller chunks** → more precise retrieval, but may lose context
- **Larger chunks** → more context preserved, but retrieval is less precise (you might retrieve a 2000-character chunk because one sentence in it is relevant)
- **More overlap** → better continuity across boundaries, but more storage and more redundant content

The defaults in this app (900 chars, 120 overlap) are a reasonable starting point for conversational transcripts.

To experiment: edit `src/ingest.py` and change the `chunk_size` and `overlap` parameters, then re-run `python app.py ingest` and compare retrieval quality.

### Top-k

The number of chunks retrieved and sent to the LLM. In this app, `top_k=5`.

- **Higher k** → the LLM has more context, useful for questions that span multiple parts of a document
- **Lower k** → the LLM has less to read through, and the signal-to-noise ratio is higher

To experiment: change `top_k` in `src/query.py` and observe whether answer quality improves or degrades for different question types.

### The Grounding Constraint

The phrase "using only the transcript excerpts below" is doing a lot of work in the prompt. Remove it, and the model will freely mix retrieved context with training data — making it hard to know where an answer came from.

This constraint is what turns RAG into a reliable, auditable system rather than an LLM that happens to have some context injected.

---

## What You Can Inspect

One of the best things about this architecture is how much of it is visible.

**`data/chunks.jsonl`** — open this file in any editor and you can read every chunk. You can see exactly what text was stored, how it was chunked, and what metadata is attached.

**The retrieval step** — you can add a `print(chunks)` in `src/query.py` after the `search()` call to see exactly what chunks were retrieved for any given question, along with their similarity scores.

**The prompt** — add a `print(prompt)` in `src/llm.py` inside `build_prompt()` to see the exact text being sent to the model.

Inspectability is a core advantage of simple RAG over more opaque approaches.

---

## Hands-On Experiments

### 1. Add a transcript and ask a question

Start the app and paste a YouTube URL. Once indexed, ask a specific question about the video content. Notice that the answer cites the video title.

### 2. Ask a question the transcripts can't answer

Ask about something not covered in any indexed video. The model should respond that it doesn't have enough information — this is the grounding constraint working correctly.

### 3. Inspect the retrieved chunks

Add a `print` statement in `src/query.py` after the `search()` call:

```python
for c in chunks:
    print(f"Score: {c['score']:.4f} | {c['video_title']} | chunk {c['chunk_index']}")
    print(c['text'][:200])
    print()
```

Run the app and ask a question. You'll see exactly which chunks are being retrieved and their similarity scores.

### 4. Explore `chunks.jsonl`

Open `data/chunks.jsonl` and read through some chunks. Notice how the text is chunked — where the boundaries fall, whether the 120-character overlap is preserving context. This builds intuition for how chunking strategy affects retrieval quality.

### 5. Try adjusting `top_k`

In `src/query.py`, change `top_k=5` to `top_k=2` and ask a broad question. Then change it to `top_k=10` and ask the same question. Compare the answers.

---

## How to Explain RAG

After working with this app, you should be able to explain RAG in plain terms:

> RAG is a pattern for grounding LLM responses in a specific set of documents. At index time, documents are split into chunks and converted to semantic vectors using an embedding model. At query time, the user's question is embedded the same way and compared against the stored vectors to find the most relevant chunks. Those chunks are included in the prompt sent to the LLM, which synthesizes a grounded answer. The key insight is that retrieval and generation are handled by two separate systems — a vector search engine for finding relevant content, and an LLM for reading comprehension and synthesis.

The important things to be able to articulate:
- Why chunking is necessary (context window limits, retrieval precision)
- What an embedding is and why it enables semantic rather than keyword search
- Why the same embedding model must be used for both indexing and querying
- What "grounding" means and how the prompt instruction enforces it
- The tradeoff between chunk size/overlap and retrieval quality

---

## What This App Deliberately Leaves Out

This is a foundational RAG implementation. Several common production techniques are intentionally absent:

| Technique | What it does | Why it's not here |
|---|---|---|
| Reranking | A second model re-scores retrieved chunks for higher precision | Adds complexity; not needed to understand the core loop |
| Hybrid search | Combines vector search with keyword search (BM25) | More infrastructure; keyword search alone is often sufficient |
| HyDE | Generates a hypothetical answer to improve retrieval | Advanced; obscures the basic flow |
| Metadata filtering | Restricts search to a subset of documents | Useful at scale, not needed for a handful of transcripts |
| Streaming ingestion | Update the index incrementally without full reprocessing | This app does full re-ingest, which is fine for learning |

Master the basics here before layering in these techniques.

---

## Summary

| Concept | Where to see it in this app |
|---|---|
| Document chunking | `src/chunking.py`, `data/chunks.jsonl` |
| Embedding generation | `src/embeddings.py` |
| Vector index (FAISS) | `data/index.faiss`, `src/ingest.py` |
| Semantic search | `src/retriever.py` |
| Prompt construction | `src/llm.py` — `build_prompt()` |
| Grounded generation | `src/llm.py` — the system prompt instruction |
| Streaming response | `src/llm.py` — `stream_answer_question()`, `app.py` SSE endpoint |
