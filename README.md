# Pure Python RAG — Chat With Video Transcripts

A Retrieval-Augmented Generation (RAG) system built in pure Python. Add YouTube videos directly from the UI, then ask questions and get grounded answers with source citations.

---

## How It Works

```
YouTube URL  →  transcript download  →  chunking  →  embeddings  →  FAISS index
                                                                           ↓
user question  →  embed query  →  vector search  →  top-k chunks  →  GPT-5.4  →  answer
```

1. **Add** — paste a YouTube URL into the chat. The app downloads the transcript and video metadata automatically, then chunks, embeds, and indexes it locally.
2. **Query** — ask a question in the same chat. It's embedded, searched against the FAISS index, and the top matching chunks are sent to OpenAI GPT-5.4 to synthesize a grounded answer.
3. **Web UI** — a Flask app serves a chat interface at `http://localhost:5000`. The input field detects YouTube URLs automatically and switches between "Add" and "Send" mode.

![Prompt](https://github.com/user-attachments/assets/956d1afc-d9a1-43e5-a0ab-5b7f6c39d9c9)

Retrieval is entirely local. OpenAI is only used for the final answer generation step.

New to RAG? Read [TUTORIAL.md](TUTORIAL.md) for a ground-up explanation of how each component works.

---

## Project Structure

```
pyrag/
├── transcripts/          # Transcript .txt files (downloaded or manually added)
│   └── <video_id>.txt
├── data/                 # Auto-created by ingest (index + chunks)
│   ├── index.faiss
│   └── chunks.jsonl
├── src/
│   ├── transcript.py     # YouTube transcript downloader
│   ├── chunking.py       # Text chunking logic
│   ├── embeddings.py     # sentence-transformers wrapper
│   ├── retriever.py      # FAISS search
│   ├── llm.py            # OpenAI answer generation
│   ├── ingest.py         # Ingestion pipeline
│   └── query.py          # End-to-end query pipeline
├── templates/
│   └── index.html        # Web chat UI
├── app.py                # CLI entrypoint (ingest + serve)
├── requirements.txt
└── .env                  # API key config
```

---

## Prerequisites

- Python 3.10+
- An OpenAI API key

---

## Setup

**1. Clone the repo and create a virtual environment**

```bash
git clone <repo-url>
cd pyrag
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

On first run, `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB). This is cached locally after the first download.

**3. Configure your API key**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## Usage

### Start the web interface

```bash
python app.py serve
```

Open your browser to [http://localhost:5000](http://localhost:5000).

To run on a different port:

```bash
python app.py serve 8080
```

### Adding transcripts via the UI

Paste any YouTube URL into the input field. The border turns purple and the button changes to **"Add"** — press it (or hit Enter) to download and index the transcript. The UI shows a progress card while it works, then a confirmation with the video title and channel when done.

Supported URL formats:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`

### Asking questions

Type any question and press **Enter** (or click **Send**). Answers include clickable source chips showing which video each claim came from, linked back to the original YouTube video.

Use **Shift+Enter** for a newline in your question.

### Viewing indexed transcripts

Type `/transcripts` in the chat to see a list of all videos currently in the index. Each entry shows the video title, chunk count, and a link to the original YouTube video.

### CLI commands

```bash
python app.py ingest        # Process all .txt files in transcripts/ and rebuild the index
python app.py transcripts   # Print a table of all indexed transcripts
python app.py serve         # Start the web server (default port 5000)
python app.py serve 8080    # Start on a custom port
```

### Pre-loading transcripts via CLI

If you have existing `.txt` transcript files or want to bulk-load before starting the server:

```bash
python app.py ingest
```

This processes all `.txt` files in `transcripts/` and rebuilds the index. The web UI's Add flow does this automatically, so you only need this command for manual imports.

---

## Transcript File Format

Downloaded transcripts are saved automatically with the correct format. If you add files manually, use this structure:

```
Title: OWASP's Top 10 Ways to Attack LLMs: AI Vulnerabilities Exposed
VideoURL: https://www.youtube.com/watch?v=gUNXZMcd2jU
Speaker: IBM

You know what's catching a lot of teams off guard right now? How easy
it is for an LLM to leak something that it shouldn't, or be steered
into doing something you never intended...
```

**Header fields** (all optional):
| Field | Description |
|---|---|
| `Title` | Video title — shown in citations |
| `VideoURL` | Linked in source chips in the UI |
| `Speaker` | Channel or speaker name |

Files are named `<video_id>.txt` when downloaded via the UI.

---

## How Transcripts Are Downloaded

`src/transcript.py` handles the download flow:

1. Extracts the video ID from the URL
2. Calls YouTube's oEmbed API to fetch the video title and channel name (no API key required)
3. Uses `youtube-transcript-api` to fetch the auto-generated or manual transcript
4. Writes a formatted `.txt` file to `transcripts/` with the metadata header
5. Triggers a full re-ingest to rebuild the FAISS index

If a video has no available transcript (captions disabled, private video, etc.), the UI shows an error card with the reason.

---

## Chunking Strategy

Transcripts are split using a sliding window:

| Parameter | Default | Notes |
|---|---|---|
| Chunk size | 900 chars | ~150 words, works well for conversational text |
| Overlap | 120 chars | Preserves context across chunk boundaries |

To adjust, edit `src/chunking.py` or pass different values to `chunk_text()`.

---

## Embedding Model

The project uses [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from `sentence-transformers`:

- Runs entirely locally — no API calls for retrieval
- 384-dimensional embeddings
- Fast and accurate for semantic search on English text
- Model is cached in `~/.cache/torch/sentence_transformers/` after first download

---

## Answer Generation

Retrieved chunks are assembled into a prompt and sent to `gpt-5.4`. The prompt instructs the model to:

- Answer only from the provided transcript excerpts
- Cite the video title for each major claim
- Acknowledge when the answer isn't supported by the transcripts

To change the model, edit `src/llm.py`.

---

## Chunk Metadata

Each chunk stored in `data/chunks.jsonl` has the following shape:

```json
{
  "chunk_id": "gUNXZMcd2jU_0003",
  "video_id": "gUNXZMcd2jU",
  "video_title": "OWASP's Top 10 Ways to Attack LLMs: AI Vulnerabilities Exposed",
  "source_file": "gUNXZMcd2jU.txt",
  "chunk_index": 3,
  "text": "In the case of prompt injection, the user basically has control over the system. The reason this occurs is that LLMs are not very good at separating input from instructions...",
  "url": "https://www.youtube.com/watch?v=gUNXZMcd2jU"
}
```

This file is plain JSONL and can be inspected directly to debug retrieval quality.

---

## Troubleshooting

**"No transcripts indexed yet" error**
Add a YouTube video first by pasting a URL into the chat, or run `python app.py ingest` if you have files in `transcripts/`.

**Transcript download fails**
- The video may have captions disabled or be private/age-restricted
- Try a different video to confirm the setup is working

**Poor answer quality**
- Inspect `data/chunks.jsonl` to verify chunks look reasonable
- Increasing `top_k` in `src/query.py` retrieves more context (default: 5)

**OpenAI API errors**
- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check your OpenAI account has available quota

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Local embedding model |
| `faiss-cpu` | Vector similarity search |
| `openai` | GPT-5.4 answer generation |
| `flask` | Web server |
| `youtube-transcript-api` | YouTube transcript fetching |
| `python-dotenv` | `.env` file loading |
| `numpy` | Array handling for FAISS |
| `tqdm` | Progress bars during ingest |
