import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python app.py ingest        - Process transcripts and build index")
        print("  python app.py transcripts   - List transcripts currently in the index")
        print("  python app.py serve         - Start the web chat interface")
        sys.exit(1)

    command = sys.argv[1]

    if command == "transcripts":
        import json
        from pathlib import Path

        chunks_path = Path("data/chunks.jsonl")
        if not chunks_path.exists():
            print("No index found. Run `python app.py ingest` first.")
            sys.exit(1)

        seen = {}
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                vid = chunk["video_id"]
                if vid not in seen:
                    seen[vid] = {"title": chunk["video_title"], "url": chunk["url"], "chunks": 0}
                seen[vid]["chunks"] += 1

        if not seen:
            print("Index is empty.")
            sys.exit(0)

        print(f"\n{'#':<4} {'Title':<55} {'Chunks':<8} URL")
        print("-" * 110)
        for i, (vid, info) in enumerate(seen.items(), 1):
            title = info["title"][:52] + "..." if len(info["title"]) > 55 else info["title"]
            print(f"{i:<4} {title:<55} {info['chunks']:<8} {info['url']}")
        print(f"\n{len(seen)} transcript(s) in index.\n")

    elif command == "ingest":
        from src.ingest import ingest_transcripts
        ingest_transcripts()

    elif command == "serve":
        from flask import Flask, render_template, request, jsonify, Response, stream_with_context
        from pathlib import Path
        import json as _json

        app = Flask(__name__)

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/ask", methods=["POST"])
        def ask_endpoint():
            data = request.get_json()
            question = (data or {}).get("question", "").strip()
            if not question:
                return jsonify({"error": "No question provided"}), 400

            index_path = Path("data/index.faiss")
            if not index_path.exists():
                return jsonify({
                    "error": "No transcripts indexed yet. Paste a YouTube URL to add one."
                }), 503

            from src.embeddings import embed_query
            from src.retriever import search
            from src.llm import stream_answer_question

            query_embedding = embed_query(question)
            chunks = search(query_embedding, top_k=5)

            sources = [
                {
                    "video_title": c["video_title"],
                    "chunk_index": c["chunk_index"],
                    "url": c["url"],
                    "score": c["score"],
                }
                for c in chunks
            ]

            def generate():
                yield f"data: {_json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                for token in stream_answer_question(question, chunks):
                    yield f"data: {_json.dumps({'type': 'token', 'text': token})}\n\n"
                yield f"data: {_json.dumps({'type': 'done'})}\n\n"

            return Response(stream_with_context(generate()), content_type="text/event-stream")

        @app.route("/transcripts", methods=["GET"])
        def transcripts_endpoint():
            import json
            chunks_path = Path("data/chunks.jsonl")
            if not chunks_path.exists():
                return jsonify([])

            seen = {}
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    chunk = json.loads(line)
                    vid = chunk["video_id"]
                    if vid not in seen:
                        seen[vid] = {
                            "video_id": vid,
                            "title": chunk["video_title"],
                            "url": chunk["url"],
                            "chunks": 0,
                        }
                    seen[vid]["chunks"] += 1

            return jsonify(list(seen.values()))

        @app.route("/add", methods=["POST"])
        def add_endpoint():
            data = request.get_json()
            url = (data or {}).get("url", "").strip()
            if not url:
                return jsonify({"error": "No URL provided"}), 400

            try:
                from src.transcript import download_and_save
                from src.ingest import ingest_transcripts

                info = download_and_save(url)
                ingest_transcripts()

                return jsonify({
                    "video_id": info["video_id"],
                    "title": info["title"],
                    "author": info["author"],
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        print(f"Starting web interface at http://localhost:{port}")
        app.run(debug=False, port=port)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
