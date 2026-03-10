import json
import urllib.request
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

TRANSCRIPTS_DIR = Path("transcripts")


def extract_video_id(url: str) -> str:
    url = url.strip()
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url


def fetch_metadata(video_id: str) -> dict:
    oembed_url = (
        f"https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch%3Fv%3D{video_id}&format=json"
    )
    try:
        with urllib.request.urlopen(oembed_url, timeout=10) as resp:
            data = json.loads(resp.read())
        return {
            "title": data.get("title", video_id),
            "author": data.get("author_name", "Unknown"),
        }
    except Exception:
        return {"title": video_id, "author": "Unknown"}


def download_and_save(url: str) -> dict:
    """Download a YouTube transcript and save it as a formatted .txt file.

    Returns a dict with video_id, title, author, and path.
    Raises an exception if the transcript cannot be fetched.
    """
    video_id = extract_video_id(url)
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    metadata = fetch_metadata(video_id)

    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    full_text = " ".join(snippet.text for snippet in transcript)

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    out_path = TRANSCRIPTS_DIR / f"{video_id}.txt"

    content = (
        f"Title: {metadata['title']}\n"
        f"VideoURL: {video_url}\n"
        f"Speaker: {metadata['author']}\n"
        f"\n"
        f"{full_text}\n"
    )
    out_path.write_text(content, encoding="utf-8")

    return {
        "video_id": video_id,
        "title": metadata["title"],
        "author": metadata["author"],
        "path": str(out_path),
    }
