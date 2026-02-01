"""
Standalone CLI for Gemini-based video class discovery.

Uploads a video to Gemini File API, asks for all distinct manipulable object
class names, and prints the comma-separated list to stdout for shell consumption.
All status/diagnostic output goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


def discover_classes(video_path: str, api_key: Optional[str] = None) -> List[str]:
    """Upload video to Gemini File API and extract object class names.

    Args:
        video_path: Path to the video file.
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var).

    Returns:
        List of short object class name strings.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No API key provided. Set GEMINI_API_KEY or pass --api-key.")

    client = genai.Client(api_key=key)

    # Upload video file
    print(f"Uploading video to Gemini File API: {video_path}", file=sys.stderr)
    uploaded_file = client.files.upload(file=video_path)
    print(f"Upload complete. File name: {uploaded_file.name}", file=sys.stderr)

    # Wait for file to be processed
    while uploaded_file.state.name == "PROCESSING":
        print("Waiting for video processing...", file=sys.stderr)
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError(f"Video processing failed: {uploaded_file.state}")

    print(f"Video ready. State: {uploaded_file.state.name}", file=sys.stderr)

    # Ask Gemini for class names
    config = GenerateContentConfig(
        temperature=0.2,
        thinking_config=ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )

    prompt = (
        "Watch this entire video carefully. "
        "List ALL distinct hand-graspable, liftable objects that appear at any point. "
        "Focus on small to medium-sized manipulable objects such as cups, bottles, tools, "
        "phones, books, food items, utensils, toys, containers, etc. "
        "Ignore non-liftable objects like furniture, walls, floors, large appliances, and structural elements. "
        "Return a JSON array of short class name strings (1-2 words each). "
        'Example: ["cup", "phone", "pen", "notebook", "water bottle"]'
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, prompt],
        config=config,
    )

    # Parse class names
    class_names = json.loads(response.text)
    if not isinstance(class_names, list):
        raise ValueError(f"Expected JSON array, got: {type(class_names)}")
    class_names = [str(name).strip().lower() for name in class_names if name]

    # Token usage
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

    print(f"\nGemini class discovery:", file=sys.stderr)
    print(f"  Classes found: {class_names}", file=sys.stderr)
    print(f"  Input tokens:  {input_tokens}", file=sys.stderr)
    print(f"  Output tokens: {output_tokens}", file=sys.stderr)
    print(f"  Est. cost:     ${cost:.6f}", file=sys.stderr)

    # Clean up uploaded file
    try:
        client.files.delete(name=uploaded_file.name)
        print(f"  Cleaned up uploaded file: {uploaded_file.name}", file=sys.stderr)
    except Exception as e:
        print(f"  Warning: Could not delete uploaded file: {e}", file=sys.stderr)

    return class_names


def _cache_path(video_path: str) -> str:
    """Return the JSON cache file path for a given video."""
    abs_video = os.path.abspath(video_path)
    return os.path.splitext(abs_video)[0] + "_classes.json"


def load_cached_classes(video_path: str) -> Optional[List[str]]:
    """Load cached classes from JSON file if it exists and is newer than the video."""
    cache = _cache_path(video_path)
    if not os.path.exists(cache):
        return None
    # Check that cache is newer than video file
    if os.path.getmtime(cache) < os.path.getmtime(video_path):
        print(f"Cache is stale (video modified after cache). Re-discovering.", file=sys.stderr)
        return None
    with open(cache, "r") as f:
        data = json.load(f)
    classes = data.get("classes", [])
    print(f"Loaded {len(classes)} cached classes from {cache}", file=sys.stderr)
    return classes


def save_classes_cache(video_path: str, classes: List[str]) -> str:
    """Save discovered classes to a JSON file next to the video. Returns the cache path."""
    cache = _cache_path(video_path)
    data = {
        "video": os.path.abspath(video_path),
        "classes": classes,
    }
    with open(cache, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(classes)} classes to {cache}", file=sys.stderr)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover object classes in a video using Gemini")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--api-key", default=None, help="Gemini API key (default: GEMINI_API_KEY env var)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached classes and re-discover")
    args = parser.parse_args()

    # Load .env from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(script_dir, ".env"))

    # Try loading from cache first
    if not args.no_cache:
        cached = load_cached_classes(args.video)
        if cached:
            print(",".join(cached))
            return

    classes = discover_classes(args.video, api_key=args.api_key)
    save_classes_cache(args.video, classes)
    # Print comma-separated list to stdout for shell consumption
    print(",".join(classes))


if __name__ == "__main__":
    main()
