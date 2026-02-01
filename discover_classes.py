"""
Standalone CLI for Gemini-based video class discovery.

Uploads a video to Gemini File API, asks for all distinct manipulable object
class names, and prints the comma-separated list to stdout for shell consumption.
All status/diagnostic output goes to stderr.

Supports multiple discovery modes:
- simple: Basic 1-2 word class names (original behavior)
- attributed: Base classes + visual attributes (color, size, material)
- referring: Base classes + spatial/relational referring expressions
- full: All of the above
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


# --- Structured output schemas for Gemini ---

SCENE_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_type": {"type": "string"},
        "environment": {"type": "string", "enum": ["indoor", "outdoor"]},
        "primary_activity": {"type": "string"},
        "key_areas": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scene_type", "environment", "primary_activity", "key_areas"],
}

BASE_CLASS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "priority": {"type": "integer"},
            "confidence": {"type": "number"},
        },
        "required": ["name", "priority", "confidence"],
    },
}

ATTRIBUTED_CLASS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "base_class": {"type": "string"},
            "variants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "attributes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "attributes"],
                },
            },
        },
        "required": ["base_class", "variants"],
    },
}

REFERRING_EXPRESSION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "base_class": {"type": "string"},
            "expressions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "type": {"type": "string", "enum": ["spatial", "relational", "descriptive"]},
                    },
                    "required": ["text", "type"],
                },
            },
        },
        "required": ["base_class", "expressions"],
    },
}

SIMPLE_CLASS_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}


def _make_config(
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    schema: Optional[dict] = None,
) -> GenerateContentConfig:
    """Create a Gemini generation config with structured JSON output."""
    return GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=schema,
    )


def _log(msg: str) -> None:
    """Print a message to stderr."""
    print(msg, file=sys.stderr)


def _upload_video(client: genai.Client, video_path: str) -> Any:
    """Upload video to Gemini File API and wait for processing."""
    _log(f"Uploading video to Gemini File API: {video_path}")
    uploaded_file = client.files.upload(file=video_path)
    _log(f"Upload complete. File name: {uploaded_file.name}")

    while uploaded_file.state.name == "PROCESSING":
        _log("Waiting for video processing...")
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError(f"Video processing failed: {uploaded_file.state}")

    _log(f"Video ready. State: {uploaded_file.state.name}")
    return uploaded_file


def _cleanup_file(client: genai.Client, uploaded_file: Any) -> None:
    """Delete an uploaded file from Gemini."""
    try:
        client.files.delete(name=uploaded_file.name)
        _log(f"  Cleaned up uploaded file: {uploaded_file.name}")
    except Exception as e:
        _log(f"  Warning: Could not delete uploaded file: {e}")


def _log_usage(usage: Any, label: str) -> float:
    """Log token usage and return estimated cost."""
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
    _log(f"\n  {label}:")
    _log(f"    Input tokens:  {input_tokens}")
    _log(f"    Output tokens: {output_tokens}")
    _log(f"    Est. cost:     ${cost:.6f}")
    return cost


def _detect_scene_context(
    client: genai.Client, uploaded_file: Any
) -> Dict[str, Any]:
    """First pass: detect scene type, environment, and primary activity.

    Args:
        client: Gemini client.
        uploaded_file: Uploaded video file reference.

    Returns:
        Dict with scene_type, environment, primary_activity, and key_areas.
    """
    prompt = (
        "Analyze this video and describe the scene context. "
        "Return a JSON object with exactly these fields:\n"
        '{\n'
        '  "scene_type": "<kitchen/office/outdoor/living_room/bathroom/garage/store/restaurant/etc>",\n'
        '  "environment": "<indoor/outdoor>",\n'
        '  "primary_activity": "<cooking/working/eating/cleaning/shopping/exercising/etc>",\n'
        '  "key_areas": ["<area1>", "<area2>"]\n'
        '}\n'
        "Be specific about the scene type and activity based on what you observe."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, prompt],
        config=_make_config(temperature=0.1, schema=SCENE_CONTEXT_SCHEMA),
    )

    _log_usage(response.usage_metadata, "Scene context detection")

    result = json.loads(response.text)
    _log(f"  Scene context: {result}")
    return result


def _generate_base_classes(
    client: genai.Client,
    uploaded_file: Any,
    scene_context: Optional[Dict[str, Any]] = None,
    max_classes: int = 30,
) -> List[Dict[str, Any]]:
    """Discover base object classes, optionally scene-aware.

    Args:
        client: Gemini client.
        uploaded_file: Uploaded video file reference.
        scene_context: Optional scene context from _detect_scene_context.
        max_classes: Maximum number of classes to return.

    Returns:
        List of dicts with 'name', 'priority', and 'confidence' keys.
    """
    if scene_context:
        scene_type = scene_context.get("scene_type", "unknown")
        activity = scene_context.get("primary_activity", "unknown activity")
        prompt = (
            f"You are analyzing a video of a {scene_type} scene where someone is {activity}.\n\n"
            "List ALL distinct hand-graspable, liftable objects that appear at any point.\n"
            "Focus on:\n"
            f"- Small to medium-sized manipulable objects\n"
            f"- Objects typical for {scene_type} environments\n"
            "- Objects being actively used or interacted with\n"
            "- Ignore non-liftable objects like furniture, walls, floors, large appliances\n\n"
            f"Return ONLY a JSON array with up to {max_classes} objects. "
            "No explanation, no commentary, just the JSON array.\n"
            'Format: [{"name": "cup", "priority": 1, "confidence": 0.95}]\n'
            "Keep class names to 1-2 words. Sort by priority (1=most relevant)."
        )
    else:
        prompt = (
            "Watch this entire video carefully. "
            "List ALL distinct hand-graspable, liftable objects that appear at any point. "
            "Focus on small to medium-sized manipulable objects. "
            "Ignore furniture, walls, floors, large appliances. "
            f"Return ONLY a JSON array with up to {max_classes} objects. "
            "No explanation, no commentary, just the JSON array.\n"
            'Format: [{"name": "cup", "priority": 1, "confidence": 0.95}]\n'
            "Keep class names to 1-2 words. Sort by priority (1=most relevant)."
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, prompt],
        config=_make_config(temperature=0.2, schema=BASE_CLASS_SCHEMA),
    )

    _log_usage(response.usage_metadata, "Base class discovery")

    text = response.text
    try:
        classes = json.loads(text)
    except json.JSONDecodeError:
        # Response may be truncated — try to salvage by closing the array
        _log("  Warning: JSON parse failed, attempting to salvage truncated response...")
        # Find the last complete object (ends with })
        last_brace = text.rfind("}")
        if last_brace > 0:
            classes = json.loads(text[: last_brace + 1] + "]")
        else:
            raise
    if not isinstance(classes, list):
        raise ValueError(f"Expected JSON array, got: {type(classes)}")

    # Normalize names
    for item in classes:
        item["name"] = str(item["name"]).strip().lower()

    # Limit to max_classes
    classes = classes[:max_classes]
    _log(f"  Base classes found: {[c['name'] for c in classes]}")
    return classes


def _generate_attributed_classes(
    client: genai.Client,
    uploaded_file: Any,
    base_classes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate visually-attributed variants for base classes.

    Args:
        client: Gemini client.
        uploaded_file: Uploaded video file reference.
        base_classes: List of base class dicts from _generate_base_classes.

    Returns:
        List of dicts with 'name', 'base_class', and 'attributes' keys.
    """
    class_names = [c["name"] for c in base_classes]

    prompt = (
        "For each object class listed below, generate visually-attributed variants "
        "that describe specific instances visible in this video.\n"
        f"Base classes: {json.dumps(class_names)}\n\n"
        "Attributes should describe visual properties like color, size, material, "
        "or distinctive features. Only include variants that are actually visible "
        "in the video.\n\n"
        "Return a JSON array:\n"
        '[\n'
        '  {\n'
        '    "base_class": "<original class name>",\n'
        '    "variants": [\n'
        '      {"name": "<attributed name>", "attributes": ["<attr1>", "<attr2>"]}\n'
        '    ]\n'
        '  }\n'
        ']\n\n'
        "Keep attributed names to 2-4 words. Only include classes where distinct "
        "visual variants exist (e.g., skip if there's only one instance with no "
        "distinguishing attributes)."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, prompt],
        config=_make_config(temperature=0.3, schema=ATTRIBUTED_CLASS_SCHEMA),
    )

    _log_usage(response.usage_metadata, "Attributed class generation")

    raw = json.loads(response.text)
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array, got: {type(raw)}")

    # Flatten into a list of attributed classes
    attributed = []
    for entry in raw:
        base_class = str(entry.get("base_class", "")).strip().lower()
        for variant in entry.get("variants", []):
            attributed.append({
                "name": str(variant["name"]).strip().lower(),
                "base_class": base_class,
                "attributes": [str(a).strip().lower() for a in variant.get("attributes", [])],
            })

    _log(f"  Attributed classes: {[a['name'] for a in attributed]}")
    return attributed


def _generate_referring_expressions(
    client: genai.Client,
    uploaded_file: Any,
    base_classes: List[Dict[str, Any]],
    scene_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate referring expressions to uniquely identify objects.

    Args:
        client: Gemini client.
        uploaded_file: Uploaded video file reference.
        base_classes: List of base class dicts from _generate_base_classes.
        scene_context: Optional scene context for spatial grounding.

    Returns:
        List of dicts with 'expression', 'base_class', and 'type' keys.
    """
    class_names = [c["name"] for c in base_classes]
    context_str = ""
    if scene_context:
        context_str = f"\nScene context: {json.dumps(scene_context)}\n"

    prompt = (
        "Generate referring expressions to uniquely identify individual objects "
        "visible in this video. These should be spatial or relational descriptions "
        "that help locate specific object instances.\n"
        f"Base classes: {json.dumps(class_names)}\n"
        f"{context_str}\n"
        "Return a JSON array:\n"
        '[\n'
        '  {\n'
        '    "base_class": "<class name>",\n'
        '    "expressions": [\n'
        '      {"text": "<referring expression>", "type": "<spatial/relational/descriptive>"}\n'
        '    ]\n'
        '  }\n'
        ']\n\n'
        "Focus on expressions that disambiguate when multiple instances of the same "
        "class are visible. Keep expressions concise (under 10 words). "
        "Only include classes where disambiguation is useful."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, prompt],
        config=_make_config(temperature=0.3, schema=REFERRING_EXPRESSION_SCHEMA),
    )

    _log_usage(response.usage_metadata, "Referring expression generation")

    raw = json.loads(response.text)
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array, got: {type(raw)}")

    # Flatten into a list of expressions
    expressions = []
    for entry in raw:
        base_class = str(entry.get("base_class", "")).strip().lower()
        for expr in entry.get("expressions", []):
            expressions.append({
                "expression": str(expr["text"]).strip().lower(),
                "base_class": base_class,
                "type": str(expr.get("type", "descriptive")).strip().lower(),
            })

    _log(f"  Referring expressions: {[e['expression'] for e in expressions]}")
    return expressions


def discover_classes_enhanced(
    video_path: str,
    api_key: Optional[str] = None,
    mode: str = "attributed",
    scene_aware: bool = True,
    max_classes: int = 30,
) -> Dict[str, Any]:
    """Upload video to Gemini and perform multi-level class discovery.

    Args:
        video_path: Path to the video file.
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var).
        mode: Discovery mode - 'simple', 'attributed', 'referring', or 'full'.
        scene_aware: Whether to detect scene context first.
        max_classes: Maximum number of base classes to return.

    Returns:
        Enhanced discovery result dict with scene_context, classes, and
        batch_name_list.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No API key provided. Set GEMINI_API_KEY or pass --api-key.")

    client = genai.Client(api_key=key)
    uploaded_file = _upload_video(client, video_path)
    total_cost = 0.0

    try:
        # Step 1: Scene context detection (optional)
        scene_context = None
        if scene_aware:
            _log("\nStep 1/4: Detecting scene context...")
            scene_context = _detect_scene_context(client, uploaded_file)

        # Step 2: Base class discovery (always performed)
        _log("\nStep 2/4: Discovering base classes...")
        base_classes = _generate_base_classes(
            client, uploaded_file, scene_context, max_classes
        )

        # Step 3: Attributed classes (if mode is attributed or full)
        attributed_classes: List[Dict[str, Any]] = []
        if mode in ("attributed", "full"):
            _log("\nStep 3/4: Generating attributed class variants...")
            attributed_classes = _generate_attributed_classes(
                client, uploaded_file, base_classes
            )

        # Step 4: Referring expressions (if mode is referring or full)
        referring_expressions: List[Dict[str, Any]] = []
        if mode in ("referring", "full"):
            _log("\nStep 4/4: Generating referring expressions...")
            referring_expressions = _generate_referring_expressions(
                client, uploaded_file, base_classes, scene_context
            )

        # Build batch_name_list: base names + attributed names (deduplicated)
        seen = set()
        batch_name_list: List[str] = []

        # Add base class names first (highest priority)
        for cls in base_classes:
            name = cls["name"]
            if name not in seen:
                batch_name_list.append(name)
                seen.add(name)

        # Add attributed class names
        for attr in attributed_classes:
            name = attr["name"]
            if name not in seen:
                batch_name_list.append(name)
                seen.add(name)

        # Build result
        result: Dict[str, Any] = {
            "video": os.path.abspath(video_path),
            "mode": mode,
        }

        if scene_context:
            result["scene_context"] = scene_context

        result["classes"] = {
            "base": base_classes,
        }
        if attributed_classes:
            result["classes"]["attributed"] = attributed_classes
        if referring_expressions:
            result["classes"]["referring_expressions"] = referring_expressions

        result["batch_name_list"] = batch_name_list

        _log(f"\nDiscovery complete ({mode} mode):")
        _log(f"  Base classes:           {len(base_classes)}")
        _log(f"  Attributed variants:    {len(attributed_classes)}")
        _log(f"  Referring expressions:  {len(referring_expressions)}")
        _log(f"  Total batch_name_list:  {len(batch_name_list)}")

        return result

    finally:
        _cleanup_file(client, uploaded_file)


def discover_classes(video_path: str, api_key: Optional[str] = None) -> List[str]:
    """Upload video to Gemini File API and extract object class names.

    This is the original simple discovery function, maintained for backward
    compatibility. For enhanced discovery, use discover_classes_enhanced().

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
    uploaded_file = _upload_video(client, video_path)

    try:
        config = _make_config(temperature=0.2, schema=SIMPLE_CLASS_SCHEMA)

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

        class_names = json.loads(response.text)
        if not isinstance(class_names, list):
            raise ValueError(f"Expected JSON array, got: {type(class_names)}")
        class_names = [str(name).strip().lower() for name in class_names if name]

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)

        _log(f"\nGemini class discovery:")
        _log(f"  Classes found: {class_names}")
        _log(f"  Input tokens:  {input_tokens}")
        _log(f"  Output tokens: {output_tokens}")
        _log(f"  Est. cost:     ${cost:.6f}")

        return class_names

    finally:
        _cleanup_file(client, uploaded_file)


def _cache_path(video_path: str, mode: str = "simple") -> str:
    """Return the JSON cache file path for a given video and mode."""
    abs_video = os.path.abspath(video_path)
    base = os.path.splitext(abs_video)[0]
    if mode == "simple":
        return base + "_classes.json"
    return base + f"_classes_{mode}.json"


def load_cached_classes(
    video_path: str, mode: str = "simple"
) -> Optional[Any]:
    """Load cached classes from JSON file if it exists and is newer than the video.

    Args:
        video_path: Path to the video file.
        mode: Discovery mode used for caching.

    Returns:
        For simple mode: List[str] of class names.
        For enhanced modes: Full discovery result dict.
        None if cache is missing or stale.
    """
    cache = _cache_path(video_path, mode)
    if not os.path.exists(cache):
        return None
    if os.path.getmtime(cache) < os.path.getmtime(video_path):
        _log("Cache is stale (video modified after cache). Re-discovering.")
        return None
    with open(cache, "r") as f:
        data = json.load(f)

    if mode == "simple":
        classes = data.get("classes", [])
        _log(f"Loaded {len(classes)} cached classes from {cache}")
        return classes
    else:
        batch_list = data.get("batch_name_list", [])
        _log(f"Loaded cached discovery result ({len(batch_list)} classes) from {cache}")
        return data


def save_classes_cache(
    video_path: str,
    data: Any,
    mode: str = "simple",
) -> str:
    """Save discovered classes to a JSON file next to the video.

    Args:
        video_path: Path to the video file.
        data: For simple mode: List[str]. For enhanced: full result dict.
        mode: Discovery mode used for caching.

    Returns:
        The cache file path.
    """
    cache = _cache_path(video_path, mode)

    if mode == "simple":
        cache_data = {
            "video": os.path.abspath(video_path),
            "classes": data,
        }
    else:
        cache_data = data

    with open(cache, "w") as f:
        json.dump(cache_data, f, indent=2)

    if mode == "simple":
        _log(f"Saved {len(data)} classes to {cache}")
    else:
        _log(f"Saved enhanced discovery result to {cache}")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover object classes in a video using Gemini"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--api-key", default=None, help="Gemini API key (default: GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore cached classes and re-discover"
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "attributed", "referring", "full"],
        default="attributed",
        help="Discovery mode (default: attributed)",
    )
    parser.add_argument(
        "--scene-aware",
        action="store_true",
        default=True,
        help="Detect scene context first (default: true)",
    )
    parser.add_argument(
        "--no-scene-aware",
        dest="scene_aware",
        action="store_false",
        help="Disable scene context detection",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=30,
        help="Maximum base classes to return (default: 30)",
    )
    parser.add_argument(
        "--output-format",
        choices=["simple", "enhanced"],
        default="enhanced",
        help="Output format: 'simple' prints comma-separated names to stdout, "
        "'enhanced' prints full JSON (default: enhanced)",
    )
    args = parser.parse_args()

    # Load .env from script directory or project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(script_dir, ".env"))
    load_dotenv(os.path.join(script_dir, "..", ".env"))

    # Try loading from cache first
    if not args.no_cache:
        cached = load_cached_classes(args.video, mode=args.mode)
        if cached is not None:
            if args.output_format == "simple":
                if args.mode == "simple":
                    print(",".join(cached))
                else:
                    # Extract batch_name_list from enhanced result
                    names = cached.get("batch_name_list", [])
                    print(",".join(names))
            else:
                print(json.dumps(cached, indent=2))
            return

    # Run discovery
    if args.mode == "simple":
        classes = discover_classes(args.video, api_key=args.api_key)
        save_classes_cache(args.video, classes, mode="simple")
        if args.output_format == "simple":
            print(",".join(classes))
        else:
            # Wrap simple result in enhanced-like format
            result = {
                "video": os.path.abspath(args.video),
                "mode": "simple",
                "classes": {"base": [{"name": c, "priority": i + 1, "confidence": 1.0} for i, c in enumerate(classes)]},
                "batch_name_list": classes,
            }
            print(json.dumps(result, indent=2))
    else:
        result = discover_classes_enhanced(
            args.video,
            api_key=args.api_key,
            mode=args.mode,
            scene_aware=args.scene_aware,
            max_classes=args.max_classes,
        )
        save_classes_cache(args.video, result, mode=args.mode)
        if args.output_format == "simple":
            print(",".join(result["batch_name_list"]))
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
