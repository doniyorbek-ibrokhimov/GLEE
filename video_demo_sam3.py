"""SAM 3 video detection + segmentation + tracking pipeline.

Replaces GLEE + SAM v1 + SORT with a single SAM 3 model that provides
detection, segmentation, and tracking natively. Outputs the same JSON
schema as video_demo.py for downstream compatibility (json_to_npz.py,
run_reconstruction.py, etc.).

Usage:
    python video_demo_sam3.py \
        --input_video ../raw_videos/sample.mp4 \
        --output_video ../output_videos/sample_sam3.mp4 \
        --classes "pizza,plate,hand" \
        --confidence_threshold 0.3
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# SAM 3 imports (ultralytics backend)
try:
    from ultralytics.models.sam.predict import SAM3SemanticPredictor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3SemanticPredictor = None
    SAM3_AVAILABLE = False
    print("WARNING: ultralytics not installed or SAM3 not available. Run: pip install ultralytics")


# 20 distinct colors for track visualization (RGB order)
TRACK_COLORS: List[Tuple[int, int, int]] = [
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (255, 225, 25),   # yellow
    (0, 130, 200),    # blue
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
    (0, 128, 128),    # teal
    (220, 190, 255),  # lavender
    (170, 110, 40),   # brown
    (255, 250, 200),  # beige
    (128, 0, 0),      # maroon
    (170, 255, 195),  # mint
    (128, 128, 0),    # olive
    (255, 215, 180),  # coral
    (0, 0, 128),      # navy
    (128, 128, 128),  # grey
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic RGB color for a given track ID."""
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


def load_discovered_classes(
    discovery_json_path: str,
    mode: str = "attributed",
) -> Tuple[List[str], Optional[Dict]]:
    """Extract classes from an enhanced discovery result JSON file.

    Args:
        discovery_json_path: Path to the discovery result JSON file.
        mode: Which class level to extract - 'simple' uses only base class
            names, 'attributed' adds attributed variants.

    Returns:
        Tuple of (class_names, discovery_data).
    """
    with open(discovery_json_path, "r") as f:
        data = json.load(f)

    if "batch_name_list" in data:
        names = data["batch_name_list"]
    elif "classes" in data:
        classes_section = data["classes"]
        if isinstance(classes_section, dict):
            names = [c["name"] for c in classes_section.get("base", [])]
            if mode in ("attributed", "full"):
                for attr in classes_section.get("attributed", []):
                    name = attr.get("name", "")
                    if name and name not in names:
                        names.append(name)
        elif isinstance(classes_section, list):
            names = [str(c) for c in classes_section]
        else:
            names = []
    else:
        names = []

    if mode == "simple":
        base_names = []
        classes_section = data.get("classes", {})
        if isinstance(classes_section, dict):
            base_names = [c["name"] for c in classes_section.get("base", [])]
        if base_names:
            names = base_names

    return names, data


def extract_frames_from_video(
    video_path: str,
    max_frames: Optional[int] = None,
    skip_frames: int = 1,
) -> Tuple[List[np.ndarray], float]:
    """Extract frames from video file. Returns (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames = []
    frame_count = 0
    processed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % skip_frames != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        processed_count += 1

        if max_frames and processed_count >= max_frames:
            break

    cap.release()
    output_fps = fps / skip_frames
    return frames, output_fps


def draw_frame(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    track_ids: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    enable_masking: bool = True,
) -> np.ndarray:
    """Draw detections on a frame.

    Args:
        img: RGB image (H, W, 3).
        boxes: (N, 4) xyxy bounding boxes.
        labels: List of N class name strings.
        scores: (N,) confidence scores.
        track_ids: Optional (N,) integer track IDs.
        masks: Optional (N, H, W) boolean masks.
        enable_masking: Whether to draw mask overlays.

    Returns:
        Annotated image (RGB).
    """
    img = img.copy()
    h, w = img.shape[:2]

    # Draw mask overlays first (so boxes appear on top)
    if enable_masking and masks is not None and len(masks) > 0:
        for i in range(len(masks)):
            mask = masks[i]
            if mask.any():
                color_mask = np.zeros_like(img)
                if track_ids is not None:
                    color = get_track_color(track_ids[i])
                else:
                    color = (0, 255, 0)
                color_mask[mask] = color
                img = cv2.addWeighted(img, 1.0, color_mask, 0.3, 0)

    # Draw boxes and labels
    for i in range(len(boxes)):
        x1 = max(0, min(int(boxes[i][0]), w - 1))
        y1 = max(0, min(int(boxes[i][1]), h - 1))
        x2 = max(0, min(int(boxes[i][2]), w - 1))
        y2 = max(0, min(int(boxes[i][3]), h - 1))

        if track_ids is not None:
            color = get_track_color(track_ids[i])
            label_text = f"[{int(track_ids[i])}] {labels[i]}: {scores[i]:.2f}"
        else:
            color = (0, 255, 0)
            label_text = f"{labels[i]}: {scores[i]:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        cv2.rectangle(img, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - baseline - 2),
                    font, font_scale, (0, 0, 0), thickness)

    return img


def process_video_with_sam3(
    frames: List[np.ndarray],
    class_names: List[str],
    model_name: str = "sam3.pt",
    confidence_threshold: float = 0.3,
    enable_masking: bool = True,
    device: str = "cuda",
    fp16: bool = False,
    imgsz: int = 1024,
    video_fps: float = 30.0,
    output_video_path: Optional[str] = None,
) -> Tuple[Dict[str, list], int]:
    """Process frames with SAM 3 for detection + segmentation + tracking.

    Args:
        frames: List of RGB images.
        class_names: List of class names for text-prompted detection.
        model_name: SAM 3 model checkpoint name.
        confidence_threshold: Minimum confidence for detections.
        enable_masking: Whether to generate and draw masks.
        device: Device string ('cuda' or 'cpu').
        fp16: Whether to use half precision.
        imgsz: Input image size for SAM 3 inference (default: 1024).
        video_fps: Output video FPS.
        output_video_path: Path to write annotated video.

    Returns:
        Tuple of (all_detections dict, total_detection_count).
    """
    if not SAM3_AVAILABLE:
        raise ImportError("ultralytics is required. Install with: pip install ultralytics")

    # Load SAM 3 semantic predictor (supports text prompts)
    print(f"Loading SAM 3 model: {model_name}")
    overrides = dict(
        conf=confidence_threshold,
        task="segment",
        mode="predict",
        model=model_name,
        device=device,
        half=fp16,
        imgsz=imgsz,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    ori_height, ori_width = frames[0].shape[:2]

    # Initialize video writer
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (ori_width, ori_height))

    all_detections: Dict[str, list] = {}
    total_detections = 0

    # Build text prompts from class names
    text_prompts = class_names

    print(f"Processing {len(frames)} frames with SAM 3...")
    print(f"Text prompts: {text_prompts}")

    for frame_idx, frame in enumerate(frames):
        frame_start = time.time()

        if frame_idx % 50 == 0:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Processing frame {frame_idx}/{len(frames)}...", flush=True)

        frame_detections = []
        wrote_frame = False

        # Run SAM 3 inference with text prompts
        predictor.set_image(frame)
        results = predictor(text=text_prompts)

        if results and len(results) > 0:
            result = results[0]

            # Extract boxes, scores, class indices, masks, track IDs
            boxes_xyxy = None
            scores = None
            cls_indices = None
            masks = None
            track_ids = None

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                cls_indices = result.boxes.cls.cpu().numpy().astype(int)

                # Track IDs from SAM 3's native tracking
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

            if result.masks is not None and len(result.masks) > 0 and enable_masking:
                masks = result.masks.data.cpu().numpy().astype(bool)

            if boxes_xyxy is not None and len(boxes_xyxy) > 0:
                # Filter by confidence
                conf_mask = scores >= confidence_threshold
                boxes_xyxy = boxes_xyxy[conf_mask]
                scores = scores[conf_mask]
                cls_indices = cls_indices[conf_mask]
                if track_ids is not None:
                    track_ids = track_ids[conf_mask]
                if masks is not None:
                    masks = masks[conf_mask]

                # Build detection entries and label names
                label_names = []
                for i in range(len(boxes_xyxy)):
                    cls_idx = int(cls_indices[i])
                    if cls_idx < len(class_names):
                        label_name = class_names[cls_idx]
                    else:
                        label_name = f"Class_{cls_idx}"
                    label_names.append(label_name)

                    det_entry: Dict = {
                        "box_2d": [
                            int(boxes_xyxy[i][0]),
                            int(boxes_xyxy[i][1]),
                            int(boxes_xyxy[i][2]),
                            int(boxes_xyxy[i][3]),
                        ],
                        "label": label_name,
                        "confidence": round(float(scores[i]), 4),
                    }
                    if track_ids is not None:
                        det_entry["track_id"] = int(track_ids[i])
                    frame_detections.append(det_entry)

                total_detections += len(boxes_xyxy)

                # Draw annotated frame
                if out is not None:
                    annotated = draw_frame(
                        frame, boxes_xyxy, label_names, scores,
                        track_ids=track_ids, masks=masks,
                        enable_masking=enable_masking,
                    )
                    img_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                    wrote_frame = True

        # Write unannotated frame if no detections were drawn
        if out is not None and not wrote_frame:
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)

        all_detections[str(frame_idx)] = frame_detections

    if out is not None:
        out.release()
        print(f"Output video saved to: {output_video_path}")

    return all_detections, total_detections


def save_detections_json(
    all_detections: Dict[str, list],
    total_frames: int,
    class_names: List[str],
    video_path: str,
    video_fps: float,
    width: int,
    height: int,
    confidence_threshold: float,
    output_path: str,
) -> None:
    """Save detections in the same JSON schema as video_demo.py.

    This ensures compatibility with json_to_npz.py and downstream
    reconstruction/tracking scripts.
    """
    # Ensure all frames have entries
    for fidx in range(total_frames):
        key = str(fidx)
        if key not in all_detections:
            all_detections[key] = []

    output_data = {
        "video_path": video_path,
        "video_fps": video_fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "coordinate_format": "pixel_xyxy",
        "class_names": class_names,
        "detector": "sam3",
        "confidence_threshold": confidence_threshold,
        "detections": all_detections,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Detections saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM 3 video detection + segmentation + tracking"
    )
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Path to save output video")
    parser.add_argument("--classes", type=str, required=True,
                        help='Comma-separated class names (e.g., "pizza,plate,hand")')
    parser.add_argument("--confidence_threshold", type=float, default=0.3,
                        help="Minimum confidence score (default: 0.3)")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Process every Nth frame (1=all frames)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to process")
    parser.add_argument("--save_detections", type=str, default="auto",
                        help='Path for detections JSON, "auto" to derive from input, "none" to disable')
    parser.add_argument("--enable_masking", action="store_true", default=True,
                        help="Enable segmentation masking (default: enabled)")
    parser.add_argument("--disable_masking", dest="enable_masking", action="store_false",
                        help="Disable segmentation masking")
    parser.add_argument("--discovery_json", type=str, default=None,
                        help="Path to enhanced discovery result JSON file")
    parser.add_argument("--class_discovery_mode", choices=["simple", "attributed", "referring"],
                        default="attributed",
                        help="Which class level from discovery JSON (default: attributed)")
    # SAM 3 specific args
    parser.add_argument("--sam3_model", type=str, default="sam3.pt",
                        help="SAM 3 model checkpoint (default: sam3.pt)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use half precision inference")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Input image size for SAM 3 inference (default: 1024). "
                             "Higher values preserve more detail but use more VRAM.")

    args = parser.parse_args()
    print("Command Line Args:", args)

    # Extract frames
    print(f"Extracting frames from video: {args.input_video}")
    frames, video_fps = extract_frames_from_video(
        args.input_video,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )
    print(f"Extracted {len(frames)} frames (output FPS: {video_fps:.2f})")

    if len(frames) == 0:
        print("No frames extracted from video!")
        return

    ori_height, ori_width = frames[0].shape[:2]

    # Get class names from discovery JSON or CLI
    discovery_data = None
    if args.discovery_json and os.path.exists(args.discovery_json):
        class_names, discovery_data = load_discovered_classes(
            args.discovery_json, mode=args.class_discovery_mode
        )
        print(f"Loaded {len(class_names)} classes from discovery JSON ({args.class_discovery_mode} mode)")
        if discovery_data and "scene_context" in discovery_data:
            print(f"Scene context: {discovery_data['scene_context']}")
    else:
        class_names = [cls.strip() for cls in args.classes.split(",")]

    print(f"Using classes ({len(class_names)}): {class_names}")

    # Ensure output directory exists
    if args.output_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)

    # Process with SAM 3
    all_detections, total_detections = process_video_with_sam3(
        frames=frames,
        class_names=class_names,
        model_name=args.sam3_model,
        confidence_threshold=args.confidence_threshold,
        enable_masking=args.enable_masking,
        device=args.device,
        fp16=args.fp16,
        imgsz=args.imgsz,
        video_fps=video_fps,
        output_video_path=args.output_video,
    )

    print(f"\nAll frames processed!")
    print(f"Found {total_detections} detections total")

    # Save detections JSON
    save_path = args.save_detections
    if save_path and save_path.lower() == "none":
        save_path = None
    if save_path == "auto":
        base = os.path.splitext(os.path.basename(args.input_video))[0]
        save_dir = os.path.dirname(args.output_video) if args.output_video else "."
        save_path = os.path.join(save_dir, f"{base}_detections.json")
    if save_path:
        save_detections_json(
            all_detections=all_detections,
            total_frames=len(frames),
            class_names=class_names,
            video_path=args.input_video,
            video_fps=video_fps,
            width=ori_width,
            height=ori_height,
            confidence_threshold=args.confidence_threshold,
            output_path=save_path,
        )


if __name__ == "__main__":
    main()
