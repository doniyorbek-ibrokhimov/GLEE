"""SAM 3 Video Predictor mode: detection + segmentation + tracking with
temporal memory propagation and optional torch.compile.

Uses ``SAM3VideoSemanticPredictor`` which propagates memory across frames
(cached features for temporal tracking) instead of the per-frame
``SAM3SemanticPredictor`` used by ``video_demo_sam3.py``.

Outputs the same JSON schema as ``video_demo_sam3.py`` for downstream
compatibility (json_to_npz.py, run_reconstruction.py, etc.).

Usage:
    python video_demo_sam3_video.py \
        --input_video ../raw_videos/sample.mp4 \
        --output_video ../output_videos/sample_sam3v.mp4 \
        --classes "pizza,plate,hand" \
        --confidence_threshold 0.3 \
        --compile reduce-overhead
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Re-use visualization / IO helpers from the per-frame script
from video_demo_sam3 import (
    TRACK_COLORS,
    draw_frame,
    get_track_color,
    load_discovered_classes,
    save_detections_json,
)

# SAM 3 Video predictor import
try:
    from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
    from ultralytics import SAM
    SAM3_VIDEO_AVAILABLE = True
except ImportError:
    SAM3VideoSemanticPredictor = None
    SAM = None
    SAM3_VIDEO_AVAILABLE = False
    print(
        "WARNING: ultralytics not installed or SAM3VideoSemanticPredictor "
        "not available. Run: pip install ultralytics"
    )


def process_video_with_sam3_video(
    video_path: str,
    class_names: List[str],
    model_name: str = "sam3.pt",
    confidence_threshold: float = 0.3,
    min_box_area: int = 0,
    enable_masking: bool = True,
    device: str = "cuda",
    fp16: bool = False,
    imgsz: int = 1024,
    skip_frames: int = 1,
    max_frames: Optional[int] = None,
    compile_mode: str = "",
    output_video_path: Optional[str] = None,
    score_threshold_detection: Optional[float] = None,
    det_nms_thresh: Optional[float] = None,
    assoc_iou_thresh: Optional[float] = None,
    trk_assoc_iou_thresh: Optional[float] = None,
    init_trk_keep_alive: Optional[int] = None,
    max_trk_keep_alive: Optional[int] = None,
    fill_hole_area: Optional[int] = None,
    hotstart_delay: Optional[int] = None,
) -> Tuple[Dict[str, list], int, float, int, int]:
    """Process a video with SAM 3 Video Predictor (temporal memory).

    Unlike the per-frame variant this predictor maintains an internal
    ``inference_state`` that carries object embeddings across frames,
    giving more temporally consistent tracks.

    Args:
        video_path: Path to input video file.
        class_names: Class names for text-prompted detection.
        model_name: SAM 3 model checkpoint name.
        confidence_threshold: Minimum confidence for detections.
        min_box_area: Minimum bounding box area in pixels (0 = no filtering).
        enable_masking: Whether to generate and draw masks.
        device: Device string ('cuda' or 'cpu').
        fp16: Whether to use half precision.
        imgsz: Input image size for SAM 3 inference.
        skip_frames: Process every Nth frame (passed as vid_stride).
        max_frames: Maximum number of frames to process.
        compile_mode: torch.compile mode ("", "default", "reduce-overhead",
            "max-autotune-no-cudagraphs"). Empty string disables compilation.
        output_video_path: Path to write annotated video.
        score_threshold_detection: Override detection score threshold.
        det_nms_thresh: Detection NMS IoU threshold.
        assoc_iou_thresh: Association IoU threshold for tracking.
        trk_assoc_iou_thresh: Track association IoU threshold.
        init_trk_keep_alive: Initial track keep-alive frames.
        max_trk_keep_alive: Maximum track keep-alive frames.
        fill_hole_area: Fill hole area in masks.
        hotstart_delay: Hot-start delay in frames.

    Returns:
        Tuple of (all_detections, total_detection_count, video_fps,
        frame_width, frame_height).
    """
    if not SAM3_VIDEO_AVAILABLE:
        raise ImportError(
            "ultralytics with SAM3VideoSemanticPredictor is required. "
            "Install with: pip install ultralytics"
        )

    # --- Build overrides dict ---
    overrides: Dict = dict(
        conf=confidence_threshold,
        task="segment",
        mode="predict",
        model=model_name,
        device=device,
        half=fp16,
        imgsz=imgsz,
        vid_stride=skip_frames,
        verbose=False,
        save=False,
        show=False,
    )
    if compile_mode:
        overrides["compile"] = compile_mode

    # --- Tracking-specific constructor kwargs ---
    predictor_kwargs: Dict = {}
    if score_threshold_detection is not None:
        predictor_kwargs["score_threshold_detection"] = score_threshold_detection
    if det_nms_thresh is not None:
        predictor_kwargs["det_nms_thresh"] = det_nms_thresh
    if assoc_iou_thresh is not None:
        predictor_kwargs["assoc_iou_thresh"] = assoc_iou_thresh
    if trk_assoc_iou_thresh is not None:
        predictor_kwargs["trk_assoc_iou_thresh"] = trk_assoc_iou_thresh
    if init_trk_keep_alive is not None:
        predictor_kwargs["init_trk_keep_alive"] = init_trk_keep_alive
    if max_trk_keep_alive is not None:
        predictor_kwargs["max_trk_keep_alive"] = max_trk_keep_alive
    if fill_hole_area is not None:
        predictor_kwargs["fill_hole_area"] = fill_hole_area
    if hotstart_delay is not None:
        predictor_kwargs["hotstart_delay"] = hotstart_delay

    print(f"Initializing SAM 3 Video Predictor with model: {model_name}")
    if compile_mode:
        print(f"torch.compile mode: {compile_mode}")

    # Initialize predictor - it will load the model from overrides["model"]
    predictor = SAM3VideoSemanticPredictor(
        overrides=overrides,
        **predictor_kwargs
    )

    # The predictor auto-initializes the model from overrides on first inference
    # Get video FPS from the video file
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Adjust FPS for skip_frames
    output_fps = video_fps / skip_frames if skip_frames > 1 else video_fps

    # We'll determine frame dimensions from the first frame
    ori_width: int = 0
    ori_height: int = 0

    # Initialize video writer lazily after first frame
    out: Optional[cv2.VideoWriter] = None

    all_detections: Dict[str, list] = {}
    total_detections = 0
    frame_idx = 0

    print(f"Processing video with SAM 3 Video Predictor...")
    print(f"Text prompts: {class_names}")

    # Use predictor's stream_inference with text prompts
    # This returns Results objects with detections and masks
    batch_start_time = time.time()
    for result in predictor.stream_inference(source=video_path, text=class_names):
        frame_start = time.time()

        # Extract frame from result object
        # result.orig_img is the original frame (BGR format)
        orig_frame_bgr = result.orig_img
        orig_frame_rgb = cv2.cvtColor(orig_frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = orig_frame_rgb.shape[:2]

        if frame_idx == 0:
            ori_width = w
            ori_height = h
            if output_video_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    output_video_path, fourcc, output_fps, (ori_width, ori_height)
                )

        if frame_idx % 50 == 0:
            timestamp = time.strftime("%H:%M:%S")
            if frame_idx == 0:
                print(f"[{timestamp}] Processing frame {frame_idx}...", flush=True)
            else:
                elapsed = time.time() - batch_start_time
                print(f"[{timestamp}] Processing frame {frame_idx} (last 50 frames: {elapsed:.1f}s, {50/elapsed:.1f} fps)...", flush=True)
            batch_start_time = time.time()

        frame_detections: List[Dict] = []
        wrote_frame = False

        # --- Extract results from stream_inference ---
        # The result object already contains inference results with detections and masks
        if result is not None:
            boxes_xyxy = None
            scores = None
            cls_indices = None
            masks = None
            track_ids = None

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                cls_indices = result.boxes.cls.cpu().numpy().astype(int)

                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

            if (
                result.masks is not None
                and len(result.masks) > 0
                and enable_masking
            ):
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

                # Filter by minimum box area
                if min_box_area > 0 and len(boxes_xyxy) > 0:
                    box_widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
                    box_heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
                    box_areas = box_widths * box_heights
                    area_mask = box_areas >= min_box_area
                    boxes_xyxy = boxes_xyxy[area_mask]
                    scores = scores[area_mask]
                    cls_indices = cls_indices[area_mask]
                    if track_ids is not None:
                        track_ids = track_ids[area_mask]
                    if masks is not None:
                        masks = masks[area_mask]

                label_names: List[str] = []
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
                    # Track IDs removed from JSON output
                    frame_detections.append(det_entry)

                total_detections += len(boxes_xyxy)

                # Draw annotated frame (track IDs removed from labels)
                if out is not None:
                    annotated = draw_frame(
                        orig_frame_rgb,
                        boxes_xyxy,
                        label_names,
                        scores,
                        track_ids=None,  # Don't show track IDs in labels
                        masks=masks,
                        enable_masking=enable_masking,
                    )
                    img_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                    wrote_frame = True

        # Write unannotated frame if no detections were drawn
        if out is not None and not wrote_frame:
            out.write(orig_frame_bgr)

        all_detections[str(frame_idx)] = frame_detections
        frame_idx += 1

        if max_frames and frame_idx >= max_frames:
            break

    if out is not None:
        out.release()
        print(f"Output video saved to: {output_video_path}")

    return all_detections, total_detections, output_fps, ori_width, ori_height


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM 3 Video Predictor: detection + segmentation + "
        "tracking with temporal memory propagation"
    )
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output_video", type=str, default=None,
        help="Path to save output video",
    )
    parser.add_argument(
        "--classes", type=str, required=True,
        help='Comma-separated class names (e.g., "pizza,plate,hand")',
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.3,
        help="Minimum confidence score (default: 0.3)",
    )
    parser.add_argument(
        "--min_box_area", type=int, default=0,
        help="Minimum bounding box area in pixels (default: 0, no filtering). "
        "Example: 400 filters boxes smaller than 20x20 pixels",
    )
    parser.add_argument(
        "--skip_frames", type=int, default=1,
        help="Process every Nth frame (1=all frames)",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--save_detections", type=str, default="auto",
        help='Path for detections JSON, "auto" to derive from input, '
        '"none" to disable',
    )
    parser.add_argument(
        "--enable_masking", action="store_true", default=True,
        help="Enable segmentation masking (default: enabled)",
    )
    parser.add_argument(
        "--disable_masking", dest="enable_masking", action="store_false",
        help="Disable segmentation masking",
    )
    parser.add_argument(
        "--discovery_json", type=str, default=None,
        help="Path to enhanced discovery result JSON file",
    )
    parser.add_argument(
        "--class_discovery_mode",
        choices=["simple", "attributed", "referring"],
        default="attributed",
        help="Which class level from discovery JSON (default: attributed)",
    )
    # SAM 3 model args
    parser.add_argument(
        "--sam3_model", type=str, default="sam3.pt",
        help="SAM 3 model checkpoint (default: sam3.pt)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False,
        help="Use half precision inference",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1024,
        help="Input image size for SAM 3 inference (default: 1024)",
    )
    # torch.compile
    parser.add_argument(
        "--compile", type=str, default="",
        dest="compile_mode",
        help='torch.compile mode: "" (disabled), "default", '
        '"reduce-overhead", "max-autotune-no-cudagraphs"',
    )
    # Video predictor tracking parameters
    parser.add_argument(
        "--score_threshold_detection", type=float, default=None,
        help="Detection score threshold for video predictor",
    )
    parser.add_argument(
        "--det_nms_thresh", type=float, default=None,
        help="Detection NMS IoU threshold",
    )
    parser.add_argument(
        "--assoc_iou_thresh", type=float, default=None,
        help="Association IoU threshold for tracking",
    )
    parser.add_argument(
        "--trk_assoc_iou_thresh", type=float, default=None,
        help="Track association IoU threshold",
    )
    parser.add_argument(
        "--init_trk_keep_alive", type=int, default=None,
        help="Initial track keep-alive frames",
    )
    parser.add_argument(
        "--max_trk_keep_alive", type=int, default=None,
        help="Maximum track keep-alive frames",
    )
    parser.add_argument(
        "--fill_hole_area", type=int, default=None,
        help="Fill hole area in masks (pixels)",
    )
    parser.add_argument(
        "--hotstart_delay", type=int, default=None,
        help="Hot-start delay in frames",
    )

    args = parser.parse_args()
    print("Command Line Args:", args)

    # Get class names from discovery JSON or CLI
    discovery_data = None
    if args.discovery_json and os.path.exists(args.discovery_json):
        class_names, discovery_data = load_discovered_classes(
            args.discovery_json, mode=args.class_discovery_mode
        )
        print(
            f"Loaded {len(class_names)} classes from discovery JSON "
            f"({args.class_discovery_mode} mode)"
        )
        if discovery_data and "scene_context" in discovery_data:
            print(f"Scene context: {discovery_data['scene_context']}")
    else:
        class_names = [cls.strip() for cls in args.classes.split(",")]

    print(f"Using classes ({len(class_names)}): {class_names}")

    # Ensure output directory exists
    if args.output_video:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True
        )

    # Process with SAM 3 Video Predictor
    all_detections, total_detections, video_fps, ori_width, ori_height = (
        process_video_with_sam3_video(
            video_path=args.input_video,
            class_names=class_names,
            model_name=args.sam3_model,
            confidence_threshold=args.confidence_threshold,
            min_box_area=args.min_box_area,
            enable_masking=args.enable_masking,
            device=args.device,
            fp16=args.fp16,
            imgsz=args.imgsz,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            compile_mode=args.compile_mode,
            output_video_path=args.output_video,
            score_threshold_detection=args.score_threshold_detection,
            det_nms_thresh=args.det_nms_thresh,
            assoc_iou_thresh=args.assoc_iou_thresh,
            trk_assoc_iou_thresh=args.trk_assoc_iou_thresh,
            init_trk_keep_alive=args.init_trk_keep_alive,
            max_trk_keep_alive=args.max_trk_keep_alive,
            fill_hole_area=args.fill_hole_area,
            hotstart_delay=args.hotstart_delay,
        )
    )

    total_frames = len(all_detections)
    print(f"\nAll frames processed!")
    print(f"Found {total_detections} detections across {total_frames} frames")

    # Save detections JSON
    save_path = args.save_detections
    if save_path and save_path.lower() == "none":
        save_path = None
    if save_path == "auto":
        base = os.path.splitext(os.path.basename(args.input_video))[0]
        save_dir = (
            os.path.dirname(args.output_video) if args.output_video else "."
        )
        save_path = os.path.join(save_dir, f"{base}_detections.json")
    if save_path:
        save_detections_json(
            all_detections=all_detections,
            total_frames=total_frames,
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
