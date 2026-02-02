import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import cv2
import argparse
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.projects.glee import add_glee_config, build_detection_train_loader, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


def load_discovered_classes(
    discovery_json_path: str,
    mode: str = "attributed",
) -> Tuple[List[str], Optional[Dict]]:
    """Extract classes from an enhanced discovery result JSON file.

    Args:
        discovery_json_path: Path to the discovery result JSON file.
        mode: Which class level to extract - 'simple' uses only base class
            names, 'attributed' adds attributed variants, 'referring' is
            reserved for future grounding mode support.

    Returns:
        Tuple of (batch_name_list, discovery_data). batch_name_list is the
        list of class name strings to pass to GLEE. discovery_data is the
        full parsed JSON dict (or None if loading failed).
    """
    with open(discovery_json_path, "r") as f:
        data = json.load(f)

    # If the file has a batch_name_list, use it directly (already deduplicated)
    if "batch_name_list" in data:
        names = data["batch_name_list"]
    elif "classes" in data:
        # Build from classes structure
        classes_section = data["classes"]
        if isinstance(classes_section, dict):
            names = [c["name"] for c in classes_section.get("base", [])]
            if mode in ("attributed", "full"):
                for attr in classes_section.get("attributed", []):
                    name = attr.get("name", "")
                    if name and name not in names:
                        names.append(name)
        elif isinstance(classes_section, list):
            # Legacy simple format: classes is a list of strings
            names = [str(c) for c in classes_section]
        else:
            names = []
    else:
        names = []

    if mode == "simple":
        # Only return base class names
        base_names = []
        classes_section = data.get("classes", {})
        if isinstance(classes_section, dict):
            base_names = [c["name"] for c in classes_section.get("base", [])]
        if base_names:
            names = base_names

    return names, data


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_glee_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  
    default_setup(cfg, args)
    return cfg

def extract_frames_from_video(video_path, max_frames=None, skip_frames=1):
    """Extract frames from video file. Returns (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    frames = []
    frame_count = 0
    processed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames if needed
        if frame_count % skip_frames != 0:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        processed_count += 1

        if max_frames and processed_count >= max_frames:
            break

    cap.release()
    # Adjust FPS when skipping frames to preserve video duration
    output_fps = fps / skip_frames
    return frames, output_fps

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    
    # Load model checkpoint
    if hasattr(args, 'model_path') and args.model_path:
        DetectionCheckpointer(model).load(args.model_path)
    else:
        DetectionCheckpointer(model).load('GLEE_Plus_joint.pth')

    # Initialize SAM model (skip if masking is disabled to save VRAM)
    if getattr(args, 'enable_masking', True):
        sam_checkpoint = getattr(args, 'sam_checkpoint', None)
        if sam_checkpoint is None:
            # Try alternative paths (relative to script location and project root)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            alt_paths = [
                os.path.join(project_root, 'segment-anything', 'checkpoints', 'sam_vit_h_4b8939.pth'),
                'segment-anything/checkpoints/sam_vit_h_4b8939.pth',
                '../segment-anything/checkpoints/sam_vit_h_4b8939.pth',
                os.path.join(script_dir, '..', 'segment-anything', 'checkpoints', 'sam_vit_h_4b8939.pth'),
                'checkpoints/sam_vit_h_4b8939.pth',
            ]
            for alt_path in alt_paths:
                abs_path = os.path.abspath(alt_path)
                if os.path.exists(abs_path):
                    sam_checkpoint = abs_path
                    break
            else:
                raise FileNotFoundError(f"SAM checkpoint not found. Tried: {alt_paths}")
        elif not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found at: {sam_checkpoint}")

        sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam_model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        sam_predictor = SamPredictor(sam_model)
        print(f"SAM model loaded successfully from {sam_checkpoint}")
    else:
        sam_predictor = None
        print("SAM masking disabled - skipping SAM model loading to save VRAM")

    # Determine input source
    if hasattr(args, 'input_video') and args.input_video:
        # Extract frames from video
        print(f"Extracting frames from video: {args.input_video}")
        frames, video_fps = extract_frames_from_video(
            args.input_video,
            max_frames=getattr(args, 'max_frames', None),
            skip_frames=getattr(args, 'skip_frames', 1)
        )
        print(f"Extracted {len(frames)} frames (source FPS: {video_fps:.2f})")
        
        if len(frames) == 0:
            print("No frames extracted from video!")
            return
        
        ori_height, ori_width = frames[0].shape[:2]
        file_names = [f"frame_{i}" for i in range(len(frames))]
    else:
        # Use image directory
        img_dir = './CAM_FRONT_LEFT'
        if not os.path.exists(img_dir):
            print(f"Image directory not found: {img_dir}")
            return

        frames = []
        file_names = []
        video_fps = 30.0  # default for image sequences
        for frame_file in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, frame_file)
            file_names.append(img_path)
            image = utils.read_image(img_path, format='RGB')
            frames.append(image)

        if len(frames) == 0:
            print("No images found in directory!")
            return

        ori_height, ori_width = frames[0].shape[:2]

    # Get custom classes: from discovery JSON or comma-separated CLI argument
    discovery_data = None
    if getattr(args, 'discovery_json', None) and os.path.exists(args.discovery_json):
        discovery_mode = getattr(args, 'class_discovery_mode', 'attributed')
        batch_name_list, discovery_data = load_discovered_classes(
            args.discovery_json, mode=discovery_mode
        )
        print(f"Loaded {len(batch_name_list)} classes from discovery JSON ({discovery_mode} mode)")
        if discovery_data and 'scene_context' in discovery_data:
            print(f"Scene context: {discovery_data['scene_context']}")
    else:
        # Parse comma-separated class names
        custom_classes = [cls.strip() for cls in args.classes.split(',')]
        batch_name_list = custom_classes

    print(f"Using classes ({len(batch_name_list)}): {batch_name_list}")
    task = 'coco_clip'  # Use coco_clip task for open-world detection
    
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    sample_style = "choice"
    aug_list = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    augumentations = T.AugmentationList(aug_list)

    # Process frames in batches to avoid OOM
    # Note: For open-world detection, use batch_size=1 to avoid CUDA OOM errors
    batch_size = getattr(args, 'batch_size', 10)  # Process N frames at a time
    model.eval()
    
    # Initialize video writer early for incremental writing (memory efficient)
    output_video_path = getattr(args, 'output_video', None)
    out = None
    if output_video_path:
        print(f"Initializing output video: {output_video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (ori_width, ori_height))
    
    confidence_threshold = getattr(args, 'confidence_threshold', 0.5)
    total_detections = 0
    all_detections = {}  # frame_idx (str) -> list of detection dicts
    
    print(f"Processing {len(frames)} frames in batches of {batch_size}...")
    
    for batch_start in range(0, len(frames), batch_size):
        batch_end = min(batch_start + batch_size, len(frames))
        batch_frames = frames[batch_start:batch_end]
        batch_file_names = file_names[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(frames)-1)//batch_size + 1} (frames {batch_start}-{batch_end-1})...")
        
        img_list = []
        for image in batch_frames:
            aug_input = T.AugInput(image)
            transforms = augumentations(aug_input)
            image = aug_input.image
            image_shape = image.shape[:2]
            img_list.append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
        
        inputs = []
        for i, img_tensor in enumerate(img_list):
            input_dict = {
                'height': ori_height,
                'width': ori_width,
                'image': [img_tensor],  # wrap in list for preprocess_video compatibility
                'task': task,  # Use 'coco_clip' for open-world detection
                'file_names': batch_file_names[i] if i < len(batch_file_names) else f"frame_{batch_start + i}",
                'prompt': None
            }
            # Add batch_name_list for open-world detection
            if batch_name_list is not None:
                input_dict['batch_name_list'] = batch_name_list
            inputs.append(input_dict)

        with torch.no_grad():
            outputs = model(inputs)
            
            # Process outputs immediately and write to video (incremental processing)
            if isinstance(outputs, list):
                # Process each frame's instances in this batch
                for frame_in_batch, output_dict in enumerate(outputs):
                    frame_idx = batch_start + frame_in_batch
                    if frame_idx >= len(frames):
                        break
                    
                    # Get the original frame
                    img = batch_frames[frame_in_batch].copy()
                    frame_detections = []
                    
                    # Extract detections for this frame
                    if 'instances' in output_dict:
                        instances = output_dict['instances']
                        if hasattr(instances, 'scores') and len(instances) > 0:
                            # Move to CPU immediately to free GPU memory
                            scores = instances.scores.cpu().numpy() if isinstance(instances.scores, torch.Tensor) else instances.scores
                            labels = instances.pred_classes.cpu().numpy() if isinstance(instances.pred_classes, torch.Tensor) else instances.pred_classes
                            
                            # Extract boxes (keep in xyxy format for SAM)
                            if hasattr(instances.pred_boxes, 'tensor'):
                                boxes_xyxy = instances.pred_boxes.tensor.cpu().numpy()  # Shape: (N, 4)
                            else:
                                boxes_xyxy = instances.pred_boxes.cpu().numpy()
                            
                            # Convert to xywh for drawing (existing logic)
                            if len(boxes_xyxy.shape) == 2 and boxes_xyxy.shape[1] == 4:
                                boxes_xywh = boxes_xyxy.copy()
                                boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # width
                                boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # height
                            else:
                                boxes_xywh = boxes_xyxy

                            # Generate SAM masks for detected objects (if enabled)
                            if getattr(args, 'enable_masking', True):
                                masks = None
                                mask_scores = None
                                if len(boxes_xyxy) > 0:
                                    # Filter boxes by confidence before SAM (efficiency)
                                    valid_indices = scores >= confidence_threshold
                                    valid_boxes = boxes_xyxy[valid_indices].copy()

                                    if len(valid_boxes) > 0:
                                        # Clip boxes to image bounds before SAM
                                        h, w = img.shape[:2]
                                        valid_boxes[:, 0] = np.clip(valid_boxes[:, 0], 0, w - 1)  # x1
                                        valid_boxes[:, 1] = np.clip(valid_boxes[:, 1], 0, h - 1)  # y1
                                        valid_boxes[:, 2] = np.clip(valid_boxes[:, 2], 0, w - 1)  # x2
                                        valid_boxes[:, 3] = np.clip(valid_boxes[:, 3], 0, h - 1)  # y2

                                        # Set image once per frame
                                        sam_predictor.set_image(img, image_format="RGB")

                                        # Process each box individually (SAM's predict expects single box)
                                        masks_list = []
                                        mask_scores_list = []
                                        for box in valid_boxes:
                                            mask, mask_score, _ = sam_predictor.predict(
                                                box=box,
                                                multimask_output=False  # Single best mask per box
                                            )
                                            masks_list.append(mask[0])  # Extract single mask from output
                                            mask_scores_list.append(mask_score[0])  # Extract single score

                                        # Convert to numpy arrays
                                        masks_all = np.array(masks_list)
                                        mask_scores_all = np.array(mask_scores_list)

                                        # Create full mask array aligned with all detections
                                        masks = np.zeros((len(scores), img.shape[0], img.shape[1]), dtype=bool)
                                        mask_scores = np.zeros(len(scores))
                                        masks[valid_indices] = masks_all
                                        mask_scores[valid_indices] = mask_scores_all
                                    else:
                                        masks = np.zeros((len(scores), img.shape[0], img.shape[1]), dtype=bool)
                                        mask_scores = np.zeros(len(scores))
                                else:
                                    masks = np.array([])
                                    mask_scores = np.array([])

                                # Draw mask overlays first (so boxes appear on top)
                                if masks is not None and len(masks) > 0:
                                    for i in range(len(scores)):
                                        if scores[i] >= confidence_threshold and i < len(masks):
                                            mask = masks[i]
                                            if mask.any():  # Check if mask has any True pixels
                                                # Create colored overlay
                                                color_mask = np.zeros_like(img)
                                                # Use different colors for different objects (optional)
                                                color = (0, 255, 0)  # Green default
                                                color_mask[mask] = color
                                                # Blend with original image (30% opacity)
                                                img = cv2.addWeighted(img, 1.0, color_mask, 0.3, 0)

                            # Draw detections on this frame
                            num_instances = len(scores)
                            for i in range(num_instances):
                                score = float(scores[i])
                                label = int(labels[i])
                                
                                # Filter by confidence threshold
                                if score < confidence_threshold:
                                    continue
                                
                                total_detections += 1

                                # Resolve label name
                                if batch_name_list is not None and label < len(batch_name_list):
                                    label_name = batch_name_list[label]
                                else:
                                    label_name = f"Class_{label}"

                                # Collect detection for JSON output
                                if len(boxes_xyxy.shape) == 2:
                                    det_box = boxes_xyxy[i]
                                    frame_detections.append({
                                        "box_2d": [int(det_box[0]), int(det_box[1]),
                                                   int(det_box[2]), int(det_box[3])],
                                        "label": label_name,
                                        "confidence": round(float(score), 4),
                                    })

                                # Get box for this instance (use xywh format)
                                if len(boxes_xywh.shape) == 2:
                                    box = boxes_xywh[i]
                                else:
                                    box = boxes_xywh[i] if i < len(boxes_xywh) else None
                                
                                # Draw bounding box if available
                                if box is not None and len(box) == 4:
                                    # box is in xywh format (x, y, width, height), convert to xyxy
                                    x, y, w, h = box
                                    x1, y1 = int(x), int(y)
                                    x2, y2 = int(x + w), int(y + h)
                                    
                                    # Clip to image bounds
                                    x1 = max(0, min(x1, ori_width - 1))
                                    y1 = max(0, min(y1, ori_height - 1))
                                    x2 = max(0, min(x2, ori_width - 1))
                                    y2 = max(0, min(y2, ori_height - 1))
                                    
                                    # Draw box
                                    color = (0, 255, 0)  # Green
                                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Draw label
                                    label_text = f"{label_name}: {score:.2f}"
                                    
                                    # Get text size
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.6
                                    thickness = 2
                                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                                    
                                    # Draw label background
                                    cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                                                (x1 + text_width, y1), color, -1)
                                    
                                    # Draw label text
                                    cv2.putText(img, label_text, (x1, y1 - baseline - 2), 
                                               font, font_scale, (0, 0, 0), thickness)
                    
                    all_detections[str(frame_idx)] = frame_detections

                    # Write frame to video immediately
                    if out is not None:
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out.write(img_bgr)
            
            # Free memory immediately after processing batch
            del outputs
            torch.cuda.empty_cache()
    
    print("All batches processed!")
    print(f"Found {total_detections} detections total")
    
    # Close video writer if it was opened
    if out is not None:
        out.release()
        print(f"Output video saved to: {output_video_path}")

    # Save detections JSON if requested
    save_detections = getattr(args, 'save_detections', None)
    if save_detections and save_detections.lower() == 'none':
        save_detections = None
    if save_detections == 'auto':
        input_video = getattr(args, 'input_video', None)
        if input_video:
            base = os.path.splitext(os.path.basename(input_video))[0]
        else:
            base = 'detections'
        save_dir = os.path.dirname(output_video_path) if output_video_path else '.'
        save_detections = os.path.join(save_dir, f"{base}_detections.json")
    if save_detections:
        # Ensure all frames have entries (even those with no detections)
        for fidx in range(len(frames)):
            key = str(fidx)
            if key not in all_detections:
                all_detections[key] = []

        output_data = {
            "video_path": getattr(args, 'input_video', None) or "",
            "video_fps": video_fps,
            "width": ori_width,
            "height": ori_height,
            "total_frames": len(frames),
            "coordinate_format": "pixel_xyxy",
            "class_names": batch_name_list,
            "detector": "glee",
            "confidence_threshold": confidence_threshold,
            "detections": all_detections,
        }
        with open(save_detections, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Detections saved to: {save_detections}")




if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--input_video', type=str, default=None, help='path to input video file')
    parser.add_argument('--output_video', type=str, default=None, help='path to save output video')
    parser.add_argument('--output_dir', type=str, default='./output_frames', help='directory to save output frames')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--max_frames', type=int, default=None, help='maximum number of frames to process')
    parser.add_argument('--skip_frames', type=int, default=1, help='process every Nth frame (1=all frames)')
    parser.add_argument('--batch_size', type=int, default=10, help='number of frames to process per batch (default: 10)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='minimum confidence score threshold for detections (default: 0.5)')
    parser.add_argument('--classes', type=str, required=True, help='comma-separated list of custom class names for open-world detection (e.g., "pizza,plate,hand,car,person")')
    parser.add_argument('--sam_checkpoint', type=str, default=None, help='path to SAM checkpoint (default: auto-detect)')
    parser.add_argument('--enable_masking', action='store_true', default=True, help='enable SAM segmentation masking (default: enabled)')
    parser.add_argument('--disable_masking', dest='enable_masking', action='store_false', help='disable SAM segmentation masking to reduce GPU memory usage')
    parser.add_argument('--discovery_json', type=str, default=None, help='path to enhanced discovery result JSON file (from discover_classes.py --output-format enhanced)')
    parser.add_argument('--class_discovery_mode', choices=['simple', 'attributed', 'referring'], default='attributed', help='which class level to use from discovery JSON (default: attributed)')
    parser.add_argument('--save_detections', type=str, default='auto', help='path to save detections JSON file, or "auto" to derive from input video name (default: auto). Use "none" to disable.')

    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
