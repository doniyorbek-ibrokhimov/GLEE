import os
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
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
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
    return frames

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    
    # Load model checkpoint
    if hasattr(args, 'model_path') and args.model_path:
        DetectionCheckpointer(model).load(args.model_path)
    else:
        DetectionCheckpointer(model).load('GLEE_Plus_joint.pth')

    # Initialize SAM model
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

    # Determine input source
    if hasattr(args, 'input_video') and args.input_video:
        # Extract frames from video
        print(f"Extracting frames from video: {args.input_video}")
        frames = extract_frames_from_video(
            args.input_video, 
            max_frames=getattr(args, 'max_frames', None),
            skip_frames=getattr(args, 'skip_frames', 1)
        )
        print(f"Extracted {len(frames)} frames")
        
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
        for frame_file in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, frame_file)
            file_names.append(img_path)
            image = utils.read_image(img_path, format='RGB')
            frames.append(image)
        
        if len(frames) == 0:
            print("No images found in directory!")
            return
        
        ori_height, ori_width = frames[0].shape[:2]

    # Get custom classes from command line (required for open-world mode)
    # Parse comma-separated class names
    custom_classes = [cls.strip() for cls in args.classes.split(',')]
    print(f"Using custom classes: {custom_classes}")
    batch_name_list = custom_classes
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
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (ori_width, ori_height))
    
    confidence_threshold = getattr(args, 'confidence_threshold', 0.5)
    total_detections = 0
    
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
        
        inputs = [{
            'height': ori_height,
            'width': ori_width,
            'image': img_list,
            'task': task,  # Use 'coco_clip' for open-world detection
            'file_names': batch_file_names,
            'prompt': None
        }]
        
        # Add batch_name_list for open-world detection
        if batch_name_list is not None:
            inputs[0]['batch_name_list'] = batch_name_list

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
                            
                            # Generate SAM masks for detected objects
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
                                    if batch_name_list is not None and label < len(batch_name_list):
                                        label_name = batch_name_list[label]
                                    else:
                                        label_name = f"Class_{label}"
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
    
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
