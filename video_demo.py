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

# Actual OVIS dataset categories (25 categories)
# Model outputs label indices 0-24 corresponding to these categories
ACTUAL_OVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "Bird"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "Cat"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "Dog"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "Horse"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "Sheep"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "Cow"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "Elephant"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "Bear"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "Zebra"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "Giraffe"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "Poultry"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "Giant_panda"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "Lizard"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "Parrot"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "Monkey"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "Rabbit"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "Tiger"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "Fish"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "Turtle"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "Bicycle"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "Motorcycle"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "Airplane"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "Boat"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "Vehical"},
]

# Custom categories for display (user-defined)
OVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "pizza"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "plate"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "hand"},
]

# Create mapping from label index (0-24) to category name
# Model outputs indices 0-24 for OVIS task (25 categories)
# OVIS categories have ids 1-25, so index = id - 1
OVIS_LABEL_TO_NAME = {}
for cat in ACTUAL_OVIS_CATEGORIES:
    label_idx = cat['id'] - 1  # Convert id (1-25) to index (0-24)
    OVIS_LABEL_TO_NAME[label_idx] = cat['name']

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

    # Get custom classes from command line or use default
    if hasattr(args, 'classes') and args.classes:
        # Parse comma-separated class names
        custom_classes = [cls.strip() for cls in args.classes.split(',')]
        print(f"Using custom classes: {custom_classes}")
        batch_name_list = custom_classes
        task = 'coco_clip'  # Use coco_clip task for open-world detection
    else:
        # Use OVIS categories as before
        prompt = [cat['name'] for cat in OVIS_CATEGORIES]
        batch_name_list = None  # Will use default OVIS categories
        task = 'ovis'
    
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    sample_style = "choice"
    aug_list = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    augumentations = T.AugmentationList(aug_list)

    # Process frames in batches to avoid OOM
    # Note: For open-world detection, use batch_size=1 to avoid CUDA OOM errors
    batch_size = getattr(args, 'batch_size', 10)  # Process N frames at a time
    model.eval()
    
    all_outputs = []
    all_masks = []
    
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
            'task': task,  # Use 'coco_clip' for open-world or 'ovis' for OVIS categories
            'file_names': batch_file_names,
            'prompt': None
        }]
        
        # Add batch_name_list for open-world detection
        if batch_name_list is not None:
            inputs[0]['batch_name_list'] = batch_name_list

        with torch.no_grad():
            outputs = model(inputs)
            # Convert outputs to expected format
            # For image tasks (coco_clip), model returns list of dicts with "instances" key
            # Each dict corresponds to one frame in the batch
            converted_outputs = {
                'pred_scores': [],
                'pred_labels': [],
                'pred_masks': [],
                'pred_boxes': []
            }
            
            if isinstance(outputs, list):
                # Process each frame's instances
                for frame_idx, output_dict in enumerate(outputs):
                    if 'instances' in output_dict:
                        instances = output_dict['instances']
                        # Extract data from Instances object and move to CPU immediately
                        if hasattr(instances, 'scores') and len(instances) > 0:
                            scores = instances.scores.cpu().numpy() if isinstance(instances.scores, torch.Tensor) else instances.scores
                            labels = instances.pred_classes.cpu().numpy() if isinstance(instances.pred_classes, torch.Tensor) else instances.pred_classes
                            masks = instances.pred_masks.cpu().numpy() if isinstance(instances.pred_masks, torch.Tensor) else instances.pred_masks
                            
                            # Extract boxes
                            if hasattr(instances.pred_boxes, 'tensor'):
                                boxes = instances.pred_boxes.tensor.cpu().numpy()
                            else:
                                boxes = instances.pred_boxes.cpu().numpy()
                            
                            # Convert boxes from xyxy to xywh format
                            if len(boxes.shape) == 2 and boxes.shape[1] == 4:
                                boxes_xywh = boxes.copy()
                                boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
                                boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
                                boxes = boxes_xywh
                            
                            # For image tasks, each instance is independent per frame
                            # We need to organize by instance across frames for tracking
                            # But for now, just store per frame and process later
                            num_instances = len(scores)
                            for i in range(num_instances):
                                converted_outputs['pred_scores'].append(float(scores[i]))
                                converted_outputs['pred_labels'].append(int(labels[i]))
                                # Masks: [H, W] -> [1, H, W] to match expected format
                                if len(masks.shape) == 2:
                                    converted_outputs['pred_masks'].append(masks[i:i+1])
                                else:
                                    converted_outputs['pred_masks'].append(masks[i])
                                # Boxes: [4] -> [1, 4] to match expected format  
                                if len(boxes.shape) == 1:
                                    converted_outputs['pred_boxes'].append(boxes[i:i+1] if i < len(boxes) else boxes)
                                else:
                                    converted_outputs['pred_boxes'].append(boxes[i])
            
            all_outputs.append(converted_outputs)
        
        # Clear GPU cache after each batch
        del outputs
        torch.cuda.empty_cache()
    
    print("All batches processed!")
    
    # Process outputs from each batch
    # Each batch output is a dict with:
    # - pred_scores: list of floats (one per instance)
    # - pred_labels: list of ints (one per instance)
    # - pred_masks: list of tensors, each [num_frames_in_batch, H, W]
    # - pred_boxes: list of tensors, each [num_frames_in_batch, 4] in xywh format
    
    if len(all_outputs) == 0:
        print("No outputs generated!")
        return
    
    all_frame_detections = {}  # frame_idx -> list of detections
    
    frame_offset = 0
    for batch_idx, batch_outputs in enumerate(all_outputs):
        if not isinstance(batch_outputs, dict):
            frame_offset += batch_size
            continue
            
        batch_scores = batch_outputs.get('pred_scores', [])
        batch_labels = batch_outputs.get('pred_labels', [])
        batch_masks = batch_outputs.get('pred_masks', [])
        batch_boxes = batch_outputs.get('pred_boxes', [])
        
        if len(batch_scores) == 0:
            frame_offset += batch_size
            continue
        
        # Get number of frames in this batch from first mask
        num_frames_in_batch = batch_size
        if len(batch_masks) > 0 and isinstance(batch_masks[0], torch.Tensor):
            num_frames_in_batch = batch_masks[0].shape[0]
        
        # Process each instance
        for inst_idx in range(len(batch_scores)):
            score = batch_scores[inst_idx]
            label = batch_labels[inst_idx]
            
            # Filter by score threshold
            confidence_threshold = getattr(args, 'confidence_threshold', 0.5)
            if score < confidence_threshold:
                continue
            
            # Get mask and box for this instance
            mask = None
            box = None
            
            if inst_idx < len(batch_masks):
                mask_tensor = batch_masks[inst_idx]
                if isinstance(mask_tensor, torch.Tensor):
                    mask = mask_tensor.cpu().numpy()
                else:
                    mask = mask_tensor
                    
            if inst_idx < len(batch_boxes):
                box_tensor = batch_boxes[inst_idx]
                if isinstance(box_tensor, torch.Tensor):
                    box = box_tensor.cpu().numpy()
                else:
                    box = box_tensor
            
            # Process each frame in this batch
            if mask is not None and len(mask.shape) == 3:  # [num_frames, H, W]
                for frame_in_batch in range(min(num_frames_in_batch, mask.shape[0])):
                    frame_idx = frame_offset + frame_in_batch
                    if frame_idx >= len(frames):
                        break
                    
                    frame_mask = mask[frame_in_batch]
                    frame_box = None
                    if box is not None:
                        if len(box.shape) == 2:  # [num_frames, 4]
                            frame_box = box[frame_in_batch]
                        elif len(box.shape) == 1 and len(box) == 4:  # [4]
                            frame_box = box
                    
                    if frame_idx not in all_frame_detections:
                        all_frame_detections[frame_idx] = []
                    
                    all_frame_detections[frame_idx].append({
                        'score': score,
                        'label': label,
                        'mask': frame_mask,
                        'box': frame_box
                    })
        
        frame_offset += num_frames_in_batch
    
    total_detections = sum(len(dets) for dets in all_frame_detections.values())
    print(f"Found {total_detections} detections across {len(all_frame_detections)} frames")
        
    # Save output frames
    output_dir = getattr(args, 'output_dir', './output_frames')
    os.makedirs(output_dir, exist_ok=True)
    
    # saved_count = 0
    # for frame_idx in all_frame_detections:
    #     if len(all_frame_detections[frame_idx]) > 0:
    #         img = frames[frame_idx].copy()
    #         # Apply mask overlay
    #         for det in all_frame_detections[frame_idx]:
    #             mask = det['mask']
    #             if mask.sum() > 0:
    #                 img[~mask] = img[~mask] * 0.5  # Dim non-masked areas
    #         output_path = os.path.join(output_dir, f'{frame_idx}.png')
    #         Image.fromarray(img).save(output_path)
    #         saved_count += 1
    
    # print(f"Saved {saved_count} masked frames to {output_dir}")
    
    # Create output video if requested
    if hasattr(args, 'output_video') and args.output_video:
        print(f"Creating output video: {args.output_video}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_video, fourcc, 30.0, (ori_width, ori_height))
        
        # Note: OVIS_CATEGORIES might not match the actual label indices from the model
        # The model uses its own label encoding for the OVIS task
        
        for frame_idx in range(len(frames)):
            img = frames[frame_idx].copy()
            
            # Draw detections for this frame
            if frame_idx in all_frame_detections:
                for det in all_frame_detections[frame_idx]:
                    score = det['score']
                    label = det['label']
                    mask = det['mask']
                    box = det['box']
                    
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
                        # Map label index to category name
                        if batch_name_list is not None and isinstance(label, (int, np.integer)) and label < len(batch_name_list):
                            # For open-world detection, use batch_name_list
                            label_name = batch_name_list[int(label)]
                        elif isinstance(label, (int, np.integer)) and label in OVIS_LABEL_TO_NAME:
                            # For OVIS task, use OVIS categories
                            label_name = OVIS_LABEL_TO_NAME[int(label)]
                        else:
                            label_name = f"Class_{label}"  # Fallback if label not in mapping
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
                    
                    # Draw mask overlay (optional, can be enabled)
                    # if mask.sum() > 0:
                    #     mask_overlay = img.copy()
                    #     mask_overlay[mask > 0] = [0, 255, 0]  # Green overlay
                    #     img = cv2.addWeighted(img, 0.7, mask_overlay, 0.3, 0)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)
        
        out.release()
        print(f"Output video saved to: {args.output_video}")




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
    parser.add_argument('--classes', type=str, default=None, help='comma-separated list of custom class names for open-world detection (e.g., "pizza,plate,hand,car,person"). If not provided, uses OVIS categories.')
    
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
