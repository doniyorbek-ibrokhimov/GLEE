#!/bin/bash

# GLEE Video Processing Script
# This script runs GLEE on a video file with tracking enabled
#
# Performance: ~3.5 fps on RTX 3060 (about 10 minutes for 2100 frames)
# Test first with MAX_FRAMES=10 to verify settings before full run

# Activate virtual environment
source glee_venv/bin/activate

# Configuration
INPUT_VIDEO="/home/bob/Development/CafeAI-DirectionTracker/raw_videos/pizza_return.mp4"
MODEL_PATH="/home/bob/Downloads/GLEE_Lite_joint.pth"
CONFIG="projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml"
OUTPUT_VIDEO="../raw_videos/room_output.mp4"

# Detection parameters
# For open-world detection: specify custom classes (comma-separated, no spaces after commas)
# Examples:
#   CUSTOM_CLASSES="pizza,plate,hand" - detect pizza, plate, and hand
#   CUSTOM_CLASSES="car,person,bicycle" - detect vehicles and people
#   CUSTOM_CLASSES="" - use OVIS categories (default behavior)
CUSTOM_CLASSES="pizza,plate,hand,person"

# Legacy COCO categories (not used in current implementation, kept for reference)
# CATEGORIES="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"

# Processing options
SKIP_FRAMES=1  # Process every Nth frame (1 = all frames)
MAX_FRAMES=0   # Limit to N frames (0 = process all frames)
BATCH_SIZE=1   # Number of frames to process per batch (reduced to avoid OOM for open-world)
CONFIDENCE_THRESHOLD=0.3  # Minimum confidence score for detections

# Build command
CMD="python3 video_demo.py \
    --input_video \"$INPUT_VIDEO\" \
    --output_video \"$OUTPUT_VIDEO\" \
    --model_path \"$MODEL_PATH\" \
    --config-file \"$CONFIG\" \
    --skip_frames $SKIP_FRAMES \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD"

# Add custom classes for open-world detection if specified
if [ -n "$CUSTOM_CLASSES" ]; then
    CMD="$CMD --classes \"$CUSTOM_CLASSES\""
fi

# Add max_frames if not 0
if [ $MAX_FRAMES -gt 0 ]; then
    CMD="$CMD --max_frames $MAX_FRAMES"
fi

# Add num-gpus if needed (for detectron2)
CMD="$CMD --num-gpus 1"

# Print configuration
echo "=========================================="
echo "GLEE Video Detection"
echo "=========================================="
echo "Input Video: $INPUT_VIDEO"
echo "Output Video: $OUTPUT_VIDEO"
if [ -n "$CUSTOM_CLASSES" ]; then
    echo "Mode: Open-World Detection"
    echo "Custom Classes: $CUSTOM_CLASSES"
else
    echo "Mode: OVIS Categories (default)"
fi
echo "Batch Size: $BATCH_SIZE"
echo "Confidence Threshold: $CONFIDENCE_THRESHOLD"
if [ $MAX_FRAMES -gt 0 ]; then
    echo "Max Frames: $MAX_FRAMES"
fi
if [ $SKIP_FRAMES -gt 1 ]; then
    echo "Skip Frames: $SKIP_FRAMES"
fi
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "Done! Output video saved to: $OUTPUT_VIDEO"
echo ""
if [ -n "$CUSTOM_CLASSES" ]; then
    echo "Open-world detection completed with classes: $CUSTOM_CLASSES"
    echo "To use OVIS categories instead, set CUSTOM_CLASSES=\"\" in this script"
else
    echo "OVIS category detection completed"
    echo "To use open-world detection, set CUSTOM_CLASSES=\"class1,class2,class3\" in this script"
fi