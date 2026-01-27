#!/bin/bash

# GLEE Video Processing Script - Open World Detection
# This script runs GLEE on a video file with open-world detection capability
# You can specify any custom classes to detect using the --classes parameter
#
# Performance: ~3.5 fps on RTX 3060 (about 10 minutes for 2100 frames)
# Test first with MAX_FRAMES=10 to verify settings before full run

# Activate virtual environment
source glee_venv/bin/activate

# Configuration
INPUT_VIDEO="/home/bob/Development/Ego3DT/raw_videos/short_sample.qt"
MODEL_PATH="/home/bob/Downloads/GLEE_Lite_joint.pth"
CONFIG="projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml"
OUTPUT_VIDEO="../raw_videos/room_short_output.mp4"

# Open-world detection parameters
# Specify any classes you want to detect (comma-separated, no spaces after commas)
# Examples:
#   "pizza,plate,hand" - detect pizza, plate, and hand
#   "car,person,bicycle" - detect vehicles and people
#   "laptop,book,phone" - detect electronics and objects
# CUSTOM_CLASSES="pizza,plate,hand,car,laptop,book,phone"
CUSTOM_CLASSES="object"

# Processing options
SKIP_FRAMES=1  # Process every Nth frame (1 = all frames)
MAX_FRAMES=0   # Limit to N frames (0 = process all frames)
BATCH_SIZE=1   # Number of frames to process per batch (reduced to avoid OOM)
CONFIDENCE_THRESHOLD=0.6  # Minimum confidence score for detections

# Build command
CMD="python3 video_demo.py \
    --input_video \"$INPUT_VIDEO\" \
    --output_video \"$OUTPUT_VIDEO\" \
    --model_path \"$MODEL_PATH\" \
    --config-file \"$CONFIG\" \
    --skip_frames $SKIP_FRAMES \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --classes \"$CUSTOM_CLASSES\""

# Add max_frames if not 0
if [ $MAX_FRAMES -gt 0 ]; then
    CMD="$CMD --max_frames $MAX_FRAMES"
fi

# Add num-gpus if needed (for detectron2)
CMD="$CMD --num-gpus 1"

# Print configuration
echo "=========================================="
echo "GLEE Open-World Video Detection"
echo "=========================================="
echo "Input Video: $INPUT_VIDEO"
echo "Output Video: $OUTPUT_VIDEO"
echo "Custom Classes: $CUSTOM_CLASSES"
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
echo "To use different classes, edit CUSTOM_CLASSES in this script"
echo "Example: CUSTOM_CLASSES=\"dog,cat,bird\""
