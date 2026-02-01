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
INPUT_VIDEO="../raw_videos/kitchen_clip_fixed.mp4"
MODEL_PATH="models/GLEE_Lite_joint.pth"
CONFIG="projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml"
OUTPUT_VIDEO="../output_videos/kitchen_clip_fixed_output_segmented.mp4"

# Dynamic class discovery via Gemini (set to false to use hardcoded classes below)
USE_DYNAMIC_CLASSES=true
# Set to true to ignore cached classes and re-run Gemini discovery
FORCE_REDISCOVER=false

# Hardcoded fallback classes (used when USE_DYNAMIC_CLASSES=false or discovery fails)
CUSTOM_CLASSES="headphone,lamp,monitor,watch,object,bottle,heater,hand,tablet,mouse,laptop,book,phone"

if [ "$USE_DYNAMIC_CLASSES" = true ]; then
    echo "Dynamic class discovery enabled. Running Gemini class discovery..."
    # Ensure google-genai and python-dotenv are installed
    pip install -q google-genai python-dotenv 2>/dev/null

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DISCOVER_ARGS="--video $INPUT_VIDEO"
    if [ "$FORCE_REDISCOVER" = true ]; then
        DISCOVER_ARGS="$DISCOVER_ARGS --no-cache"
    fi
    DISCOVERED_CLASSES=$(python3 "$SCRIPT_DIR/discover_classes.py" $DISCOVER_ARGS)

    if [ $? -eq 0 ] && [ -n "$DISCOVERED_CLASSES" ]; then
        CUSTOM_CLASSES="$DISCOVERED_CLASSES"
        echo "Discovered classes: $CUSTOM_CLASSES"
    else
        echo "WARNING: Class discovery failed. Falling back to hardcoded classes."
    fi
fi

# Processing options
SKIP_FRAMES=1  # Process every Nth frame (1 = all frames)
MAX_FRAMES=0   # Limit to N frames (0 = process all frames)
BATCH_SIZE=8   # Number of frames to process per batch (reduced to avoid OOM)
CONFIDENCE_THRESHOLD=0.3  # Minimum confidence score for detections
# SAM Masking: --disable_masking flag is used to reduce GPU memory usage
# Remove --disable_masking to enable segmentation masks (uses more GPU memory)

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
