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
INPUT_VIDEO="../raw_videos/egocentric_kitchen_part1.mp4"
MODEL_PATH="weights/GLEE_Pro_joint.pth"
CONFIG="projects/GLEE/configs/images/Pro/Stage2_joint_training_CLIPteacher_EVA02L.yaml"
OUTPUT_VIDEO="../output_videos/kitchen_part1_output.mp4"

# Dynamic class discovery via Gemini (set to false to use hardcoded classes below)
USE_DYNAMIC_CLASSES=True
# Set to true to ignore cached classes and re-run Gemini discovery
FORCE_REDISCOVER=False

# Enhanced discovery settings
# DISCOVERY_MODE controls how many Gemini API calls are made and what class info is generated.
# Each mode includes scene context detection first (if SCENE_AWARE=true), then:
#
#   simple     - 1 API call.  Basic 1-2 word class names via legacy path. Fastest.
#   base       - 1 API call.  Scene-aware base classes only (e.g. "watering can", "cup").
#                             Best for avoiding label switching between similar classes.
#   attributed - 2 API calls. Base classes + visual attributes (e.g. "white ceramic cup").
#                             Can cause label switching (e.g. "cup" vs "red cup" both ~0.3).
#   referring  - 2 API calls. Base classes + spatial/relational expressions
#                             (e.g. "the cup on the counter near the sink"). For GLEE grounding mode.
#   full       - 3 API calls. All of the above combined. Slowest but most comprehensive.
#
DISCOVERY_MODE="base"
SCENE_AWARE=true               # Enable scene context detection (adds 1 API call)
MAX_CLASSES=32                 # Limit base classes

# Hardcoded fallback classes (used when USE_DYNAMIC_CLASSES=false or discovery fails)
# CUSTOM_CLASSES="headphone,lamp,monitor,watch,object,bottle,heater,hand,tablet,mouse,laptop,book,phone"
CUSTOM_CLASSES="small glass,cup,plate,sponge,bottle,lemon,chocolate box,tray,bowl,coffee bean,almond,container,glass,tissue box"
# CUSTOM_CLASSES="object"


if [ "${USE_DYNAMIC_CLASSES,,}" = true ]; then
    echo "Dynamic class discovery enabled. Running Gemini class discovery..."
    # Ensure google-genai and python-dotenv are installed
    pip install -q google-genai python-dotenv 2>/dev/null

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DISCOVER_ARGS="--video $INPUT_VIDEO --mode $DISCOVERY_MODE --output-format simple --max-classes $MAX_CLASSES"
    if [ "${FORCE_REDISCOVER,,}" = true ]; then
        DISCOVER_ARGS="$DISCOVER_ARGS --no-cache"
    fi
    if [ "${SCENE_AWARE,,}" = false ]; then
        DISCOVER_ARGS="$DISCOVER_ARGS --no-scene-aware"
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
BATCH_SIZE=36   # Number of frames to process per batch (reduced to avoid OOM)
CONFIDENCE_THRESHOLD=0.3  # Minimum confidence score for detections
# SAM Masking: --disable_masking flag is used to reduce GPU memory usage
# Remove --disable_masking to enable segmentation masks (uses more GPU memory)

# SORT Tracking options
ENABLE_TRACKING=true  # Enable SORT-style IoU tracking for stable IDs
MAX_AGE=3             # Max frames a track survives without detection
MIN_HITS=3            # Min consecutive hits to confirm a track
IOU_THRESHOLD=0.3     # Min IoU to match detection to track

# Build command
CMD="python3 video_demo.py \
    --input_video \"$INPUT_VIDEO\" \
    --output_video \"$OUTPUT_VIDEO\" \
    --model_path \"$MODEL_PATH\" \
    --config-file \"$CONFIG\" \
    --skip_frames $SKIP_FRAMES \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --classes \"$CUSTOM_CLASSES\" \
    --disable_masking \
    --max_age $MAX_AGE \
    --min_hits $MIN_HITS \
    --iou_threshold $IOU_THRESHOLD"

# Add tracking flag
if [ "${ENABLE_TRACKING,,}" = true ]; then
    CMD="$CMD --enable_tracking"
else
    CMD="$CMD --disable_tracking"
fi

# Add max_frames if not 0
if [ $MAX_FRAMES -gt 0 ]; then
    CMD="$CMD --max_frames $MAX_FRAMES"
fi

# Add num-gpus if needed (for detectron2)
# Override DATASET_MAPPER_NAME to disable LSJ box postprocessing,
# since video_demo.py uses ResizeShortestEdge (not LSJ padding)
CMD="$CMD --num-gpus 1 INPUT.DATASET_MAPPER_NAME coco_instance_new_baselines"

# Print configuration
echo "=========================================="
echo "GLEE Open-World Video Detection"
echo "=========================================="
echo "Input Video: $INPUT_VIDEO"
echo "Output Video: $OUTPUT_VIDEO"
echo "Custom Classes: $CUSTOM_CLASSES"
echo "Batch Size: $BATCH_SIZE"
echo "Confidence Threshold: $CONFIDENCE_THRESHOLD"
if [ "${USE_DYNAMIC_CLASSES,,}" = true ]; then
    echo "Discovery Mode: $DISCOVERY_MODE"
    echo "Scene Aware: $SCENE_AWARE"
    echo "Max Classes: $MAX_CLASSES"
fi
if [ $MAX_FRAMES -gt 0 ]; then
    echo "Max Frames: $MAX_FRAMES"
fi
if [ $SKIP_FRAMES -gt 1 ]; then
    echo "Skip Frames: $SKIP_FRAMES"
fi
echo "Tracking: $ENABLE_TRACKING (max_age=$MAX_AGE, min_hits=$MIN_HITS, iou=$IOU_THRESHOLD)"
echo "=========================================="
echo ""

# Run the command
eval $CMD

echo ""
echo "Done! Output video saved to: $OUTPUT_VIDEO"
echo ""

croc send $OUTPUT_VIDEO
