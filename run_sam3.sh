#!/bin/bash

# SAM 3 Video Processing Script - Open World Detection + Segmentation + Tracking
# Parallel pipeline to run_video_openworld.sh using SAM 3 instead of GLEE+SAM+SORT
#
# SAM 3 provides detection, segmentation, and tracking in a single model.
# Output JSON is compatible with json_to_npz.py and downstream reconstruction scripts.
#
# Setup:
#   bash setup_sam3_env.sh   # One-time environment setup
#
# Usage:
#   bash run_video_openworld_sam3.sh

# Activate SAM 3 virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/sam3_venv" ]; then
    source "$SCRIPT_DIR/sam3_venv/bin/activate"
else
    echo "ERROR: sam3_venv not found. Run: bash setup_sam3_env.sh"
    exit 1
fi

# Configuration
INPUT_VIDEO="../raw_videos/kitchen_clip_fixed.mp4"
OUTPUT_VIDEO="../output_videos/kitchen_clip_fixed_output.mp4"
SAM3_MODEL="sam3.pt"  # Options: sam3.pt, sam3-t.pt, sam3-s.pt

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_VIDEO")"

# Dynamic class discovery via Gemini (set to false to use hardcoded classes below)
USE_DYNAMIC_CLASSES=True
# Set to true to ignore cached classes and re-run Gemini discovery
FORCE_REDISCOVER=False

# Enhanced discovery settings
DISCOVERY_MODE="base"
SCENE_AWARE=true
MAX_CLASSES=32

# Hardcoded fallback classes (used when USE_DYNAMIC_CLASSES=false or discovery fails)
CUSTOM_CLASSES="small glass,cup,plate,sponge,bottle,lemon,chocolate box,tray,bowl,coffee bean,almond,container,glass,tissue box"

if [ "${USE_DYNAMIC_CLASSES,,}" = true ]; then
    echo "Dynamic class discovery enabled. Running Gemini class discovery..."
    pip install -q google-genai python-dotenv 2>/dev/null

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
SKIP_FRAMES=1       # Process every Nth frame (1 = all frames)
MAX_FRAMES=0        # Limit to N frames (0 = process all frames)
CONFIDENCE_THRESHOLD=0.3  # Minimum confidence score for detections
DEVICE="cuda"       # Device: cuda or cpu
FP16=true           # Half precision inference (safe with 80GB VRAM, ~2x faster)
IMGSZ=1024          # Input image size (default was 640, higher = better quality, more VRAM)

# Build command
CMD="python3 video_demo_sam3.py \
    --input_video \"$INPUT_VIDEO\" \
    --output_video \"$OUTPUT_VIDEO\" \
    --sam3_model \"$SAM3_MODEL\" \
    --classes \"$CUSTOM_CLASSES\" \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --device $DEVICE \
    --skip_frames $SKIP_FRAMES"

# Add optional flags
if [ $MAX_FRAMES -gt 0 ]; then
    CMD="$CMD --max_frames $MAX_FRAMES"
fi

if [ "${FP16,,}" = true ]; then
    CMD="$CMD --fp16"
fi

CMD="$CMD --imgsz $IMGSZ"

# SAM 3 provides masks natively, enable by default
# Add --disable_masking to skip mask overlay on output video
# CMD="$CMD --disable_masking"

# Print configuration
echo "=========================================="
echo "SAM 3 Open-World Video Detection"
echo "=========================================="
echo "Input Video: $INPUT_VIDEO"
echo "Output Video: $OUTPUT_VIDEO"
echo "SAM 3 Model: $SAM3_MODEL"
echo "Custom Classes: $CUSTOM_CLASSES"
echo "Confidence Threshold: $CONFIDENCE_THRESHOLD"
echo "Device: $DEVICE"
echo "Image Size: $IMGSZ"
echo "FP16: $FP16"
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
echo "=========================================="
echo ""

# Run the command
eval $CMD

# Convert detections JSON to NPZ format
OUTPUT_DIR=$(dirname "$OUTPUT_VIDEO")
INPUT_BASE=$(basename "$INPUT_VIDEO")
INPUT_BASE="${INPUT_BASE%.*}"
DETECTIONS_JSON="${OUTPUT_DIR}/${INPUT_BASE}_detections.json"

if [ -f "$DETECTIONS_JSON" ]; then
    echo "Converting detections to NPZ format..."
    python3 "$SCRIPT_DIR/../scripts/json_to_npz.py" "$DETECTIONS_JSON"
else
    echo "WARNING: Detections JSON not found at $DETECTIONS_JSON, skipping NPZ conversion."
fi

echo ""
echo "Done! Output video saved to: $OUTPUT_VIDEO"
echo ""

croc send $OUTPUT_VIDEO
