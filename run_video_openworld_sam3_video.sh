#!/bin/bash

# SAM 3 Video Predictor Mode - Open World Detection + Segmentation + Tracking
# Uses SAM3VideoSemanticPredictor which propagates memory across frames for
# more temporally consistent tracking vs the per-frame SAM3SemanticPredictor.
#
# Optionally enables torch.compile for faster steady-state throughput (expect
# slow first ~3 frames due to compilation, then faster).
#
# Setup:
#   bash setup_sam3_env.sh   # One-time environment setup
#
# Usage:
#   bash run_video_openworld_sam3_video.sh

# Activate SAM 3 virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/sam3_venv" ]; then
    source "$SCRIPT_DIR/sam3_venv/bin/activate"
else
    echo "ERROR: sam3_venv not found. Run: bash setup_sam3_env.sh"
    exit 1
fi

# ==========================================
# Configuration
# ==========================================
INPUT_VIDEO="../raw_videos/kitchen_clip_fixed.mp4"
OUTPUT_VIDEO="../output_videos/kitchen_clip_fixed_output.mp4"
SAM3_MODEL="sam3.pt"  # Options: sam3.pt, sam3-t.pt, sam3-s.pt

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_VIDEO")"

# --- Dynamic class discovery via Gemini ---
USE_DYNAMIC_CLASSES=True
FORCE_REDISCOVER=False
DISCOVERY_MODE="base"
SCENE_AWARE=true
MAX_CLASSES=32

# Hardcoded fallback classes
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

# --- Processing options ---
SKIP_FRAMES=1               # Process every Nth frame (1 = all frames)
MAX_FRAMES=0                # Limit to N frames (0 = process all frames)
CONFIDENCE_THRESHOLD=0.45   # Minimum confidence score for detections
DEVICE="cuda"               # Device: cuda or cpu
FP16=true                   # Half precision inference
IMGSZ=1036                  # Input image size

# --- torch.compile ---
# Options: "" (disabled), "default", "reduce-overhead", "max-autotune-no-cudagraphs"
# "reduce-overhead" recommended for A100 (best steady-state throughput after warmup)
COMPILE_MODE="reduce-overhead"

# --- Video predictor tracking parameters (leave empty for defaults) ---
SCORE_THRESHOLD_DETECTION=""   # e.g. 0.3
DET_NMS_THRESH=""              # e.g. 0.7
ASSOC_IOU_THRESH=""            # e.g. 0.3
TRK_ASSOC_IOU_THRESH=""        # e.g. 0.3
INIT_TRK_KEEP_ALIVE=""         # e.g. 5
MAX_TRK_KEEP_ALIVE=""          # e.g. 30
FILL_HOLE_AREA=""              # e.g. 8
HOTSTART_DELAY=""              # e.g. 3

# ==========================================
# Build command
# ==========================================
CMD="python3 video_demo_sam3_video.py \
    --input_video \"$INPUT_VIDEO\" \
    --output_video \"$OUTPUT_VIDEO\" \
    --sam3_model \"$SAM3_MODEL\" \
    --classes \"$CUSTOM_CLASSES\" \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --device $DEVICE \
    --skip_frames $SKIP_FRAMES \
    --imgsz $IMGSZ"

# Optional flags
if [ $MAX_FRAMES -gt 0 ]; then
    CMD="$CMD --max_frames $MAX_FRAMES"
fi

if [ "${FP16,,}" = true ]; then
    CMD="$CMD --fp16"
fi

if [ -n "$COMPILE_MODE" ]; then
    CMD="$CMD --compile \"$COMPILE_MODE\""
fi

# Video predictor tracking params (only add if non-empty)
if [ -n "$SCORE_THRESHOLD_DETECTION" ]; then
    CMD="$CMD --score_threshold_detection $SCORE_THRESHOLD_DETECTION"
fi
if [ -n "$DET_NMS_THRESH" ]; then
    CMD="$CMD --det_nms_thresh $DET_NMS_THRESH"
fi
if [ -n "$ASSOC_IOU_THRESH" ]; then
    CMD="$CMD --assoc_iou_thresh $ASSOC_IOU_THRESH"
fi
if [ -n "$TRK_ASSOC_IOU_THRESH" ]; then
    CMD="$CMD --trk_assoc_iou_thresh $TRK_ASSOC_IOU_THRESH"
fi
if [ -n "$INIT_TRK_KEEP_ALIVE" ]; then
    CMD="$CMD --init_trk_keep_alive $INIT_TRK_KEEP_ALIVE"
fi
if [ -n "$MAX_TRK_KEEP_ALIVE" ]; then
    CMD="$CMD --max_trk_keep_alive $MAX_TRK_KEEP_ALIVE"
fi
if [ -n "$FILL_HOLE_AREA" ]; then
    CMD="$CMD --fill_hole_area $FILL_HOLE_AREA"
fi
if [ -n "$HOTSTART_DELAY" ]; then
    CMD="$CMD --hotstart_delay $HOTSTART_DELAY"
fi

# ==========================================
# Print configuration
# ==========================================
echo "=========================================="
echo "SAM 3 Video Predictor - Open World"
echo "=========================================="
echo "Input Video: $INPUT_VIDEO"
echo "Output Video: $OUTPUT_VIDEO"
echo "SAM 3 Model: $SAM3_MODEL"
echo "Custom Classes: $CUSTOM_CLASSES"
echo "Confidence Threshold: $CONFIDENCE_THRESHOLD"
echo "Device: $DEVICE"
echo "Image Size: $IMGSZ"
echo "FP16: $FP16"
echo "Compile Mode: ${COMPILE_MODE:-disabled}"
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

# ==========================================
# Post-processing: Convert JSON to NPZ
# ==========================================
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
