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
CATEGORIES="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"
# CATEGORIES="object"
TOPK=20
THRESHOLD=0.3

# Processing options
SKIP_FRAMES=1  # Process every 5th frame to reduce memory usage (set to 1 for all frames)
MAX_FRAMES=0   # Limit to 15 frames to avoid OOM (0 = process all frames)

# Build command
CMD="python3 video_demo.py \
    --input_video \"$INPUT_VIDEO\" \
    --output_video \"$OUTPUT_VIDEO\" \
    --model_path \"$MODEL_PATH\" \
    --config-file \"$CONFIG\" \
    --skip_frames $SKIP_FRAMES \
    --batch_size 3"

# Add max_frames if not 0
if [ $MAX_FRAMES -gt 0 ]; then
    CMD="$CMD --max_frames $MAX_FRAMES"
fi

# Add num-gpus if needed (for detectron2)
CMD="$CMD --num-gpus 1"

# Run the command
eval $CMD

# Optional: Add --show_mask to also show segmentation masks (slower)
# To test on first N frames, set MAX_FRAMES=N above

echo ""
echo "Done! Output video saved to: $OUTPUT_VIDEO"
