# GLEE Video Processing Guide

## Quick Start with `run_video_openworld.sh`

The `run_video_openworld.sh` script has been updated to support the new 3D reconstruction pipeline.

---

## Configuration Options

Edit these variables in `run_video_openworld.sh`:

### Video Input/Output
```bash
INPUT_VIDEO="../raw_videos/room_short_more_objects.qt"
OUTPUT_VIDEO="../output_videos/room_short_more_objects_output_segmented.mp4"
```

### Detection Classes
```bash
# Specify any objects you want to detect (comma-separated, no spaces)
CUSTOM_CLASSES="headphone,lamp,monitor,watch,bottle,hand,laptop,book,phone"

# Examples:
# CUSTOM_CLASSES="pizza,plate,hand,cup"
# CUSTOM_CLASSES="person,car,bicycle,dog,cat"
# CUSTOM_CLASSES="laptop,mouse,keyboard,monitor,phone"
```

### Processing Parameters
```bash
SKIP_FRAMES=1              # Process every Nth frame (1 = all frames)
MAX_FRAMES=0               # Limit to N frames (0 = all frames)
BATCH_SIZE=25              # Frames per batch
CONFIDENCE_THRESHOLD=0.3   # Minimum detection confidence
```

### **NEW: Data Persistence for 3D Reconstruction**
```bash
# Enable saving detection data for DUSt3R 3D reconstruction
SAVE_DETECTIONS=false      # Set to "true" to enable
DETECTIONS_OUTPUT_DIR="../outputs/"  # Where to save data
```

---

## Usage Examples

### Example 1: Quick Test (Detection Only)

**Goal:** Test detection without saving data (faster, less disk space)

```bash
# Edit run_video_openworld.sh:
INPUT_VIDEO="../raw_videos/short_sample.qt"
OUTPUT_VIDEO="../outputs/test_detection.mp4"
CUSTOM_CLASSES="pizza,plate,hand,cup"
MAX_FRAMES=30              # Test on first 30 frames
SAVE_DETECTIONS=false      # Don't save data

# Run
./run_video_openworld.sh
```

**Output:** Annotated video with bounding boxes and masks

---

### Example 2: Full Pipeline (Detection + Save for 3D Reconstruction)

**Goal:** Detect objects AND save data for 3D reconstruction

```bash
# Edit run_video_openworld.sh:
INPUT_VIDEO="../raw_videos/short_sample.qt"
OUTPUT_VIDEO="../outputs/detection_output.mp4"
CUSTOM_CLASSES="pizza,plate,hand,cup,bottle"
MAX_FRAMES=30              # Start with 30 frames
SAVE_DETECTIONS=true       # ← Enable data saving
DETECTIONS_OUTPUT_DIR="../outputs/"

# Run
./run_video_openworld.sh
```

**Output:**
1. Annotated video: `../outputs/detection_output.mp4`
2. Detection data: `../outputs/short_sample/`
   - `metadata.json`
   - `frames/*.{json,pkl}`
   - `rgb_frames/*.jpg`

The script will automatically print the next step:
```
Next step: Run 3D reconstruction
  source ../dust3r/activate_dust3r.sh
  python ../scripts/run_reconstruction.py \
    --data_dir ../outputs/short_sample/ \
    --output_dir ../outputs/reconstruction/
```

---

### Example 3: Long Video with Data Saving

**Goal:** Process full video and prepare for 3D tracking

```bash
# Edit run_video_openworld.sh:
INPUT_VIDEO="../raw_videos/full_egocentric_video.mp4"
OUTPUT_VIDEO="../outputs/full_video_detected.mp4"
CUSTOM_CLASSES="person,hand,cup,bottle,phone,laptop,table,chair"
MAX_FRAMES=0               # Process all frames
SKIP_FRAMES=1              # Process every frame
BATCH_SIZE=10              # Reduce if OOM occurs
SAVE_DETECTIONS=true       # Enable data saving
DETECTIONS_OUTPUT_DIR="../outputs/"

# Run
./run_video_openworld.sh
```

**Warning:** Full videos may take considerable time and disk space.
- 500 frames @ 10 FPS ≈ 50 seconds processing + ~2GB disk space
- Recommend testing with MAX_FRAMES=100 first

---

## Understanding the Flags

### When SAVE_DETECTIONS=false (default)
```bash
python3 video_demo.py \
  --input_video ... \
  --output_video ... \
  --classes "..." \
  --num-gpus 1
```
**Result:** Only creates annotated output video (no saved data)

### When SAVE_DETECTIONS=true
```bash
python3 video_demo.py \
  --input_video ... \
  --output_video ... \
  --classes "..." \
  --save_detections \                      # ← Added
  --detections_output_dir "../outputs/" \  # ← Added
  --num-gpus 1
```
**Result:** Creates annotated video + structured detection data

---

## Output Directory Structure

When `SAVE_DETECTIONS=true`, outputs are saved as:

```
${DETECTIONS_OUTPUT_DIR}/${VIDEO_NAME}/
├── metadata.json          # Video-level info
├── frames/
│   ├── frame_000000.json  # Detection metadata
│   ├── frame_000000.pkl   # Segmentation masks
│   ├── frame_000001.json
│   └── ...
└── rgb_frames/
    ├── frame_000000.jpg   # Original RGB frames
    └── ...
```

**Example:**
- INPUT_VIDEO = "../raw_videos/short_sample.qt"
- DETECTIONS_OUTPUT_DIR = "../outputs/"
- **Saved to:** `../outputs/short_sample/`

---

## Workflow: Detection → 3D Reconstruction

### Step 1: Enable Data Saving
```bash
# Edit run_video_openworld.sh
SAVE_DETECTIONS=true
MAX_FRAMES=30  # Test with 30 frames first

# Run
./run_video_openworld.sh
```

### Step 2: Follow On-Screen Instructions
The script will print:
```
Next step: Run 3D reconstruction
  source ../dust3r/activate_dust3r.sh
  python ../scripts/run_reconstruction.py \
    --data_dir ../outputs/short_sample/ \
    --output_dir ../outputs/reconstruction/
```

### Step 3: Copy-Paste the Commands
```bash
cd ..  # Return to project root
source dust3r/activate_dust3r.sh
python scripts/run_reconstruction.py \
  --data_dir outputs/short_sample/ \
  --output_dir outputs/reconstruction/ \
  --max_windows 1
```

### Step 4: Verify Results
```bash
cat outputs/reconstruction/window_000.json
```

---

## Performance Tips

### For Testing (Fast)
```bash
MAX_FRAMES=30
BATCH_SIZE=25
SAVE_DETECTIONS=false  # Skip saving if you just want to see detections
```

### For 3D Reconstruction (Prepare Data)
```bash
MAX_FRAMES=30              # Start small
BATCH_SIZE=10              # Reduce if OOM
SAVE_DETECTIONS=true       # Enable saving
```

### For Full Pipeline (Production)
```bash
MAX_FRAMES=0               # Process all frames
SKIP_FRAMES=1              # Don't skip
BATCH_SIZE=10              # Adjust based on GPU memory
SAVE_DETECTIONS=true       # Save for reconstruction
```

---

## Memory Considerations

### GPU Memory (RTX 3090 24GB)
- GLEE + SAM: ~6-7GB
- Batch size 25: ~4-5GB working memory
- **Total:** ~10-12GB peak

**If OOM occurs:**
1. Reduce `BATCH_SIZE` (25 → 10)
2. Reduce `MAX_FRAMES` for testing
3. Add `--disable_masking` to CMD (in script, line 51)

### Disk Space
- **Per frame:** ~500KB JSON + ~2MB pickle + ~100KB JPEG ≈ **2.6MB**
- **30 frames:** ~78MB
- **100 frames:** ~260MB
- **500 frames:** ~1.3GB

---

## Troubleshooting

### Script won't run
```bash
# Make executable
chmod +x run_video_openworld.sh

# Check environment
source glee_venv/bin/activate
which python  # Should be GLEE/glee_venv/bin/python
```

### "Command not found" errors
```bash
# Ensure you're in GLEE directory
cd GLEE
./run_video_openworld.sh
```

### No output video created
```bash
# Check input video exists
ls "$INPUT_VIDEO"

# Check model weights exist
ls models/GLEE_Lite_joint.pth
```

### No detection data saved (even with SAVE_DETECTIONS=true)
```bash
# Verify the flag is being added to command
# Check the printed configuration shows "Save Detections: ENABLED"
# If not, check the if-statement syntax in the script
```

---

## Quick Reference

| Goal | SAVE_DETECTIONS | MAX_FRAMES | Time | Output |
|------|-----------------|------------|------|--------|
| Quick test | false | 30 | ~30 sec | Video only |
| Test + save data | true | 30 | ~30 sec | Video + data |
| Full pipeline prep | true | 100 | ~2 min | Video + data |
| Production run | true | 0 | ~10+ min | Video + data |

---

**Updated:** January 29, 2026
**New Features:** --save_detections, --detections_output_dir
**See Also:** ../TEST_GUIDE.md, ../QUICKSTART_RECONSTRUCTION.md
