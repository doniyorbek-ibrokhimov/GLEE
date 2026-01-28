# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GLEE (General Object Foundation Model) is a unified vision foundation model for object-centric tasks across images and videos. It's built on Detectron2 and trained on 10M+ images from 16 datasets with diverse supervision levels.

**Key Capabilities:**
- Object detection, instance segmentation, grounding, tracking
- Video instance segmentation (VIS), video object segmentation (VOS)
- Interactive segmentation with visual prompts (points, boxes, scribbles)
- Open-world/large-vocabulary detection with CLIP text encoder
- Zero-shot transfer across object-level tasks

## Project Structure

```
GLEE/
├── projects/GLEE/           # Core GLEE implementation
│   ├── glee/                # Main package
│   │   ├── GLEE.py          # MetaArch orchestrator (training/inference entry)
│   │   ├── models/          # Core neural network components
│   │   │   ├── glee_model.py         # Main model architecture
│   │   │   ├── pixel_decoder/        # Multi-scale feature extraction
│   │   │   ├── transformer_decoder/  # Query-based prediction heads
│   │   │   ├── matcher.py            # Hungarian matching
│   │   │   └── criterion.py          # Multi-task loss computation
│   │   ├── backbone/        # Visual encoders (Swin, EVA02, ResNet, ViT)
│   │   ├── data/            # Dataset mappers and loaders
│   │   └── config.py        # Configuration system
│   ├── configs/             # Training/inference YAML configs
│   │   ├── images/          # Image tasks (Stage1/2/3 for Lite/Plus/Pro)
│   │   └── videos/          # Video tasks (YTVIS, OVIS, TAO, BURST, etc.)
│   └── train_net.py         # Training entry point
├── detectron2/              # Modified detectron2 framework
├── video_demo.py            # Video inference script with SAM integration
├── app.py                   # Gradio web interface
└── launch.py                # Multi-node distributed training launcher
```

## Architecture Overview

### Model Pipeline

```
Input → [Backbone] → [Pixel Decoder] → [Transformer Decoder] → Outputs
         (Swin/EVA02)  (Multi-scale      (Query-based)         (boxes, masks,
                        Deformable        + Task-specific       classes, track
                        Transformer)      Label Encoder)        embeddings)
                           ↑
                    [Text Encoder] (CLIP)
                    [Visual Prompter] (Early Fusion)
```

### Key Components

**GLEE.py** (`projects/GLEE/glee/GLEE.py`):
- Registered as `META_ARCH_REGISTRY` with Detectron2
- Orchestrates training/inference pipelines
- Handles multi-dataset configuration (COCO, LVIS, Objects365, video tasks)
- Routes between image and video processing modes
- Implements category sampling for open-world detection (up to 100 random negatives per batch)

**GLEE_Model** (`projects/GLEE/glee/models/glee_model.py`):
- Core neural network module with:
  - **Backbone**: Vision encoder (supports Swin, EVA02, EVA01, ViT, ResNet)
  - **Pixel Decoder**: Multi-scale deformable transformer encoder with early fusion
  - **Transformer Decoder**: Query-based detection with task-specific label encoders
  - **Text Encoder**: CLIP-based language model (frozen or trainable)
  - **Feature Fuser**: Combines visual prompts with image features

**Pixel Decoder** (`glee/models/pixel_decoder/maskdino_encoder.py`):
- Bottom-up multi-scale feature extraction
- MSDeformAttn (Multi-Scale Deformable Attention)
- Early fusion layer for text and visual prompts

**Transformer Decoder** (`glee/models/transformer_decoder/maskdino_decoder.py`):
- Deformable cross-attention for query-to-feature interaction
- Deep supervision with intermediate predictions
- Two-stage design: initial proposals + iterative refinement

## Development Commands

### Environment Setup

```bash
# Activate virtual environment
source glee_venv/bin/activate

# Install dependencies (see assets/INSTALL.md for full details)
pip3 install -e .
cd projects/GLEE/glee/models/pixel_decoder/ops/
python3 setup.py build install --user
cd -

# Download CLIP text encoder
wget -P projects/GLEE/clip_vit_base_patch32/ \
  https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin
```

### Training

**Single Machine (8 GPUs):**
```bash
# Stage 2 joint training (most common)
python3 projects/GLEE/train_net.py \
  --config-file projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml \
  --num-gpus 8

# Video task fine-tuning
python3 projects/GLEE/train_net.py \
  --config-file projects/GLEE/configs/videos/Lite/ytvis19_base.yaml \
  --num-gpus 8 \
  MODEL.WEIGHTS /path/to/GLEE_Lite_joint.pth
```

**Multi-Machine (distributed):**
```bash
python3 launch.py \
  --nn <num_machines> \
  --port <PORT> \
  --worker_rank <rank> \
  --master_address <MASTER_IP> \
  --config-file projects/GLEE/configs/<config.yaml>
```

### Inference

**Video Inference (Open-World Detection):**
```bash
# Using the shell script (recommended)
./run_video_openworld.sh

# Direct Python invocation
python3 video_demo.py \
  --input_video /path/to/video.mp4 \
  --output_video /path/to/output.mp4 \
  --model_path /path/to/GLEE_Lite_joint.pth \
  --config-file projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml \
  --classes "person,car,dog,laptop" \
  --confidence_threshold 0.3 \
  --batch_size 1 \
  --skip_frames 1 \
  --disable_masking  # Optional: disable SAM masking to reduce GPU memory
  --num-gpus 1
```

**Gradio Web App:**
```bash
python app.py  # Launches Gradio interface at http://localhost:7860
```

## Configuration System

### Config Hierarchy
1. **Base Detectron2 Config**: Standard D2 settings
2. **GLEE Config** (`add_glee_config(cfg)`): Adds GLEE-specific options
3. **Task-Specific Configs**: Image vs. video processing settings
4. **Dataset-Specific**: Batch sizes, sampling strategies, loss weights

### Config Structure
```
Stage{1,2,3}_{task}_{text_encoder}_{backbone}.yaml

Stages:
- Stage 1: Pre-training on Objects365 + OpenImages
- Stage 2: Joint training on 15 datasets (most commonly used)
- Stage 3: Scale-up with SA1B + GRIT data

Variants:
- Lite:  ResNet-50 backbone
- Plus:  Swin-Large backbone
- Pro:   EVA02-Large backbone
```

### Key Config Groups
- **INPUT**: Frame sampling, augmentation, dataset mapper selection
- **MODEL**: Architecture (backbone, pixel_decoder, transformer, text_encoder)
- **DATALOADER**: Multi-dataset ratios, batch sizes per dataset
- **SOLVER**: Optimizer, learning rates, gradient clipping, warmup

## Data Loading

### Supported Datasets
**Images**: COCO, LVIS, Objects365, OpenImage, VG, BDD100K, ODINW, SA1B
**Videos**: YTVIS (2019/2021), OVIS, UVOv2, TAO, BURST, RVOS, YTBVoS

### Dataset Mappers (in `glee/data/`)
- `joint_image_dataset_LSJ_mapper.py`: Large-scale jittering for image datasets
- `joint_image_video_dataset_LSJ_mapper.py`: Mixed image-video training
- `uni_video_image_mapper.py`: Unified video/image pipeline
- `vis_dataset_mapper.py`: Video instance segmentation
- `refcoco_dataset_mapper.py`: Referring expression grounding

### Multi-Dataset Training Features
- Category sampling: Up to 100 random negative classes per batch for open-world learning
- Task-specific loss weighting
- Different batch sizes per dataset
- Pseudo-video generation from image datasets (repeat single frame)

## Training vs. Inference Pipelines

### Training Flow
```
Multi-dataset loader
  ↓
Dataset mapper (augmentation + normalization)
  ↓
GLEE.forward(batched_inputs, training=True)
  ↓
get_task_name() → determine dataset type (coco/lvis/o365/video)
  ↓
prepare_targets() → normalize boxes, process masks
  ↓
category_name_sampling() → select class labels (including negatives)
  ↓
GLEE_Model.forward() → backbone + pixel decoder + transformer decoder
  ↓
criterion(outputs, targets) → compute losses (cls, mask, box, tracking)
  ↓
backward + optimize
```

### Inference Flow
```
Input (image/video/directory)
  ↓
preprocess_image() → normalize + pad
  ↓
GLEE.forward(batched_inputs, training=False)
  ↓
Task routing:
  - Video tasks: MinVIS_inference() with temporal tracking
  - Image tasks: instance_inference() with top-k selection
  - Grounding: special handling for referring expressions
  ↓
Post-processing (box scaling, mask upsampling)
  ↓
Output: Instances with masks, boxes, scores, classes, track IDs
```

## Key Architectural Features

| Feature | Implementation |
|---------|----------------|
| **Multi-Task Learning** | Task name routing in `get_task_name()` with dataset-specific configs |
| **Visual Prompting** | Spatial prompt encoding via `GLEE_Model.get_template()` for interactive segmentation |
| **Early Fusion** | Text + visual features fused at pixel decoder (VLFuse layer) |
| **Temporal Tracking** | Embedding-based tracking with contrastive loss (`get_tracking_contrastive_lossv3()`) |
| **Open-World Detection** | Dynamic class embeddings from CLIP + category sampling strategy |
| **Deformable Attention** | MSDeformAttn for efficient multi-scale feature interaction |
| **Two-Stage Queries** | Initial proposals refined through iterative decoder layers |
| **Deep Supervision** | Intermediate predictions at each decoder layer for better gradients |

## Loss Functions

**Criterion** (`glee/models/criterion.py`):
- **Classification**: Focal loss + Cross-entropy
- **Segmentation**: Dice loss + Mask BCE loss
- **Box Regression**: L1 loss + GIoU loss
- **Tracking**: Contrastive loss for video tasks (video_demo.py:232-290)
- **Denoising**: Query denoising loss (two-stage)

**Target Assignment**: Hungarian matching (`matcher.py`) for optimal prediction-target pairing

## Video Processing Notes

### video_demo.py
- Extracts frames from video files or image directories
- Integrates SAM (Segment Anything Model) for mask refinement
- Supports custom class names for open-world detection via `--classes` argument
- Batch processing with configurable batch size (use `batch_size=1` for open-world to avoid OOM)
- Optional SAM masking toggle:
  - `--enable_masking` (default): Generate segmentation masks (GPU-intensive)
  - `--disable_masking`: Bounding boxes only (faster, less memory)
- Incremental video writing for memory efficiency

### run_video_openworld.sh
- Shell script wrapper for video_demo.py
- Activates virtual environment automatically
- Configurable parameters:
  - `CUSTOM_CLASSES`: Comma-separated class names (no spaces)
  - `SKIP_FRAMES`: Process every Nth frame (1 = all frames)
  - `MAX_FRAMES`: Limit processing to N frames (0 = all)
  - `BATCH_SIZE`: Frames per batch (1 recommended for open-world)
  - `CONFIDENCE_THRESHOLD`: Minimum detection confidence
- Currently configured with `--disable_masking` to avoid OOM errors

## Common Development Patterns

### Adding New Backbone
1. Register in `glee/backbone/registry.py`
2. Initialize weights in config YAML under `MODEL.BACKBONE`
3. For large backbones, initialize non-backbone components with GLEE-Lite weights

### Custom Dataset Integration
1. Create dataset mapper in `glee/data/`
2. Register dataset in `glee/data/datasets/`
3. Add to multi-dataset loader in `glee/data/build.py`
4. Update config YAML with dataset ratios and batch size

### Task-Specific Head Modifications
- Modify `glee/models/transformer_decoder/maskdino_decoder.py`
- Add task-specific label encoder if needed
- Update loss computation in `glee/models/criterion.py`

## Model Weights

Models are stored as `.pth` files containing state dicts. Key checkpoints:
- `GLEE_Lite_joint.pth`: ResNet-50 based (fastest)
- `GLEE_Plus_joint.pth`: Swin-Large based (better quality)
- `GLEE_Pro_joint.pth`: EVA02-Large based (best quality)

Download from MODEL_ZOO.md or HuggingFace.

## Important Constraints

### Memory Management
- Open-world detection with multiple classes is memory-intensive
- Use `batch_size=1` for video processing with many classes
- SAM masking significantly increases GPU memory usage
- Process frames incrementally and write to video immediately (don't accumulate)

### Detectron2 Integration
- GLEE extends Detectron2's `META_ARCH_REGISTRY`
- Must be registered and built via Detectron2's config system
- Uses D2's checkpoint management, data API, and evaluation tools
- Custom components in `glee/` override D2 defaults when registered

### Multi-Dataset Training
- Each dataset has its own task name (e.g., 'coco', 'lvis', 'o365', 'ytvis19')
- Task routing happens in `GLEE.get_task_name()` based on file paths
- Category sampling ensures diverse negative examples for open-world learning
- Loss weights and batch sizes vary by dataset (configured in YAML)
