# ASL Segmentation library

A ROS/Catkin package for Temporal Action Segmentation (TAS). Supports training and evaluation of multiple state-of-the-art TAS models with various feature types.

## Setup

**Conda environment (Python 3.9, CUDA 11.8):**

```bash
conda env create -f environment.yml
conda activate ASL_Segmentation
pip install -e .
```

Configure the project path in [configs/paths.py](configs/paths.py):

```python
project_path = "/path/to/folder"
```

**Docker (GPU training without Conda):**

Prerequisites: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed on the host.

```bash
# Copy the example env file and fill in your W&B API key
cp .env.example .env
# edit .env: WANDB_API_KEY=your_key_here

# Build the image
docker compose build

# Run with default args (mstcn, REASSEMBLEmm, split 1, M2R2_features_demo_prio)
docker compose up

# Override model / dataset / split on the fly
docker compose run training --model ASFormer --dataset JIGSAWS --split 2 --f_type BRP
```

Data, checkpoints, and results are bind-mounted from the host (`./data`, `./checkpoints`, `./results`) so outputs persist after the container exits. If `data/` contains symlinks to external paths, mount those host paths directly in [docker-compose.yml](docker-compose.yml) instead.

## Supported Models

| Model | Description |
|-------|-------------|
| `mstcn` | Multi-Stage Temporal Convolutional Network |
| `ASFormer` | Transformer for Action Segmentation |
| `DiffAct` | Diffusion-based Action Segmentation |
| `asrf` | Action Segmentation with Refinement Framework |
| `BaFormer` | Boundary-aware Transformer |
| `onlinetas` | Online Temporal Action Segmentation |
| `HASR` | Hierarchical Action Segmentation Refiner |

## Feature Types

| Flag | Description | Dim |
|------|-------------|-----|
| `I3D` | I3D video features | 2048 |
| `BRP` | BridgePrompt features | 768 |
| `M2R2` | M2R2 features | 512 |
| `BRPOAF` | BRP + OAF combined | 1536 |
| `UM2R` | Unified M2R features | 512 |

## Data Preparation

### 1. Required directory structure

Every dataset must follow this layout under `data/<DatasetName>/`:

```
data/MyDataset/
├── videos/                    # raw .mp4 files, one per recording
│   ├── recording_01.mp4
│   └── recording_02.mp4
├── frames/                    # extracted frames, one sub-folder per video
│   ├── recording_01/
│   │   ├── 00000.png
│   │   └── 00001.png
│   └── recording_02/
├── annotations/               # per-frame action labels, one file per video
│   ├── recording_01.txt       # one label per line (no header)
│   └── recording_02.txt
├── splits/                    # train/test bundle files
│   ├── train.split1.bundle    # one video name per line (no extension)
│   └── test.split1.bundle
└── mapping.txt                # action index → label, one per line
```

**mapping.txt** format (index and name separated by a space):
```
0 pick_up_bottle
1 pour_juice_into_cup
2 no_action
```

**annotations/recording_01.txt** format (one action label per frame, matching `mapping.txt`):
```
no_action
no_action
pick_up_bottle
pick_up_bottle
pour_juice_into_cup
```

**splits/train.split1.bundle** format (one video base-name per line):
```
recording_01
recording_03
recording_05
```

### 2. Extract frames from raw video (ffmpeg)

If you have plain MP4 files (not ROS bags):

```bash
# Extract frames at 15 fps for each video
for f in data/MyDataset/videos/*.mp4; do
    name=$(basename "$f" .mp4)
    mkdir -p data/MyDataset/frames/$name
    ffmpeg -i "$f" -vf fps=15 data/MyDataset/frames/$name/%05d.png
done
```

If your recordings come from ROS bags, use `preprocess.py` instead — it reads image topics and saves both an MP4 and individual frames:

```bash
conda run -n INVERSE_TAS python scripts/preprocess.py \
    -i /path/to/bags/ \
    -o data/MyDataset/ \
    -f                      # -f also saves individual frames
```

### 3. Create train/test splits

```bash
conda run -n INVERSE_TAS python scripts/create_splits.py \
    data/MyDataset \        # dataset root
    4                       # number of splits (cross-validation folds)
```

### 4. Extract I3D features

I3D extracts RGB features directly from MP4 files. Run it on the host with conda (GPU required):

```bash
conda run -n INVERSE_TAS python scripts/extract_i3d.py \
    --video_dir data/MyDataset/videos/ \
    --device cuda:0 \
    --stack_size 21 \
    --step_size 1 \
    --extraction_fps 15
```

Or inside Docker (overrides the default training entrypoint):

```bash
docker compose run --entrypoint python training \
    scripts/extract_i3d.py \
    --video_dir /workspace/data/MyDataset/videos/ \
    --device cuda:0 \
    --stack_size 21 \
    --step_size 1
```

The script saves intermediate per-video `.npy` files to `data/MyDataset/i3d/`. The expected final location used by training scripts is `data/MyDataset/I3D_features/<video_name>.npy` — rename/move the files from `i3d/` to `I3D_features/` after extraction (the commented-out code block at the bottom of the script does this stacking if you uncomment it and set the right paths).

Each feature file is shaped `(T, 2048)` — T temporal steps × 2048-dim I3D feature.

### 5. Extract BridgePrompt (BRP) features

BRP extraction is a two-step process.

**Step 1 — extract per-window frame features** (requires frames already extracted, and a BridgePrompt config YAML). Copy and adapt an existing config (e.g. `models/BridgePrompt/configs/breakfast/breakfast_exfm.yaml`) for your dataset, then run:

```bash
cd models/BridgePrompt
conda run -n INVERSE_TAS python extract_frame_features.py \
    --config configs/MyDataset/mydataset_exfm.yaml \
    --dataset MyDataset
```

This writes per-window features to `data/MyDataset/brp_raw/<video_name>_<window_idx>.npy`.

**Step 2 — combine windows into per-video features:**

```bash
conda run -n INVERSE_TAS python scripts/extract_brp.py \
    --dataset_name MyDataset \
    --base_path data/ \
    --config_path models/BridgePrompt/configs/MyDataset/mydataset_exfm.yaml
```

This merges the `brp_raw/` windows into `data/MyDataset/BRP_features/<video_name>.npy` files shaped `(768, T)` — 768-dim CLIP features × T frames.

### 6. Train the BridgePrompt model (fine-tuning)

BridgePrompt can be fine-tuned on your dataset to produce action-aware visual features. Training uses a YAML config and must be run from inside the `models/BridgePrompt/` directory (the config paths are relative).

Create a config by copying and editing an existing one:

```bash
cp models/BridgePrompt/configs/breakfast/breakfast_ft.yaml \
   models/BridgePrompt/configs/MyDataset/mydataset_ft.yaml
```

Key fields to change in the YAML:

```yaml
pretrain: "/workspace/models/BridgePrompt/checkpoints/ViT-B-16.pt"  # pre-trained CLIP checkpoint
data:
    dataset: MyDataset
    num_frames: 32          # frames sampled per clip
    num_classes: 7          # number of action classes in mapping.txt
    split: 1
    save_dir: "../../data/MyDataset/brp_raw"
network:
    arch: ViT-B/16
```

Then fine-tune:

```bash
cd models/BridgePrompt
conda run -n INVERSE_TAS python train.py \
    --config configs/MyDataset/mydataset_ft.yaml
```

After training, re-run steps 5a and 5b using the fine-tuned weights (set `pretrain:` in the extract config to point to the saved checkpoint) to regenerate BRP features with the adapted model.

---

## Mock Dataset Walkthrough

The mock dataset (`data/MockDataset/`) provides 5 synthetic 2-minute videos where the frame color changes every 30 seconds — each color maps to one action label.  It lets you run the entire pipeline (feature extraction → training → evaluation) without real recordings.

All commands below assume you are in the project root.  Docker is the recommended runtime; add `conda run -n INVERSE_TAS` as a prefix if you prefer the host Conda environment.

### Step 0 — Generate the dataset

```bash
docker compose run --entrypoint python training scripts/gen_mock_dataset.py
```

This creates:

```
data/MockDataset/
├── videos/                        # 5 × 2-min .mp4 files
├── frames/{video_name}/           # img_00001.jpg … img_01800.jpg
├── annotations/{video_name}.txt   # one action label per frame
├── mapping.txt                    # for onlineTAS / HASR
├── mapping.json                   # for BridgePrompt
└── splits/
    ├── hand_train.split1.bundle   # videos 1–4  (onlineTAS / HASR)
    ├── hand_test.split1.bundle    # video 5
    ├── train.split1.bundle        # videos 1–4  (mstcn / ASFormer)
    ├── test.split1.bundle         # video 5
    └── exfm_nf32.npy              # 285 clip entries for BRP extraction
```

---

### Step 1 — Train BridgePrompt (optional fine-tuning)

BridgePrompt can be used zero-shot (skip this step) or fine-tuned on the dataset to improve feature quality.  Fine-tuning runs from inside `models/BridgePrompt/` and requires the pre-trained ViT-B/16 weights at `models/BridgePrompt/models/vit-16-32f.pt`.  A dedicated GPU with at least 16 GB VRAM is recommended; reduce `batch_size` in the config if you hit OOM.

Config: [configs/BRP/MockDataset/MockDataset_ft.yaml](configs/BRP/MockDataset/MockDataset_ft.yaml)

```bash
docker compose run --entrypoint bash training \
    -c "cd models/BridgePrompt && python train.py \
        --config ../../configs/BRP/MockDataset/MockDataset_ft.yaml \
        --log_time mock_run"
```

The checkpoint is saved to `models/BridgePrompt/exp/clip_ucf/ViT-B/16/MockDataset/mock_run/`.  Set the `pretrain:` key in `MockDataset_exfm.yaml` to that checkpoint path before running feature extraction below.

---

### Step 2 — Extract features

#### 2a. BridgePrompt features

Config: [configs/BRP/MockDataset/MockDataset_exfm.yaml](configs/BRP/MockDataset/MockDataset_exfm.yaml)

**Extract per-clip raw features** (requires GPU):

```bash
docker compose run --entrypoint bash training \
    -c "cd models/BridgePrompt && python extract_frame_features.py \
        --config ../../configs/BRP/MockDataset/MockDataset_exfm.yaml \
        --dataset MockDataset"
```

Writes `data/MockDataset/brp_raw/{video}_{start_frame}.npy` — one (32, 768) array per 32-frame clip.

**Combine clips into per-video feature files:**

```bash
docker compose run --entrypoint python training \
    scripts/extract_brp.py \
    --dataset_name MockDataset \
    --base_path /workspace/data \
    --config_path ""
```

Writes `data/MockDataset/BRP_features/{video}.npy` — shape (T, 768).

#### 2b. I3D features

```bash
docker compose run --entrypoint python training \
    scripts/extract_i3d.py \
    --video_dir /workspace/data/MockDataset/videos/ \
    --device cuda:0 \
    --stack_size 21 \
    --step_size 1
```

Writes `data/MockDataset/I3D_features/{video}.npy` — shape (T, 2048).

---

### Step 3 — Train offline action segmentation

The `docker compose run training` command routes through [scripts/train_TAS.py](scripts/train_TAS.py), which trains, predicts, and evaluates in one call.

#### MS-TCN

```bash
docker compose run training \
    --model mstcn \
    --dataset MockDataset \
    --split 1 \
    --f_type BRP
```

#### ASFormer

```bash
docker compose run training \
    --model ASFormer \
    --dataset MockDataset \
    --split 1 \
    --f_type I3D
```

Checkpoints are saved to `checkpoints/mstcn/MockDataset_BRP_features/split_1/` (or `ASFormer/…`).
Prediction files land in `results/mstcn/MockDataset_BRP_features/split_1/`.

---

### Step 4 — Train online action segmentation (onlineTAS)

```bash
docker compose run --entrypoint python training \
    models/onlinetas/main.py \
    --action train \
    --dataset MockDataset \
    --split 1 \
    --f_type I3D_features \
    --num_epochs 50
```

Use `--f_type BRP_features` to train on BRP features instead.
Use `--inference_mode online` for purely causal inference (default: `semi_online`).

To run prediction after training:

```bash
docker compose run --entrypoint python training \
    models/onlinetas/main.py \
    --action predict \
    --dataset MockDataset \
    --split 1 \
    --f_type I3D_features
```

---

### Step 5 — Train the HASR refiner

HASR takes predictions from a trained backbone and learns to refine them.  The backbone must be trained first (Step 3 or Step 4).

#### Offline refiner (backbone: mstcn or ASFormer)

```bash
docker compose run --entrypoint python training \
    models/HASR/main.py \
    --action train \
    --dataset MockDataset \
    --split 1 \
    --f_type BRP_features \
    --backbone mstcn \
    --refiner gru \
    --num_epochs 60
```

Replace `--backbone mstcn` with `--backbone ASFormer` to use the ASFormer backbone.
Replace `--refiner transformer` with `--refiner gru` for the GRU variant.

#### Online refiner (backbone: onlineTAS)

```bash
docker compose run --entrypoint python training \
    models/HASR/main.py \
    --action online_train \
    --dataset MockDataset \
    --split 1 \
    --f_type BRP_features \
    --backbone onlinetas \
    --refiner gru \
    --num_epochs 60
```

To run online inference after training:

```bash
docker compose run --entrypoint python training \
    models/HASR/main.py \
    --action online_infer \
    --dataset MockDataset \
    --split 1 \
    --f_type BRP_features \
    --backbone onlinetas \
    --refiner gru
```

---

### Step 6 — Evaluate performance

#### Per-model evaluation (runs automatically in train_TAS.py)

When you use `docker compose run training --model mstcn …`, `train_TAS.py` calls the model's `eval.py` automatically after prediction.  To re-run evaluation on existing prediction files:

```bash
docker compose run --entrypoint python training \
    models/mstcn/eval.py \
    --dataset MockDataset \
    --split 1 \
    --f_type BRP_features
```

```bash
docker compose run --entrypoint python training \
    models/onlinetas/eval.py \
    --dataset MockDataset \
    --split 1 \
    --f_type I3D_features
```

#### Compare backbone vs HASR refiner

Once both backbone predictions and HASR predictions exist, compare them side by side:

```bash
docker compose run --entrypoint python training \
    scripts/compare_HASR.py \
    --dataset MockDataset \
    --split 1 \
    --f_type BRP_features \
    --backbone mstcn \
    --refiner transformer
```

Add `--online` to compare the online HASR variant (requires the `online_train` + `online_infer` results).

The script prints a per-epoch table of Acc / Edit / F1@10/25/50 for both the backbone alone and the HASR-refined output, then a best-vs-best delta summary.

---

## Usage

### Training

```bash
conda run -n INVERSE_TAS python scripts/train_TAS.py \
    --dataset ImPerfectPour \
    --split 1 \
    --model mstcn \
    --f_type BRP
```

**HASR with custom backbone:**

```bash
conda run -n INVERSE_TAS python scripts/train_TAS.py \
    --dataset ImPerfectPour \
    --split 1 \
    --model HASR \
    --f_type BRP \
    --hasr_backbone ASFormer \
    --hasr_refiner gru
```

### Inference / Deployment

```bash
conda run -n INVERSE_TAS python scripts/deploy_TAS.py --dataset ImPerfectPour --split 1 --model mstcn --f_type BRP
```

### HASR Comparison

After training a backbone (e.g. mstcn) and a HASR refiner, compare their per-epoch results side by side:

```bash
conda run -n INVERSE_TAS python scripts/compare_HASR.py \
    --dataset ImPerfectPour \
    --split 1 \
    --f_type BRP_features \
    --backbone mstcn \
    --refiner gru
```

The script prints a per-epoch table for both the backbone alone and HASR-refined results, then a best-vs-best summary showing the delta on Acc, Edit, F1@10/25/50.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--backbone` | `mstcn` | `mstcn`, `ASFormer`, or `onlinetas` |
| `--refiner` | `gru` | `gru` or `transformer` |
| `--online` | off | Compare the online HASR variant |

> Requires prediction files to exist under `results/`. Run inference first with `train_TAS.py --action predict` (or the equivalent for the backbone) before running this script.

### Video Visualization

**Single video** — overlay predictions (and optionally ground truth) on extracted frames:

```bash
conda run -n INVERSE_TAS python scripts/gen_video.py \
    /path/to/frames_folder \
    /path/to/prediction_file.txt \
    output.mp4 \
    --fps 30 \
    --gt /path/to/ground_truth.txt
```

The prediction file is the per-video result written to `results/<model>/<dataset>/<split>/epoch_N/<video_name>`. The GT file comes from `data/<dataset>/annotations/<video_name>.txt`. Both flags are optional — omitting `--gt` generates prediction-only output.

**Batch** — generate videos for all recordings across all splits:

```bash
conda run -n INVERSE_TAS python scripts/gen_all_vids.py \
    /path/to/frames_base_dir \
    results/mstcn/ImPerfectPour_BRP_features/split_1/epoch_50 \
    output_videos/ \
    --annotations_dir data/ImPerfectPour/annotations \
    --fps 30
```

Each video is written to `output_videos/split_N/<recording>.mp4`. Use `--dry_run` to preview the commands without executing, and `--splits split_1 split_2` to restrict which splits are processed.

## Unit Tests

The `tests/` directory contains a pytest suite that validates training loops, checkpoint saving, W&B logging, evaluation metrics, model forward passes, and the inference pipeline — all without requiring real datasets or a GPU.

### Running locally (Conda)

```bash
conda run -n INVERSE_TAS python -m pytest tests/ -v
```

### Running in Docker

```bash
docker compose run tests
```

No GPU or data volumes are needed — the test service uses synthetic tensors and temporary directories.

### What is covered

| Test file | Components tested |
|-----------|------------------|
| `test_eval_utils.py` | `_levenstein`, `_edit_score`, `_f_score`, `_dr_f1`, `compute_metrics` |
| `test_wandb_utils.py` | `init_wandb` — run name, project, config keys |
| `test_mstcn.py` | `MultiStageModel`/`SingleStageModel` forward, `BatchGenerator`, `Trainer` training + checkpointing + W&B logging + evaluate |
| `test_asformer.py` | `MyTransformer` forward, `Trainer` training + checkpointing (every-10-epoch rule) + W&B logging + evaluate |
| `test_diffast.py` | `EncoderModel`, `DecoderModel`, `ASDiffusionModel` forward, `prepare_targets`, `get_training_loss` (all six loss terms, finite values, gradient flow) |
| `test_onlinetas.py` | `OnlineTASModel` forward, `MemoryBank` FIFO behaviour, `post_process`, `_split_into_clips`, `Trainer` training + W&B logging |
| `test_hasr.py` | `SparseSampleEmbedder`, `BaseHASR` segment extraction + rollout, `GRURefiner`, `TransformerHASR` forward + gradient flow, `train_refiner` |
| `test_predict.py` | `Predictor.save_pred` output format, end-to-end `predict()` pipeline (feature extractor + seg model mocked) |

### Design notes

- **No real data required** — `BatchGenerator` tests use temporary `.npy` feature files and annotation `.txt` files generated by the `make_dataset` fixture; all trainer tests use a `MockBatchGenerator` that returns synthetic tensors.
- **No GPU required** — models run on CPU. ASFormer's module-level `device` variable is overridden to `cpu` after import to prevent `window_mask` device mismatches.
- **Heavy dependencies mocked** — `test_predict.py` pre-registers `MagicMock` stubs for the I3D and BridgePrompt extractor modules so the script loads without model checkpoints or CUDA.
- **Module name collisions avoided** — `mstcn/model.py`, `ASFormer/model.py`, `onlinetas/model.py`, `HASR/model.py`, and `DiffAct/model.py` all share the filename `model.py`; each is loaded via `importlib.util.spec_from_file_location` with a unique module name to prevent `sys.modules` collisions.

## Project Structure

```
inverse_tas/
├── configs/
│   ├── paths.py          # Project root path
│   ├── wandb_utils.py    # Shared wandb init helper
│   ├── eval_utils.py     # Shared in-memory evaluation metrics
│   ├── ASRF/             # Auto-generated ASRF configs
│   ├── BRP/              # BridgePrompt training configs
│   ├── DiffAct/          # DiffAct training configs
│   └── I3D/              # I3D configs
├── data/                 # Dataset directories (gitignored)
├── models/               # Model implementations
│   ├── mstcn/
│   ├── ASFormer/
│   ├── DiffAct/
│   ├── asrf/
│   ├── BaFormer/
│   ├── onlinetas/
│   ├── HASR/
│   ├── BridgePrompt/     # Feature extractor
│   └── video_features/
├── scripts/              # Training, extraction, and evaluation scripts
├── checkpoints/          # Saved model weights (gitignored)
├── results/              # Evaluation outputs (gitignored)
└── environment.yml       # Conda environment
```

## Experiment Tracking

All models log to the same W&B project `inverse_tas`. Every training run records:

- `train/loss`, `train/acc` — per epoch
- `val/acc`, `val/edit`, `val/f1_10`, `val/f1_25`, `val/f1_50`, `val/dr` — computed on the test split each epoch

Runs are named `{model}_{dataset}_split{N}_{f_type}`. The `wandb/` directory is gitignored; logs are stored remotely.
