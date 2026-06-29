"""Generate a synthetic colored-video dataset for testing the TAS pipeline.

Structure created under data/MockDataset/:
  videos/                    - 5 × 2-minute .mp4 files (solid color, changes every 30 s)
  frames/{video_name}/       - extracted JPEG frames  img_NNNNN.jpg  (1-indexed)
  annotations/{video_name}.txt  - one action label per frame
  splits/
    hand_train.split1.bundle - video names for training (no extension)
    hand_test.split1.bundle  - video names for testing
    exfm_nf32.npy            - (N_clips, 2) str array [video_name, start_frame] for BRP
  mapping.txt                - "idx action_name" per line  (for onlineTAS / HASR)
  mapping.json               - {"idx": "action_name", ...}  (for BridgePrompt)
"""

import json
import os
import shutil

import cv2
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
N_VIDEOS       = 5
DURATION_S     = 120      # 2 minutes
SEGMENT_S      = 30       # colour / action every 30 s
FPS            = 15
FRAME_W        = 320
FRAME_H        = 240
BRP_CLIP_FRAMES = 32      # num_frames used by BridgePrompt exfm split
TRAIN_SPLIT    = 4        # first N videos → train, rest → test

ACTIONS = [
    ("action_red",    (  0,   0, 200)),   # BGR
    ("action_green",  (  0, 180,   0)),
    ("action_blue",   (200,   0,   0)),
    ("action_yellow", (  0, 200, 200)),
]

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_ROOT     = os.path.join(PROJECT_ROOT, "data", "MockDataset")

DIRS = {
    "videos":      os.path.join(OUT_ROOT, "videos"),
    "frames":      os.path.join(OUT_ROOT, "frames"),
    "annotations": os.path.join(OUT_ROOT, "annotations"),
    "splits":      os.path.join(OUT_ROOT, "splits"),
}


def make_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    # remove stale features/ dir from previous generation
    old_features = os.path.join(OUT_ROOT, "features")
    if os.path.isdir(old_features):
        shutil.rmtree(old_features)
        print("  Removed old features/")


def write_mapping():
    # mapping.txt  (for onlineTAS / HASR)
    txt_path = os.path.join(OUT_ROOT, "mapping.txt")
    lines = [f"{idx} {name}" for idx, (name, _) in enumerate(ACTIONS)]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))   # no trailing newline — avoids empty-line crash in parsers

    # mapping.json  (for BridgePrompt SegDataset_FRAMES)
    json_path = os.path.join(OUT_ROOT, "mapping.json")
    with open(json_path, "w") as f:
        json.dump({str(idx): name for idx, (name, _) in enumerate(ACTIONS)}, f, indent=4)

    print(f"  mapping.txt / mapping.json → {OUT_ROOT}")


def generate_video(vid_name: str, action_order: list) -> int:
    """Write one MP4, its annotation .txt, and extract all frames.

    Returns total frame count.
    """
    frames_per_segment = SEGMENT_S * FPS
    total_frames       = DURATION_S * FPS
    action_map         = {name: bgr for name, bgr in ACTIONS}

    # ── video ─────────────────────────────────────────────────────────────────
    vid_path = os.path.join(DIRS["videos"], f"{vid_name}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(vid_path, fourcc, FPS, (FRAME_W, FRAME_H))

    # ── frames dir ────────────────────────────────────────────────────────────
    frame_dir = os.path.join(DIRS["frames"], vid_name)
    os.makedirs(frame_dir, exist_ok=True)

    action_to_idx = {name: idx for idx, (name, _) in enumerate(ACTIONS)}
    labels     = []
    label_ints = []
    frame_n    = 0

    for action_name in action_order:
        color = action_map[action_name]
        frame = np.full((FRAME_H, FRAME_W, 3), color, dtype=np.uint8)
        for _ in range(frames_per_segment):
            frame_n += 1
            writer.write(frame)
            # 1-indexed, 5-digit zero-padded: img_00001.jpg
            frame_path = os.path.join(frame_dir, f"img_{frame_n:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            labels.append(action_name)
            label_ints.append(action_to_idx[action_name])

    writer.release()

    # ── annotations (.txt — one label per line; .npy — integer index per frame) ──
    ann_path = os.path.join(DIRS["annotations"], f"{vid_name}.txt")
    with open(ann_path, "w") as f:
        f.write("\n".join(labels) + "\n")

    np.save(os.path.join(DIRS["annotations"], f"{vid_name}.npy"),
            np.array(label_ints, dtype=np.int32))

    print(f"  {vid_name}: {total_frames} frames extracted → {frame_dir}")
    return total_frames


def write_splits(video_names: list):
    train_vids = video_names[:TRAIN_SPLIT]
    test_vids  = video_names[TRAIN_SPLIT:]

    for fname, vids in [
        ("hand_train.split1.bundle", train_vids),
        ("hand_test.split1.bundle",  test_vids),
    ]:
        path = os.path.join(DIRS["splits"], fname)
        with open(path, "w") as f:
            f.write("\n".join(vids) + "\n")
        print(f"  {fname} ({len(vids)} videos) → {path}")

    # train.split1.bundle / test.split1.bundle used by mstcn and ASFormer.
    # Entries must include .txt so ASFormer eval can do ground_truth_path + vid directly.
    for fname, vids in [
        ("train.split1.bundle", train_vids),
        ("test.split1.bundle",  test_vids),
    ]:
        path = os.path.join(DIRS["splits"], fname)
        with open(path, "w") as f:
            f.write("\n".join(v + ".txt" for v in vids) + "\n")

    # remove stale generic bundles if present
    for old in ("train.bundle", "test.bundle"):
        p = os.path.join(DIRS["splits"], old)
        if os.path.exists(p):
            os.remove(p)


def write_exfm_npy(video_names: list, total_frames: int):
    """Build splits/exfm_nf32.npy for BridgePrompt feature extraction.

    Each row: [video_name, str(start_frame_index)]
    Clips are non-overlapping windows of BRP_CLIP_FRAMES frames.
    """
    rows = []
    start_indices = list(range(0, total_frames, BRP_CLIP_FRAMES))
    for vid in video_names:
        for s in start_indices:
            rows.append([vid, str(s)])

    arr = np.array(rows, dtype=str)
    out_path = os.path.join(DIRS["splits"], "exfm_nf32.npy")
    np.save(out_path, arr)
    print(f"  exfm_nf32.npy  shape={arr.shape} → {out_path}")


def write_brp_train_splits(video_names: list, total_frames: int):
    """Create SegDataset-compatible clip split files for BridgePrompt fine-tuning.

    SegDataset.__init__ builds the filename from its DEFAULT ds/ol list values
    ([24, 32] and [1, 1]) since train.py does not pass them.  We match that
    filename exactly.  Actual per-clip ds is 1 (consecutive frames) so every
    clip samples 32 contiguous frames from our short mock videos.

    Each row: [video_name, str(start_frame), str(ds_factor)].
    """
    # These must mirror SegDataset __init__ defaults so the filename matches.
    ds_list = [24, 32]   # default ds list in SegDataset
    ol_list = [1, 1]     # default ol list in SegDataset
    ds_per_clip = 1      # actual sampling stride stored in each row
    n_split = 1
    nf      = BRP_CLIP_FRAMES  # 32

    # SegDataset filename pattern: f'train_split{n}_nf{nf}_ol{ol}_ds{ds}.npy'
    # Python list repr: [24, 32] and [1, 1]
    suffix = f"_nf{nf}_ol{ol_list}_ds{ds_list}"

    train_vids = video_names[:TRAIN_SPLIT]
    test_vids  = video_names[TRAIN_SPLIT:]

    def _make_rows(vids):
        rows = []
        for vid in vids:
            for s in range(0, total_frames, nf):
                rows.append([vid, str(s), str(ds_per_clip)])
        return np.array(rows, dtype=str)

    for prefix, vids in [("train", train_vids), ("test", test_vids)]:
        arr  = _make_rows(vids)
        name = f"{prefix}_split{n_split}{suffix}.npy"
        path = os.path.join(DIRS["splits"], name)
        np.save(path, arr)
        print(f"  {name}  shape={arr.shape} → {path}")


def main():
    np.random.seed(42)

    action_names = [name for name, _ in ACTIONS]

    make_dirs()
    write_mapping()

    print("\nGenerating videos + extracting frames…")
    video_names  = []
    last_n_frames = 0
    for i in range(1, N_VIDEOS + 1):
        vid_name = f"mock_video_{i:02d}"
        video_names.append(vid_name)

        offset       = (i - 1) % len(action_names)
        action_order = [action_names[(offset + j) % len(action_names)]
                        for j in range(DURATION_S // SEGMENT_S)]

        last_n_frames = generate_video(vid_name, action_order)

    print("\nWriting splits…")
    write_splits(video_names)
    write_exfm_npy(video_names, last_n_frames)
    write_brp_train_splits(video_names, last_n_frames)

    print(f"\nDone.  Dataset → {OUT_ROOT}")
    print(f"  Videos  : {N_VIDEOS} × {DURATION_S}s @ {FPS} fps = {last_n_frames} frames each")
    print(f"  Actions : {action_names}")
    print(f"\nPipeline commands (run from project root):")
    print(f"  # 1. Extract BRP features (needs GPU + BridgePrompt checkpoint)")
    print(f"  cd models/BridgePrompt && python extract_frame_features.py \\")
    print(f"      --config ../../configs/BRP/MockDataset/MockDataset_exfm.yaml \\")
    print(f"      --dataset MockDataset")
    print(f"  # 2. Combine BRP clips → BRP_features/")
    print(f"  python scripts/extract_brp.py \\")
    print(f"      --dataset_name MockDataset --base_path data --config_path ''")
    print(f"  # 3. Extract I3D features")
    print(f"  python scripts/extract_i3d.py --video_dir data/MockDataset/videos/")
    print(f"  # 4. Train onlineTAS")
    print(f"  cd models/onlinetas && python main.py \\")
    print(f"      --action train --dataset MockDataset --split 1 --f_type I3D_features")
    print(f"  # 5. Train HASR refiner")
    print(f"  cd models/HASR && python main.py \\")
    print(f"      --action train --dataset MockDataset --split 1 \\")
    print(f"      --f_type BRP_features --backbone mstcn --refiner transformer")


if __name__ == "__main__":
    main()
