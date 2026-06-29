import os
import argparse
import glob
import subprocess
from configs.paths import project_path
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare and run video processing with i3d")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the model")
    parser.add_argument('--stack_size', type=int, default=21, help="Stack size")
    parser.add_argument('--step_size', type=int, default=1, help="Step size")
    parser.add_argument('--extraction_fps', type=int, default=15, help="Frames per second for extraction")
    parser.add_argument('--video_dir', type=str, required=True, help="Directory containing video files")
    parser.add_argument('--flow', action='store_true', help="Also extract optical flow features (RAFT). Default: RGB only")
    return parser.parse_args()

def get_video_paths(directory):
    paths = []
    for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)

def create_output_path(base_path):
    return os.path.join(base_path, 'i3d')

def create_output_final_path(base_path):
    return os.path.join(base_path, 'I3D_features')

def get_main_py_path():
    # Assuming the main.py is in a subfolder, e.g., "scripts"
    # project_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script's directory
    main_py_path = os.path.join(project_path, 'models', "video_features", 'main.py')  # Adjust relative path to main.py
    return main_py_path

def main():
    args = parse_arguments()

    # Get list of video files
    video_paths = get_video_paths(args.video_dir)

    # os.path.split preserves a trailing slash, which would make base_dir point
    # inside videos/ rather than the dataset root.  Normalize first.
    base_dir = os.path.dirname(os.path.abspath(args.video_dir))

    # Save the list of video paths to a .txt file
    video_list_file = os.path.join(base_dir, 'vids.txt')
    with open(video_list_file, 'w') as f:
        for path in video_paths:
            f.write(f"{path}\n")

    # Set output path
    output_path = create_output_path(base_dir)
    os.makedirs(output_path, exist_ok=True)

    # Get the dynamic path to main.py
    main_py_path = get_main_py_path()

    # Prepare the command to run main.py with the required arguments
    command = [
        'python', main_py_path,
        f'feature_type=i3d',
        f'device={args.device}',
        f'stack_size={args.stack_size}',
        f'step_size={args.step_size}',
        f'file_with_video_paths={video_list_file}',
        f'output_path={output_path}',
        'on_extraction=save_numpy',
        # omitting `streams` passes None to ExtractI3D, which uses both rgb+flow
        *([] if args.flow else ['streams=rgb']),
        # f'extraction_fps={args.extraction_fps}'
    ]

    # Call main.py with the constructed command
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

    # Rename i3d/{name}_rgb.npy → I3D_features/{name}.npy
    # Features stay as (T, D) — no transpose; batch_gen loads with .T to get (D, T).
    target_path = create_output_final_path(base_dir)
    os.makedirs(target_path, exist_ok=True)

    rgb_paths = sorted(glob.glob(os.path.join(output_path, "i3d", "*_rgb.npy")))
    if not rgb_paths:
        # video_features writes directly into output_path (no i3d/ subdir)
        rgb_paths = sorted(glob.glob(os.path.join(output_path, "*_rgb.npy")))

    for rgb_path in rgb_paths:
        _, fname = os.path.split(rgb_path)
        name     = "_".join(fname.split("_")[:-1])   # strip _rgb suffix
        new_path = os.path.join(target_path, name + ".npy")
        feats    = np.load(rgb_path)                  # shape (T, D), already correct
        np.save(new_path, feats)

    print(f"I3D features saved to {target_path}  ({len(rgb_paths)} files)")

if __name__ == '__main__':
    main()
