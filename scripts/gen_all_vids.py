#!/usr/bin/env python3
"""
Batch video generation script for processing all splits and recordings within each split.
Handles structure: main_folder/split_*/folder_name/prediction/*.txt
"""

import os
import subprocess
import argparse
import glob
from pathlib import Path

def find_all_splits(base_results_dir):
    """Find all split directories in the results directory."""
    splits = []
    if not os.path.exists(base_results_dir):
        print(f"Error: Base results directory does not exist: {base_results_dir}")
        return splits
    
    # Look for split_* directories
    split_pattern = os.path.join(base_results_dir, "split_*")
    split_dirs = glob.glob(split_pattern)
    
    for split_dir in sorted(split_dirs):
        if os.path.isdir(split_dir):
            split_name = os.path.basename(split_dir)
            splits.append((split_name, split_dir))
            
    return splits

def find_prediction_dir_in_split(split_path, split_name):
    """
    Find the prediction directory within a split.
    Structure: split_*/folder_name/prediction/
    """
    print(f"Debug [{split_name}]: Looking in split directory: {split_path}")
    
    # List contents of split directory
    if not os.path.exists(split_path):
        print(f"Error [{split_name}]: Split directory does not exist: {split_path}")
        return None
        
    split_contents = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    print(f"Debug [{split_name}]: Subdirectories in split: {split_contents}")
    
    if not split_contents:
        print(f"Warning [{split_name}]: No subdirectories found in split")
        return None
    
    if len(split_contents) > 1:
        print(f"Warning [{split_name}]: Multiple subdirectories found, using first one: {split_contents[0]}")
    
    # Use the first (and presumably only) subdirectory
    folder_name = split_contents[0]
    folder_path = os.path.join(split_path, folder_name)
    
    # Look for prediction subdirectory
    prediction_path = os.path.join(folder_path, "prediction")
    
    if os.path.exists(prediction_path):
        print(f"Debug [{split_name}]: Found prediction directory: {prediction_path}")
        return prediction_path
    else:
        # Also check if prediction are directly in the folder
        print(f"Warning [{split_name}]: No 'prediction' subdirectory found in {folder_path}")
        print(f"Debug [{split_name}]: Contents of {folder_path}: {os.listdir(folder_path) if os.path.exists(folder_path) else 'N/A'}")
        return None

def find_recordings_in_split(frames_dir, prediction_dir, annotations_dir, split_name):
    """
    Find all recordings by matching frame directories with prediction files for a specific split.
    """
    recordings = []
    
    if not prediction_dir:
        return recordings
    
    print(f"Debug [{split_name}]: Looking for prediction files in: {prediction_dir}")
    
    # Get all prediction files
    prediction_files = glob.glob(os.path.join(prediction_dir, "*.txt"))
    print(f"Debug [{split_name}]: Found {len(prediction_files)} .txt files")
    
    if not prediction_files:
        # Show what files are actually there
        all_files = os.listdir(prediction_dir) if os.path.exists(prediction_dir) else []
        print(f"Debug [{split_name}]: All files in prediction_dir: {all_files[:10]}{'...' if len(all_files) > 10 else ''}")
        return recordings
    
    print(f"Debug [{split_name}]: Sample prediction files:")
    for pred_file in prediction_files[:5]:
        print(f"  - {os.path.basename(pred_file)}")
    if len(prediction_files) > 5:
        print(f"  ... and {len(prediction_files) - 5} more")
    
    for pred_file in prediction_files:
        # Extract recording name from prediction file
        recording_name = Path(pred_file).stem
        
        # Check if corresponding frame directory exists
        frames_path = os.path.join(frames_dir, recording_name)
        if not os.path.exists(frames_path):
            print(f"Warning [{split_name}]: Frame directory not found for {recording_name}")
            print(f"  Expected: {frames_path}")
            continue
            
        # Check if ground truth file exists (if annotations_dir is provided)
        gt_path = None
        if annotations_dir and os.path.exists(annotations_dir):
            gt_path = os.path.join(annotations_dir, f"{recording_name}.txt")
            if not os.path.exists(gt_path):
                print(f"Info [{split_name}]: Ground truth file not found for {recording_name} (proceeding without GT)")
                gt_path = None
            
        recordings.append((recording_name, frames_path, pred_file, gt_path))
    
    return recordings

def generate_video(frames_path, prediction_path, output_path, gt_path=None, fps=30):
    """Generate a single video using the gen_video.py script."""
    cmd = [
        "python", "scripts/gen_video.py",
        frames_path,
        prediction_path,
        output_path,
        "--fps", str(fps)
    ]
    
    if gt_path:
        cmd.extend(["--gt", gt_path])
    
    print(f"Generating video: {output_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully generated: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating {output_path}:")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch generate videos for all recordings in all splits")
    
    # Required arguments
    parser.add_argument("frames_base_dir", help="Base directory containing frame subdirectories")
    parser.add_argument("results_base_dir", help="Base directory containing split_* subdirectories")
    parser.add_argument("output_base_dir", help="Base output directory for generated videos")
    
    # Optional arguments
    parser.add_argument("--annotations_dir", help="Directory containing ground truth annotations")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--output_suffix", default="", help="Suffix to add to output video names")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--splits", nargs="+", help="Specific splits to process (e.g., split_1 split_2). If not specified, processes all splits.")
    
    args = parser.parse_args()
    
    # Find all splits
    all_splits = find_all_splits(args.results_base_dir)
    
    if not all_splits:
        print(f"No splits found in {args.results_base_dir}")
        return
    
    # Filter splits if specific ones requested
    if args.splits:
        requested_splits = set(args.splits)
        all_splits = [(name, path) for name, path in all_splits if name in requested_splits]
        
        if not all_splits:
            print(f"None of the requested splits found: {args.splits}")
            return
    
    print(f"Found {len(all_splits)} splits to process:")
    for split_name, _ in all_splits:
        print(f"  - {split_name}")
    
    # Process each split
    total_successful = 0
    total_failed = 0
    
    for split_name, split_path in all_splits:
        print(f"\n{'='*50}")
        print(f"Processing {split_name}")
        print(f"{'='*50}")
        
        # Find the prediction directory in this split
        prediction_dir = find_prediction_dir_in_split(split_path, split_name)
        
        if not prediction_dir:
            print(f"Could not find prediction directory for {split_name}, skipping...")
            continue
        
        # Find recordings in this split
        recordings = find_recordings_in_split(
            args.frames_base_dir, 
            prediction_dir, 
            args.annotations_dir,
            split_name
        )
        
        if not recordings:
            print(f"No recordings found for {split_name}")
            continue
        
        print(f"\nFound {len(recordings)} recordings in {split_name}:")
        for recording_name, _, _, _ in recordings:
            print(f"  - {recording_name}")
        
        # Create output directory for this split
        split_output_dir = os.path.join(args.output_base_dir, split_name)
        if not args.dry_run:
            os.makedirs(split_output_dir, exist_ok=True)
        
        # Generate videos for this split
        split_successful = 0
        split_failed = 0
        
        for recording_name, frames_path, prediction_path, gt_path in recordings:
            # Create output filename
            output_filename = f"{recording_name}{args.output_suffix}.mp4"
            output_path = os.path.join(split_output_dir, output_filename)
            
            if args.dry_run:
                print(f"\n[DRY RUN] Would generate: {output_path}")
                cmd_preview = [
                    "python", "scripts/gen_video.py",
                    frames_path, prediction_path, output_path,
                    "--fps", str(args.fps)
                ]
                if gt_path:
                    cmd_preview.extend(["--gt", gt_path])
                print(f"[DRY RUN] Command: {' '.join(cmd_preview)}")
                continue
            
            # Generate video
            print(f"\n[{split_name}] Processing {recording_name}...")
            if generate_video(frames_path, prediction_path, output_path, gt_path, args.fps):
                split_successful += 1
                total_successful += 1
            else:
                split_failed += 1
                total_failed += 1
        
        # Split summary
        if not args.dry_run:
            print(f"\n{split_name} Summary: {split_successful} successful, {split_failed} failed")
    
    # Overall summary
    if not args.dry_run:
        print(f"\n{'='*50}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*50}")
        print(f"Total videos generated: {total_successful}")
        print(f"Total failures: {total_failed}")
        if total_failed > 0:
            print("Check the error messages above for failed generations.")

if __name__ == "__main__":
    main()