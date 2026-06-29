import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse
from pathlib import Path

class VideoGenerator:
    def __init__(self, frames_folder, annotations_file, output_path, fps=30, gt_file=None):
        self.frames_folder = Path(frames_folder)
        self.annotations_file = Path(annotations_file)
        self.gt_file = Path(gt_file) if gt_file else None
        self.output_path = Path(output_path)
        self.fps = fps
        self.predictions = []
        self.ground_truth = []
        self.action_colors = {}
        self.unique_actions = set()
        
    def load_annotations(self):
        """Load annotations from text file"""
        print("Loading prediction annotations...")
        with open(self.annotations_file, 'r') as f:
            content = f.read().strip()
        
        # Handle prediction file format (all on one line)
        if content.startswith('### Frame level recognition:'):
            lines = content.split('\n')
            prediction_line = None
            for line in lines:
                if line.strip() and not line.startswith('###') and not line.startswith('Frame'):
                    if len(line.split()) > 100:  # Heuristic: the prediction line should be very long
                        prediction_line = line.strip()
                        break
        else:
            prediction_line = content.replace('\n', ' ').strip()
        
        if not prediction_line:
            print("Error: Could not find prediction line in annotation file")
            return
            
        # Split the single line into individual frame predictions
        all_predictions = prediction_line.split()
        print(f"Found {len(all_predictions)} total prediction actions")
        
        # Each action corresponds to one frame
        for action in all_predictions:
            self.predictions.append([action])
            self.unique_actions.add(action)
        
        # Load ground truth if available
        if self.gt_file and self.gt_file.exists():
            print("Loading ground truth annotations...")
            with open(self.gt_file, 'r') as f:
                gt_lines = f.readlines()
            
            # Ground truth format: one action per line
            for line in gt_lines:
                action = line.strip()
                if action:  # Skip empty lines
                    self.ground_truth.append([action])
                    self.unique_actions.add(action)
            
            print(f"Found {len(self.ground_truth)} ground truth actions")
            
            # Ensure both lists have the same length
            min_len = min(len(self.predictions), len(self.ground_truth))
            self.predictions = self.predictions[:min_len]
            self.ground_truth = self.ground_truth[:min_len]
            
            if len(self.predictions) != len(self.ground_truth):
                print(f"Warning: Trimmed to {min_len} frames to match both files")
        else:
            print("No ground truth file provided")
        
        # Generate colors for each unique action
        if self.unique_actions:
            colors = plt.cm.tab20(np.linspace(0, 1, len(self.unique_actions)))
            for i, action in enumerate(sorted(self.unique_actions)):
                self.action_colors[action] = tuple(int(c * 255) for c in colors[i][:3])
        
        total_frames = len(self.predictions)
        print(f"Loaded {total_frames} frames with {len(self.unique_actions)} unique actions")
        print(f"Unique actions: {sorted(self.unique_actions)}")
        
        # Debug: show first few frame annotations
        print("First 5 frame annotations:")
        for i in range(min(5, total_frames)):
            pred_text = self.predictions[i][0] if i < len(self.predictions) else "N/A"
            if self.ground_truth and i < len(self.ground_truth):
                gt_text = self.ground_truth[i][0]
                print(f"  Frame {i+1}: Pred={pred_text}, GT={gt_text}")
            else:
                print(f"  Frame {i+1}: Pred={pred_text}")
        print(f"...and {total_frames-5} more frames" if total_frames > 5 else "")
    
    def get_frame_files(self):
        """Get sorted list of frame files"""
        print("Scanning for frame files...")
        frame_files = []
        
        # First, try to find all existing frame files in the folder
        all_jpg_files = list(self.frames_folder.glob("*.jpg"))
        all_jpg_files.extend(list(self.frames_folder.glob("*.png")))  # Also support PNG
        
        # Sort by the numeric part of the filename
        def extract_number(filename):
            # Extract number from filename like "00001.jpg"
            stem = filename.stem
            try:
                return int(stem)
            except ValueError:
                return 0
        
        all_jpg_files.sort(key=extract_number)
        
        print(f"Found {len(all_jpg_files)} image files in folder")
        
        # If we have more frames than annotations, use frame count
        max_frames = max(len(self.predictions), len(all_jpg_files))
        
        for i, frame_file in enumerate(all_jpg_files):
            if frame_file.exists():
                frame_files.append(frame_file)
            if i >= len(self.predictions) - 1 and len(self.predictions) > 0:
                # Don't go beyond what we have annotations for
                break
                
        print(f"Using {len(frame_files)} frames")
        
        if len(frame_files) != len(self.predictions) and len(self.predictions) > 0:
            print(f"Warning: Mismatch between frames ({len(frame_files)}) and predictions ({len(self.predictions)})")
            
        return frame_files
    
    def create_action_bar(self, width, height=60, actions_list=None, label=""):
        """Create an action prediction bar"""
        if actions_list is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
            
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        frame_width = width / len(actions_list)
        
        for frame_idx, actions in enumerate(actions_list):
            x_start = int(frame_idx * frame_width)
            x_end = int((frame_idx + 1) * frame_width)
            
            if actions:
                # If multiple actions, divide the space equally
                action_height = height // len(actions)
                for i, action in enumerate(actions):
                    y_start = i * action_height
                    y_end = (i + 1) * action_height if i < len(actions) - 1 else height
                    color = self.action_colors.get(action, (128, 128, 128))
                    bar[y_start:y_end, x_start:x_end] = color
        
        # Add label text if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position label on the left side
            x_pos = 5
            y_pos = height // 2 + text_height // 2
            
            # Add background for text
            cv2.rectangle(bar, (x_pos - 2, y_pos - text_height - 2), 
                         (x_pos + text_width + 2, y_pos + 2), (0, 0, 0), -1)
            cv2.putText(bar, label, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        return bar
    
    def add_current_position_line(self, bar, current_frame, total_frames):
        """Add vertical line showing current position"""
        bar_copy = bar.copy()
        x_pos = int((current_frame / total_frames) * bar.shape[1])
        
        # Draw white line with black outline for visibility
        cv2.line(bar_copy, (x_pos, 0), (x_pos, bar.shape[0]), (0, 0, 0), 3)
        cv2.line(bar_copy, (x_pos, 0), (x_pos, bar.shape[0]), (255, 255, 255), 1)
        
        return bar_copy
    
    def add_text_to_frame(self, frame, text, position=(10, 25), font_scale=0.6, thickness=1):
        """Add text with background for better readability"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Create background rectangle
        x, y = position
        padding = 5
        cv2.rectangle(frame, 
                     (x - padding, y - text_height - padding), 
                     (x + text_width + padding, y + baseline + padding), 
                     (0, 0, 0), -1)
        
        # Add white border
        cv2.rectangle(frame, 
                     (x - padding, y - text_height - padding), 
                     (x + text_width + padding, y + baseline + padding), 
                     (255, 255, 255), 1)
        
        # Add text
        cv2.putText(frame, text, position, font, font_scale, (255, 255, 255), thickness)
    
    def create_legend(self, width, height=80):
        """Create a legend for the actions"""
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        
        actions_list = sorted(self.unique_actions)
        
        # Calculate optimal layout based on width
        if width >= 600:
            cols = 6  # More columns for wider frames
        elif width >= 400:
            cols = 4
        else:
            cols = 3
            
        rows = (len(actions_list) + cols - 1) // cols
        
        # If we have too many rows, reduce height per item
        if rows * 15 > height:
            item_height = max(height // rows, 10)
        else:
            item_height = 15
        
        col_width = width // cols
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        
        for i, action in enumerate(actions_list):
            row = i // cols
            col = i % cols
            
            x_start = col * col_width + 5
            y_pos = (row * item_height) + (item_height // 2) + 5
            
            # Skip if we're running out of vertical space
            if y_pos >= height - 5:
                break
                
            # Draw color rectangle (smaller)
            color = self.action_colors[action]
            cv2.rectangle(legend, 
                         (x_start, y_pos - 4), 
                         (x_start + 12, y_pos + 4), 
                         color, -1)
            
            # Truncate long action names
            display_action = action[:15] + "..." if len(action) > 15 else action
            
            # Add text
            cv2.putText(legend, display_action, 
                       (x_start + 16, y_pos + 3), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return legend
    
    def generate_video(self):
        """Generate the final video"""
        print("Starting video generation...")
        
        # Load annotations and get frame files
        self.load_annotations()
        frame_files = self.get_frame_files()
        
        if not frame_files:
            print("No frame files found!")
            return
        
        # Get video dimensions from first frame
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            print(f"Could not load first frame: {frame_files[0]}")
            return
            
        frame_height, frame_width = first_frame.shape[:2]
        
        # Create action bar and legend with adjusted sizes for 640x480
        pred_bar_height = 30   # Reduced for two bars
        gt_bar_height = 30 if self.ground_truth else 0
        legend_height = 0     # Reduced height
        
        # Total video dimensions
        video_width = frame_width
        video_height = frame_height + pred_bar_height + gt_bar_height + legend_height
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (video_width, video_height))
        
        # Create action bars and legend
        gt_bar = None
        if self.ground_truth:
            gt_bar = self.create_action_bar(video_width, gt_bar_height, self.ground_truth, "GT")
        pred_bar = self.create_action_bar(video_width, pred_bar_height, self.predictions, "Pred")
        # legend = self.create_legend(video_width, legend_height)
        
        print(f"Processing {len(frame_files)} frames...")
        
        for frame_idx, frame_file in enumerate(frame_files):
            # Load frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not load frame {frame_file}")
                continue
            
            # Resize frame if necessary
            if frame.shape[:2] != (frame_height, frame_width):
                frame = cv2.resize(frame, (frame_width, frame_height))
            
            # Get current actions
            pred_actions = self.predictions[frame_idx] if frame_idx < len(self.predictions) else []
            pred_text = ", ".join(pred_actions) if pred_actions else "No action"
            
            gt_actions = []
            gt_text = ""
            if self.ground_truth and frame_idx < len(self.ground_truth):
                gt_actions = self.ground_truth[frame_idx]
                gt_text = ", ".join(gt_actions) if gt_actions else "No action"
            
            # Truncate text if too long for frame width
            max_action_length = frame_width // 9
            if len(pred_text) > max_action_length:
                pred_text = pred_text[:max_action_length-3] + "..."
            if len(gt_text) > max_action_length:
                gt_text = gt_text[:max_action_length-3] + "..."
            
            # Add ground truth text if available (second line)
            if gt_text:
                self.add_text_to_frame(frame, f"GT: {gt_text}", (10, frame_height-50), font_scale=0.4)

            # Add prediction text to frame (top left)
            self.add_text_to_frame(frame, f"Pred: {pred_text}", (10, frame_height-20), font_scale=0.4)
            
            # Add frame number (top right)
            frame_text = f"Frame: {frame_idx + 1}/{len(frame_files)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            (text_width, text_height), _ = cv2.getTextSize(frame_text, font, font_scale, 1)
            frame_pos_x = frame_width - text_width - 10
            self.add_text_to_frame(frame, frame_text, (frame_pos_x, 20), font_scale=0.4)
            
            # Create prediction bar with current position
            current_pred_bar = self.add_current_position_line(pred_bar, frame_idx, len(frame_files))
            
            # Create ground truth bar with current position if available
            current_gt_bar = None
            if gt_bar is not None:
                current_gt_bar = self.add_current_position_line(gt_bar, frame_idx, len(frame_files))
            
            # Combine all elements
            combined_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            
            # Place main frame
            combined_frame[0:frame_height, 0:frame_width] = frame
            
            # Place prediction bar
            # Place ground truth bar if available
            if current_gt_bar is not None:
                y_start = frame_height
                combined_frame[y_start:y_start + gt_bar_height, 0:video_width] = current_gt_bar

            y_start = frame_height + gt_bar_height
            combined_frame[y_start:y_start + pred_bar_height, 0:video_width] = current_pred_bar
            
            
            
            # Place legend
            # y_start = frame_height + pred_bar_height + gt_bar_height
            # combined_frame[y_start:, 0:video_width] = legend
            
            # Write frame
            out.write(combined_frame)
            
            if (frame_idx + 1) % 100 == 0:
                print(f"Processed {frame_idx + 1}/{len(frame_files)} frames")
        
        # Release everything
        out.release()
        print(f"Video saved to: {self.output_path}")
        print("Video generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate video with action annotations")
    parser.add_argument("frames_folder", help="Path to folder containing frame images")
    parser.add_argument("annotations_file", help="Path to predictions text file")
    parser.add_argument("output_video", help="Path for output video file")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (default: 30)")
    parser.add_argument("--gt", "--ground-truth", dest="gt_file", 
                       help="Path to ground truth annotations file (optional)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.frames_folder):
        print(f"Error: Frames folder '{args.frames_folder}' does not exist")
        return
    
    if not os.path.exists(args.annotations_file):
        print(f"Error: Annotations file '{args.annotations_file}' does not exist")
        return
    
    if args.gt_file and not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file '{args.gt_file}' does not exist")
        return
    
    # Create video generator and generate video
    generator = VideoGenerator(
        frames_folder=args.frames_folder,
        annotations_file=args.annotations_file,
        output_path=args.output_video,
        fps=args.fps,
        gt_file=args.gt_file
    )
    
    generator.generate_video()

if __name__ == "__main__":
    main()