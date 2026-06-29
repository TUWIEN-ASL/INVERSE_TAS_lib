import cv2
import os
import argparse
import json
import numpy as np

def annotate_video(video_path, output_txt_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize a list to store the actions per frame
    frame_actions = ['no_action'] * total_frames  # Default all frames to "no_action"

    print("Press 'a' to start the first annotation.")
    print("For each segment, press 's' to stop the current action and set the action label.")
    print("Press 'q' to stop annotating and set all remaining frames as 'no_action'.")

    frame_idx = 0
    in_action = False
    action_start = None
    action_label = None

    while cap.isOpened():
        # Read the frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Display the frame
        cv2.imshow('Video Frame', frame)
        
        # Show current frame number
        print(f"Frame: {frame_idx}/{total_frames}")
        
        # Check for key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):  # Quit and mark remaining frames as no_action
            print("Stopping annotation. Marking remaining frames as 'no_action'.")
            # Mark all remaining frames as 'no_action'
            for i in range(frame_idx, total_frames):
                frame_actions[i] = 'no_action'
            break
        elif key == ord('a') and not in_action:  # Start annotating the first segment
            in_action = True
            action_start = frame_idx
            print(f"Started annotating segment from frame {action_start}")
        elif key == ord('s') and in_action:  # Stop annotating the current segment and set action label
            action_label = input(f"Enter the action label for the segment from frame {action_start} to {frame_idx}: ")

            # Mark all frames in this segment with the action label
            for i in range(action_start, frame_idx + 1):
                frame_actions[i] = action_label
            print(f"Finished annotating action: {action_label} from frame {action_start} to {frame_idx}")

            # Start a new segment automatically
            action_start = frame_idx + 1

        print(frame_idx, total_frames - 1, in_action)
        # If it's the last frame, end the segment and ask for the label
        if frame_idx == total_frames - 1 and in_action:
            action_label = input("Enter the action label for the last segment: ")
            # Mark all frames in the last segment with the action label
            for i in range(action_start, total_frames):
                frame_actions[i] = action_label
            print(f"Finished annotating action: {action_label} from frame {action_start} to {total_frames - 1}")
        
        # Automatically assign action to the current frame
        if in_action:
            frame_actions[frame_idx] = action_label

        # Move to next frame
        frame_idx += 1
    
    # Write the frame-by-frame annotations to the output text file
    with open(output_txt_path, 'w') as f:
        for i, action in enumerate(frame_actions):
            if i == len(frame_actions) - 1:
                f.write(action)  # No newline for the last line
            else:
                f.write(f"{action}\n")  # Write with newline for other lines
    
    # Release the video capture and close OpenCV window
    cap.release()
    cv2.destroyAllWindows()
    print(f"Annotations saved to {output_txt_path}")


def generate_mapping(base_dir):
    # Define the paths for video and annotation directories based on the base directory
    annotation_dir = os.path.join(base_dir, 'annotations')
    
    # Check if the annotation directory exists
    if not os.path.exists(annotation_dir):
        print(f"Error: The 'annotations' directory does not exist in the base directory: {annotation_dir}")
        return
    
    # Get all annotation files
    annotation_files = [f for f in os.listdir(annotation_dir) if f.lower().endswith('.txt')]
    
    if not annotation_files:
        print("No annotation files found in the 'annotations' directory.")
        return
    
    # Set to store unique action labels
    unique_actions = set()

    # Read all annotation files and collect unique actions
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        with open(annotation_path, 'r') as f:
            for line in f:
                action = line.strip()  # Remove newline and any extra spaces
                unique_actions.add(action)
    
    # Sort the unique actions for consistency
    sorted_actions = sorted(unique_actions)
    
    # Create the mapping and save it to 'mapping.txt'
    mapping_txt_path = os.path.join(base_dir, 'mapping.txt')
    with open(mapping_txt_path, 'w') as f:
        for idx, action in enumerate(sorted_actions):
            f.write(f"{idx} {action}\n")
    
    # Replace underscores with spaces in actions for the JSON file
    actions_with_spaces = {idx: action.replace('_', ' ') for idx, action in enumerate(sorted_actions)}
    
    # Also create a JSON file for the mapping
    mapping_json_path = os.path.join(base_dir, 'mapping.json')
    
    with open(mapping_json_path, 'w') as json_file:
        json.dump(actions_with_spaces, json_file, indent=4)
    
    print(f"Mapping saved to {mapping_txt_path} and {mapping_json_path}")

def save_annotations_as_numpy(base_dir):
    # Load the mapping from the JSON file
    mapping_json_path = os.path.join(base_dir, 'mapping.json')
    if not os.path.exists(mapping_json_path):
        print(f"Error: The mapping file '{mapping_json_path}' does not exist.")
        return
    
    with open(mapping_json_path, 'r') as f:
        action_mapping = json.load(f)
    
    # Get all annotation files
    annotation_dir = os.path.join(base_dir, 'annotations')
    annotation_files = [f for f in os.listdir(annotation_dir) if f.lower().endswith('.txt')]
    
    if not annotation_files:
        print("No annotation files found in the 'annotations' directory.")
        return
    
    # Iterate over annotation files and convert action names to indices
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        # Read the annotation file
        with open(annotation_path, 'r') as f:
            actions = [line.strip() for line in f]
        
        # Convert actions to indices using the mapping
        annotation_indices = np.array([list(action_mapping.values()).index(action.replace('_', ' ')) if action.replace('_', ' ') in action_mapping.values() else -1 for action in actions])
        
        # Save the NumPy array of indices
        output_file = os.path.join(base_dir, 'annotations', f"{os.path.splitext(annotation_file)[0]}.npy")
        np.save(output_file, annotation_indices)
        print(f"Annotations saved to {output_file}")

def main(base_dir):
    # Define the paths for video and annotation directories based on the base directory
    video_dir = os.path.join(base_dir, 'videos')
    output_dir = os.path.join(base_dir, 'annotations')
    
    # Check if the video directory exists
    if not os.path.exists(video_dir):
        print(f"Error: The 'videos' directory does not exist in the base directory: {video_dir}")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all video files in the video directory
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print("No video files found in the 'videos' directory.")
        return
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        output_txt_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.txt")
        
        print(f"\nProcessing video: {video_file}")
        annotate_video(video_path, output_txt_path)

    generate_mapping(base_dir)
    save_annotations_as_numpy(base_dir)

if __name__ == "__main__":
    # Set up argparse to take base directory as an argument
    parser = argparse.ArgumentParser(description="Annotate videos for temporal action segmentation.")
    parser.add_argument('base_dir', type=str, help="Base directory containing 'videos' and where 'annotations' will be saved.")
    
    args = parser.parse_args()

    # Run the main function with the provided base directory
    main(args.base_dir)