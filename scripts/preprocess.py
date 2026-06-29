import os
import numpy as np
import rosbag
from rosbag.bag import ROSBagException
import cv2
from cv_bridge import CvBridge
import argparse
import glob

def find_image_topics(bag_path):
    """
    Finds all image topics in the specified ROS bag file.

    Args:
        bag_path (str): Path to the ROS bag file.

    Returns:
        list: A list of tuples, each containing the topic name and message type for image topics.
    """
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # Get a list of all topics and their message types
            topics = bag.get_type_and_topic_info()[1]
            image_topics = []

            # Identify image-related topics (sensor_msgs/Image or sensor_msgs/CompressedImage)
            for topic, info in topics.items():
                if info.msg_type in ['sensor_msgs/Image', 'sensor_msgs/CompressedImage']:
                    image_topics.append((topic, info.msg_type))

            return image_topics

    except ROSBagException as e:
        print(f"Error reading ROS bag: {e}")
        return []  # Return an empty list if there is an error

def extract_rosbag_data(bag_path, output_dir, extract_frames):
    """
    Extracts image frames and saves them as video or individual frames from the given ROS bag.

    Args:
        bag_path (str): Path to the ROS bag file.
        output_dir (str): Directory to save the extracted data.
        extract_frames (bool): Whether to extract individual frames.
    """
    # Find all image topics in the ROS bag
    image_topics = find_image_topics(bag_path)

    for image_topic, msg_type in image_topics:
        # Get camera name from the topic
        cam_name = image_topic.split("/")[1]

        # Create output directories for videos and frames
        _, bag_name = os.path.split(bag_path)
        bag_name, _ = os.path.splitext(bag_name)
        vid_base_path = os.path.join(output_dir, "videos")
        os.makedirs(vid_base_path, exist_ok=True)
        
        if extract_frames:
            img_base_path = os.path.join(output_dir, "frames", f"{bag_name}_{cam_name}")
            os.makedirs(img_base_path, exist_ok=True)

        # Initialize storage for video frames and timestamps
        video_frames = []
        timestamps = []

        # Open the ROS bag file for reading
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[image_topic]):
                if topic == image_topic:
                    # Decode image from the message data
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    video_frames.append(image)
                    timestamps.append(t.to_sec())
        
        # Calculate FPS (frames per second) based on timestamps
        total_time = timestamps[-1] - timestamps[0]
        frame_count = len(timestamps)
        fps = frame_count / total_time if total_time > 0 else 0
        fps = round(fps)

        # Save the video constructed from extracted frames
        video_path = os.path.join(vid_base_path, f"{bag_name}_{cam_name}.mp4")
        save_video(video_frames, video_path, fps)

        if extract_frames:
            # Save individual frames as image files
            for id_, frame in enumerate(video_frames):
                cv2.imwrite(os.path.join(img_base_path, f"{id_:05}.png"), frame)
                
        return video_path

def save_video(frames, output_path, fps=30):
    """
    Saves a list of frames as a video file.

    Args:
        frames (list of numpy.ndarray): The video frames to save. Each frame should be a numpy array with consistent resolution.
        output_path (str): The file path to save the video.
        fps (int): Frames per second for the output video. Default is 30.

    Raises:
        ValueError: If the list of frames is empty.
    """
    if not frames:
        raise ValueError("No frames to save.")
    # Determine video resolution from the first frame
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        out.write(frame)
    out.release()

def main(input_path, output_dir, extract_frames):
    """
    Main function to process ROS bag files from the input directory and extract frames or videos.

    Args:
        input_path (str): Directory containing the ROS bag files.
        output_dir (str): Directory to save the processed data.
        extract_frames (bool): Whether to extract individual frames.
    """
    # Find all ROS bag files in the input directory
    bags = glob.glob(os.path.join(input_path, "*.bag"))

    for bag_path in bags:
        print(bag_path)
        extract_rosbag_data(bag_path, output_dir, extract_frames)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Path of the folder containing the rosbags")
    parser.add_argument("-o", help="Path of the folder where the processed data will be saved")
    parser.add_argument("-f", help="Whether to extract individual frames", action="store_true")
    args = parser.parse_args()

    # Run the main function
    main(args.i, args.o, args.f)
