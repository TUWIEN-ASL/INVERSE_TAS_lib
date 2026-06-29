import rospy
from std_msgs.msg import String
import subprocess
import os
import sys  # getKey stuff
from select import select
from datetime import datetime
import time

import multiprocessing

from scripts.preprocess import extract_rosbag_data

from scripts.predict import Predictor

if os.name == 'nt':
  import msvcrt, time
else:
  import tty, termios

def worker(task_queue, executor_instance, stop_event):
    while not stop_event.is_set():  # Check if the stop event is set
        try:
            # Get a task from the queue if it's available
            if not task_queue.empty():
                task = task_queue.get_nowait()
                executor_instance.predict(task, save=True)
        except Exception as e:
            print(f"Error while processing task: {e}")
        time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

class PredictionManager:
    def __init__(self):
        # Recording states
        self.is_recording = False
        self.segment_num = 0
        self.segment_success = {}
        self.recording_process = None
        self.timestamp = None

        rospy.init_node("tas_processing", anonymous=True)

        self.base_path = rospy.get_param("~base_path")
        
        f_ext = rospy.get_param("~f_type")
        seg_model = rospy.get_param("~segmentation_model")
        device = rospy.get_param("~device")
        stack_size = rospy.get_param("~stack_size")
        step_size = rospy.get_param("~step_size")
        brp_config = rospy.get_param("~brp_config_path")
        seg_config = rospy.get_param("~segmentation_config_path")
        seg_weights = rospy.get_param("~segmentation_weights_path")
        num_cls = rospy.get_param("~num_classes")
        mapping_file = rospy.get_param("~mapping_file")
        
        self.pred = Predictor(f_ext, seg_model, device, stack_size, step_size, brp_config, seg_config, seg_weights, num_cls, mapping_file)
        
        # Create a task queue
        self.task_queue = multiprocessing.Queue()

        # Create a stop event to signal the worker to stop
        self.stop_event = multiprocessing.Event()

        # Start a worker process, passing the task queue, class instance, and stop event
        self.process = multiprocessing.Process(target=worker, args=(self.task_queue, self.pred, self.stop_event))
        self.process.start()
    
    def start_recording(self):
        if not self.is_recording:
            # Get params
            topic = rospy.get_param("~topic")
            topic = topic.split()
            path_save = os.path.join(self.base_path, "teleop_bags")
            os.makedirs(path_save, exist_ok=True)
            file_name = self.get_timestamp()

            self.path = os.path.join(path_save, file_name)

            # Start the recording node
            command = ["rosbag", "record",
                        "--output-name", self.path]
            command.extend(topic)
            rospy.logwarn("Launching roslaunch with args: {}".format(command))
            rospy.logwarn(" ".join(command))
            self.recording_process = subprocess.Popen(command)

            self.is_recording = True
            rospy.loginfo("Started recording.")

    def stop_recording(self):
        if self.is_recording:
            rospy.loginfo("Stopping recording...")
            if self.recording_process:
                # Terminate the rosbag process
                self.recording_process.terminate()
                self.recording_process.wait()
                self.recording_process = None
            self.is_recording = False
            rospy.loginfo("Stopped recording.")

            if self.path is not None:
                output_path, name = os.path.split(self.path)
                output_path, _ = os.path.split(output_path)
                output_path = os.path.join(output_path)
                os.makedirs(output_path, exist_ok=True)
                vid_path = extract_rosbag_data(self.path, output_path, True)
                self.task_queue.put(vid_path)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def get_timestamp(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        msg = String()
        msg.data = self.timestamp
        self.pub_curr_timestamp.publish(msg)
        return self.timestamp



def getKey(settings, timeout):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key



if __name__ == "__main__":
    settings = None if sys.platform == 'win32' else termios.tcgetattr(sys.stdin)
    record_manager = PredictionManager()
    
    try:
        while not rospy.is_shutdown():
            key = getKey(settings=settings, timeout=0.1)
            if key == 's':
                record_manager.start_recording()
            elif key == 'p':
                record_manager.stop_recording()
                
            else:
                if key == '\x03':   # ctrl-c
                    break
            
    except Exception as e:
        rospy.logerr(f"Exception occurred: {e}")
