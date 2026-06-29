import rospy
from std_msgs.msg import String
import subprocess
import os
import sys  # getKey stuff
from select import select
from datetime import datetime
import tf

if os.name == 'nt':
  import msvcrt, time
else:
  import tty, termios

class RecordManager:
    def __init__(self):
        # Recording states
        self.is_recording = False
        self.segment_num = 0
        self.segment_success = {}
        self.recording_process = None
        self.timestamp = None

        rospy.init_node("record_manager", anonymous=True)

        self.base_path = rospy.get_param("~base_path")

    def start_recording(self):
        if not self.is_recording:
            # Get params
            topic = rospy.get_param("~topic")
            topic = topic.split()
            path_save = os.path.join(self.base_path, "teleop_bags")
            os.makedirs(path_save, exist_ok=True)
            file_name = self.get_timestamp()

            path = os.path.join(path_save, file_name)

            # Start the recording node
            command = ["rosbag", "record",
                        "--output-name", path]
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

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def get_timestamp(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
    record_manager = RecordManager()

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
            


    # rospy.spin()



