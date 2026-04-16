"""
source install/setup.bash
ros2 run texture_analyser OpticalFlowSubscriber

Implement ROS2 node to subscribe to optical flow camera and display images using OpenCV
"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Package to convert between ROS and OpenCV Images
from cv_bridge import CvBridge

import cv2

class OpticalFlowSubscriber(Node):
    """
    Class constructor to set up the node
    """


    def __init__(self):
        super().__init__("optical_flow_subscriber")
        self.subsciber_ = self.create_subscription(Image,
                                                    '/world/default/model/x500_flow_0/link/flow_link/sensor/flow_camera/image',
                                                    self.listener_callback,
                                                    10
                                                    )
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
        self.get_logger().info('Image Subscriber Node has started.')


    def listener_callback(self, data):
        """
        Call back function
        """

        self.get_logger().info("Receiving..")

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        # Display Image
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        # Window size 
        cv2.resizeWindow("camera", 500, 500)
        cv2.imshow("camera", current_frame)
        cv2.waitKey(1)
        

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
