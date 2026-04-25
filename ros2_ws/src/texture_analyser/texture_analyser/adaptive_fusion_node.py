import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from px4_msgs.msg import SensorOpticalFlow
from cv_bridge import CvBridge
import cv2
import numpy as np

# Camera constants from SDF
CAMERA_FOV  = 0.733038  # radians
IMAGE_WIDTH  = 100       # pixels
IMAGE_HEIGHT = 100       # pixels
FRAME_DT_US  = 20000     # microseconds (1/50hz × 1,000,000)

# World
# WORLD_NAME = "forest"
# WORLD_NAME = "lawn"
WORLD_NAME = "default"


class AdaptiveFusionNode(Node):

    def __init__(self):
        super().__init__("adaptive_fusion_node")

        # Subscriber - camera image
        self.image_sub_ = self.create_subscription(
            Image,
            f'/world/{WORLD_NAME}/model/x500_flow_0/link/flow_link/sensor/flow_camera/image',
            self.image_callback,
            10
        )

        # Subscriber - confidence score
        self.confidence_sub_ = self.create_subscription(
            Float32,
            'texture/confidence',
            self.confidence_callback,
            10
        )

        # Publisher - optical flow to PX4
        self.optical_flow_pub_ = self.create_publisher(
            SensorOpticalFlow,
            '/fmu/in/sensor_optical_flow',
            10
        )

        self.prev_frame = None
        self.confidence = 0.5
        self.br         = CvBridge()

        self.get_logger().info('Adaptive Fusion Node has started.')


    def confidence_callback(self, msg: Float32):
        self.confidence = msg.data
        self.get_logger().info(f"Confidence received: {self.confidence:.2f}")


    def image_callback(self, data):

        # Convert ROS image to OpenCV grayscale
        current_frame = self.br.imgmsg_to_cv2(data)
        img           = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # First frame check
        if self.prev_frame is None:
            self.prev_frame = img
            return

        # Compute optical flow between previous and current frame
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame,
            img,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Average flow across all pixels
        # flow shape: (100, 100, 2)
        # flow[..., 0] = X displacement
        # flow[..., 1] = Y displacement
        mean_flow_x = float(flow[..., 0].mean())
        mean_flow_y = float(flow[..., 1].mean())

        # Convert pixels to radians
        flow_x_rad = (mean_flow_x / IMAGE_WIDTH)  * CAMERA_FOV
        flow_y_rad = (mean_flow_y / IMAGE_HEIGHT) * CAMERA_FOV

        # Map confidence to quality (0-255)
        quality = int(self.confidence * 255)
        quality = max(0, min(255, quality))

        # Build SensorOpticalFlow message
        timestamp = self.get_clock().now().nanoseconds // 1000  # convert to microseconds

        msg                         = SensorOpticalFlow()
        msg.timestamp               = timestamp
        msg.timestamp_sample        = timestamp
        msg.device_id               = 1
        msg.pixel_flow              = [flow_x_rad, flow_y_rad]
        msg.delta_angle             = [float('nan'), float('nan'), float('nan')]
        msg.delta_angle_available   = False
        msg.distance_available      = False
        msg.integration_timespan_us = FRAME_DT_US
        msg.quality                 = quality
        msg.max_flow_rate           = 2.5
        msg.min_ground_distance     = 0.3
        msg.max_ground_distance     = 25.0

        # Publish
        self.optical_flow_pub_.publish(msg)
        self.get_logger().info(
            f"Flow X: {flow_x_rad:.4f} | "
            f"Flow Y: {flow_y_rad:.4f} | "
            f"Quality: {quality}"
        )

        # Save current frame for next iteration
        self.prev_frame = img


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()