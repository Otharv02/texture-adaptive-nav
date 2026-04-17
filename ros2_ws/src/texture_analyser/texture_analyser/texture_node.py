import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Package to convert between ROS and OpenCV Images
from cv_bridge import CvBridge

import cv2
import numpy as np


# WORLD
WORLD_NAME = "forest"


class OpticalFlowSubscriber(Node):
    """
    Class constructor to set up the node
    """


    def __init__(self):
        super().__init__("optical_flow_subscriber")
        self.subsciber_ = self.create_subscription(Image,
                                                    f'/world/{WORLD_NAME}/model/x500_flow_0/link/flow_link/sensor/flow_camera/image',
                                                    self.listener_callback,
                                                    10
                                                    )
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
        

        # cv2.namedWindow("Sobel X", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Sobel Y", cv2.WINDOW_NORMAL)
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Gradient Magnitude", cv2.WINDOW_NORMAL)
        cv2.namedWindow("laplacian", cv2.WINDOW_NORMAL)        
        # Window size 

        # cv2.resizeWindow("Sobel X", 500, 500)
        # cv2.resizeWindow("Sobel Y", 500, 500)
        cv2.resizeWindow("camera", 500, 500)
        cv2.resizeWindow("Gradient Magnitude", 500, 500)
        cv2.resizeWindow("laplacian", 500, 500)

        self.get_logger().info('Image Subscriber Node has started.')



    def sobel_scharr(self,img):
        """
        Computes image gradients using the Sobel operator.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # X gradient Sobel 
        sobelX = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        # Y gradient Sobel 
        sobelY = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

        return sobelX, sobelY
        

    def laplacian_operation(self,img):
        """
        Laplacian Operator:
        It provides enhanced edge localization.

        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Laplacian(img,cv2.CV_64F)
        return img


    def listener_callback(self, data):
        """
        Call back function
        """
        # self.get_logger().info("Receiving..")
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)


        """
        Compute gradient magnitude:
        Combines X and Y gradients to get overall edge strength at each pixel
        G = sqrt(Gx^2 + Gy^2)
        """
        gx, gy     = self.sobel_scharr(current_frame)
        G_value    = cv2.magnitude(gx, gy)

        """
        compute the variance of the gradient magnitude.
        higher variance -> rich texture
        Low variance    -> flat surface
        """
        G_variance = np.var(G_value)

        G_display = cv2.convertScaleAbs(G_value)
        # gx_display = cv2.convertScaleAbs(gx)
        # gy_display = cv2.convertScaleAbs(gy)

        self.get_logger().info(f"Gradient variance: {G_variance:.2f}")


        """
        Apply Laplacian to detect fine edges, 
        compute its variance as a sharpness metric
        higher variance -> rich texture
        Low variance    -> flat surface
        """
        laplacian = self.laplacian_operation(current_frame)
        # Compute variance (single scalar value)
        laplacian_var = np.var(laplacian)

        # convertScaleAbs -> remove negative + convert large float values + float64 → uint8 (standard image formate)
        laplacian_display = cv2.convertScaleAbs(laplacian)        
        self.get_logger().info(f"Laplacian variance: {laplacian_var:.2f}")



        # Display Image
        
        cv2.imshow("camera", current_frame)
        
        # cv2.imshow("Sobel X", gx_display)
        # cv2.imshow("Sobel Y", gy_display)
        
        cv2.imshow("Gradient Magnitude", G_display)
        cv2.imshow("laplacian", laplacian_display)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
