import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

# Package to convert between ROS and OpenCV Images
from cv_bridge import CvBridge

import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk


# WORLDS
# WORLD_NAME = "forest"
# WORLD_NAME = "lawn"
WORLD_NAME = "default"


# Range values (test only)
G_min = 100
G_max = 230000

L_min = 0.09
L_max = 1100

E_min = 0.15
E_max = 5.0


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
        
        self.confidence_pub_ = self.create_publisher(Float32, 
                                                     'texture/confidence',
                                                     10)
    
        

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
        
        cv2.namedWindow("Texture Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Texture Analysis", 700, 700)
        self.get_logger().info('Image Subscriber Node has started.')

    def img_combined(self, camera, gradient, laplacian, entropy_disp):
        h, w = 350,350

        cam  = cv2.resize(camera, (w,h))
        grad = cv2.resize(gradient, (w,h))
        lap  = cv2.resize(laplacian, (w,h))
        ent  = cv2.resize(entropy_disp, (w,h))

        if len(grad.shape) == 2:
            grad = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        if len(lap.shape) == 2:
            lap  = cv2.cvtColor(lap,  cv2.COLOR_GRAY2BGR)
        if len(ent.shape) == 2:
            ent  = cv2.cvtColor(ent,  cv2.COLOR_GRAY2BGR)
        if len(cam.shape) == 2:
            cam  = cv2.cvtColor(cam,  cv2.COLOR_GRAY2BGR)

        cv2.putText(cam,  "Camera",    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(grad, "Gradient",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(lap,  "Laplacian", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(ent,  "Entropy",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Stitch into 2x2 grid
        top    = np.hstack([cam, grad])
        bottom = np.hstack([lap, ent])
        grid   = np.vstack([top, bottom])

        cv2.imshow("Texture Analysis", grid)


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
    
    
    def entropy_operation(self,img):
        """
        Compute local entropy over a grayscale image using a sliding window.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)
        entropy_img = entropy(img, disk(5)) 

        return entropy_img



    def clamp(self,x):
        """
        Clamp a value to the range [0, 1].
        """
        return max(0.0, min(1.0, x))

    def compute_texture_confidence(self, G_variance, laplacian_var, entropy_value, G_min, G_max, L_min, L_max, E_min, E_max):
        """
        Normalizes gradient variance, Laplacian variance, and entropy to [0, 1]
        Higher confidence -> strong texture (reliable for vision tasks)
        Lower confidence  -> weak texture (less reliable)
        """
        G_norm = self.clamp((G_variance - G_min)/(G_max - G_min))
        L_norm = self.clamp((laplacian_var - L_min)/(L_max - L_min))
        E_norm = self.clamp((entropy_value - E_min)/(E_max - E_min))

        confidence = (G_norm + L_norm + E_norm)/3.0

        return confidence

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

        # self.get_logger().info(f"Gradient variance: {G_variance:.2f}")


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
        # self.get_logger().info(f"Laplacian variance: {laplacian_var:.2f}")


        """
        Compute local entropy of the image to measure texture complexity.
        - High entropy  -> highly detailed / textured regions (good for optical flow)
        - Low entropy   -> uniform / flat regions (poor features)
        """
        entropy_img = self.entropy_operation(current_frame)
        entropy_display = cv2.normalize(entropy_img, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to uint8
        entropy_display = entropy_display.astype(np.uint8)

        entropy_value = np.mean(entropy_img)
        # self.get_logger().info(f"Entropy: {entropy_value:.2f}")


        # Logging confidence values
        confidence = self.compute_texture_confidence(G_variance, laplacian_var, entropy_value,G_min, G_max, L_min, L_max, E_min, E_max)
        # self.get_logger().info(f"Texture Confidence: {confidence:.2f}")


        # Publish the confidence values
        msg = Float32()
        msg.data = confidence
        self.confidence_pub_.publish(msg)
        # self.get_logger().info(f'Confidence: {msg.data}')

        # Display Image
        self.img_combined(current_frame, G_display, laplacian_display, entropy_display)
        cv2.waitKey(1)




def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
