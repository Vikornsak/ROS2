import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import datetime

class ShapeDetectorNode(Node):

    def __init__(self):
        super().__init__('shape_detector_node')
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(String, '/detected_shape', 10)
        self.servo_pub = self.create_publisher(Int32, 'servo_angle', 10)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Shape Detector Node has been started.")

    def detect_shapes(self, cv_image):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

            if len(approx) == 4:
                shape = "Square"
                detection_result = f"Square (2) detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                self.get_logger().info(detection_result)
                self.publisher_.publish(String(data="Square"))
            elif len(approx) > 4:
                shape = "Circle"
                detection_result = f"Circle (3) detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                self.get_logger().info(detection_result)
                self.publisher_.publish(String(data="Circle"))
                # Publish 90-degree angle for servo to rotate
                self.servo_pub.publish(Int32(data=90))
            else:
                shape = "Unknown"
                detection_result = f"Unknown shape detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                self.get_logger().info(detection_result)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info("Image received from camera.")
            self.detect_shapes(cv_image)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
