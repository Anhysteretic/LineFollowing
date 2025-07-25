#!/usr/bin/env python


###########
# IMPORTS #
###########
import numpy as np
import rclpy
from rclpy.node import Node
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from line_interfaces.msg import Line
import sys


#############
# CONSTANTS #
#############
LOW = 200  # Lower image thresholding bound
HI = 255   # Upper image thresholding bound
LENGTH_THRESH = 200  # If the length of the largest contour is less than LENGTH_THRESH, we will not consider it a line
KERNEL = np.ones((5, 5), np.uint8)
DISPLAY = True
float0 = (float(0), float(0), float(0), float(0))


class LineDetector(Node):
   def __init__(self):
       super().__init__('detector')


       # A subscriber to the topic '/aero_downward_camera/image'
       self.camera_sub = self.create_subscription(
           Image,
           '/world/line_following_track/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image',
           self.camera_sub_cb,
           10
       )


       # A publisher which will publish a parametrization of the detected line to the topic '/line/param'
       self.param_pub = self.create_publisher(Line, '/line/param', 1)


       # A publisher which will publish an image annotated with the detected line to the topic 'line/detector_image'
       self.detector_image_pub = self.create_publisher(Image, '/line/detector_image', 1)


       # Initialize instance of CvBridge to convert images between OpenCV images and ROS images
       self.bridge = CvBridge()


   ######################
   # CALLBACK FUNCTIONS #
   ######################
   def camera_sub_cb(self, msg):
       """
       Callback function which is called when a new message of type Image is received by self.camera_sub.
       """
       # Convert Image msg to OpenCV image
       image = self.bridge.imgmsg_to_cv2(msg, "mono8")


       # Detect line in the image. detect returns a parameterize the line (if one exists)
       line = self.detect_line(image)


       # If a line was detected, publish the parameterization to the topic '/line/param'
       if line is not None:
           msg = Line()
           msg.x, msg.y, msg.vx, msg.vy = line
           self.param_pub.publish(msg)


       # Publish annotated image if DISPLAY is True and a line was detected
       if DISPLAY and line is not None:
           annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
           x, y, vx, vy = line
           pt1 = (int(x - 100*vx), int(y - 100*vy))
           pt2 = (int(x + 100*vx), int(y + 100*vy))
           cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)
           cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
           cv2.putText(annotated, 'A', pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
           cv2.putText(annotated, 'B', pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
           # Optionally, draw an arrow to indicate direction from A to B
           cv2.arrowedLine(annotated, pt1, pt2, (0, 0, 255), 2, tipLength=0.1)
           annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
           self.detector_image_pub.publish(annotated_msg)


   ##########
   # DETECT #
   ##########
   def detect_line(self, image):
       if LOW is None or HI is None:
           # Default to 200-255 for white
           low = 200
           hi = 255
       else:
           low = LOW
           hi = HI


       thresh = cv2.inRange(image, low, hi)
       # Morphological operations to clean up noise
       thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL)


       # Find contours
       contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       if not contours:
           return float0


       # Find largest contour
       largest = max(contours, key=cv2.contourArea)
       if LENGTH_THRESH is not None and cv2.arcLength(largest, False) < LENGTH_THRESH:
           return float0


       # Fit line to largest contour
       if len(largest) < 2:
           return float0


       [vx, vy, x, y] = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
       # vx, vy are normalized direction vector; x, y is a point on the line


       # Convert to float
       x = float(x)
       y = float(y)
       vx = float(vx)
       vy = float(vy)


       return (x, y, vx, vy)
  
def main(args=None):
   rclpy.init(args=args)
   detector = LineDetector()
   detector.get_logger().info("Line detector initialized")
   try:
       rclpy.spin(detector)
   except KeyboardInterrupt:
       print("Shutting down")
   except Exception as e:
       print(e)
   finally:
       detector.destroy_node()
       rclpy.shutdown()


if __name__ == '__main__':
   main()