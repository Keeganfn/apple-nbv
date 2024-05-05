#!/usr/bin/env python3

# ROS
import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Interfaces
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, TwistStamped

# Image processing
from cv_bridge import CvBridge
import cv2
import numpy as np

# TF2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import tf2_geometry_msgs


class ImageProcessing(Node):

    def __init__(self):
        super().__init__('image_processing_node')
        ### Subscribers/ Publishers
        self.rgb_sub = Subscriber(self,Image,"/camera2/image_raw")
        self.points_sub = Subscriber(self,PointCloud2,"/camera2/points")
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.points_sub],30,0.05,)
        self.ts.registerCallback(self.rgbd_callback)
        # Publisher to end effector servo controller, sends velocity commands
        self.filtered_points_publisher = self.create_publisher(PointCloud2, "filtered_apple_points", 10)

        ### Image Processing
        self.br = CvBridge()

        ### Tf2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    def rgbd_callback(self, rgb, points):
            # Convert to opencv format from msg
            image = self.br.imgmsg_to_cv2(rgb, "bgr8")
            pc = pc2.read_points_numpy(points, ["x", "y", "z"]).reshape((480, 640, 3))
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Define the range of red color in HSV
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            # Threshold the HSV image to get only red colors
            color_mask = cv2.inRange(hsv, lower_red, upper_red)
            apple_mask = np.array(color_mask, dtype=bool)
            pc = pc[apple_mask]
            pc = pc2.create_cloud_xyz32(points.header, pc)
            self.filtered_points_publisher.publish(pc)
            
            # NOT NEEDED RIGHT NOW FOR BASIC RED CIRCLE
            # Find contours that match criteria
            # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # cv2.imshow("test", mask)
            # cv2.waitKey(1)
            




def main(args=None):
    rclpy.init(args=args)
    img_proc = ImageProcessing()
    executor = MultiThreadedExecutor()
    rclpy.spin(img_proc, executor=executor)
    rclpy.shutdown()


if __name__ == '__main__':
    main()