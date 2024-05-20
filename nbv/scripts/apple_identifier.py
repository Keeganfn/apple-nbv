#!/usr/bin/env python3
# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rosidl_runtime_py.set_message import set_message_fields
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nbv_interfaces.msg import Apple, AppleArr, Ellipsoid
from geometry_msgs.msg import Pose, Point, Point32, Vector3, Quaternion
from std_msgs.msg import ColorRGBA

from visualization_msgs.msg import Marker, MarkerArray

from sklearn.cluster import KMeans

from toolz import functoolz as ft

import matplotlib.pyplot as plt


import numpy as np

from typing import Dict, List, Any, Union


class AppleIdentifier(Node):
    def __init__(self) -> None:
        super().__init__(node_name="apple_identifier")
        self.info_logger = lambda x: self.get_logger().info(f"{x}")

        # Subscribers
        self._sub_octomap_pc = self.create_subscription(
            msg_type=PointCloud2, topic="/filtered_apple_points", callback=self._sub_cb_octomap_pc, qos_profile=1
        )

        # Publishers
        self._pub_apples = self.create_publisher(msg_type=AppleArr, topic=f"{self.get_name()}/apples", qos_profile=1)
        self._pub_apple_markers = self.create_publisher(
            msg_type=MarkerArray, topic=f"{self.get_name()}/apple_markers", qos_profile=1
        )

        # Timers
        self._timer_pub_apples = self.create_timer(timer_period_sec=1 / 10, callback=self._timer_cb_pub_apples)

        self.k_means = KMeans(5)  # TODO: how to we get the number of clusters dynamically?
        self.k_means_params = self.k_means.get_params()

        self.apples = None
        return

    def _sub_cb_octomap_pc(self, msg: PointCloud2) -> None:
        """Subscriber to the filtered point cloud from image_processor.py"""
        pc = pc2.read_points_numpy(msg, ["x", "y", "z"])
        self.apples = self.identify_apples(pc)
        return

    def _timer_cb_pub_apples(self) -> None:
        if self.apples is None:
            return
        msg_apples = AppleArr()
        msg_apple_markers = MarkerArray()

        for i, apple in enumerate(self.apples):
            # Apple message
            msg_apple = Apple()
            msg_ellipsoid = Ellipsoid()

            center = apple["ellipsoid"]["center"]
            radii = apple["ellipsoid"]["radii"]
            msg_ellipsoid.center = Point(x=center[0], y=center[1], z=center[2])
            msg_ellipsoid.evecs = [Vector3(x=evec[0], y=evec[1], z=evec[2]) for evec in apple["ellipsoid"]["evecs"]]
            msg_ellipsoid.radii = Vector3(x=radii[0], y=radii[1], z=radii[2])

            msg_apple.cluster_num = apple["cluster_num"]
            msg_apple.points = [
                Point32(x=point[0].item(), y=point[1].item(), z=point[2].item()) for point in apple["points"]
            ]
            msg_apple.ellipsoid = msg_ellipsoid

            msg_apples.apples.append(msg_apple)

            # Marker message
            msg_marker = Marker()
            msg_marker.header.frame_id = "camera_link_optical"
            msg_marker.header.stamp = self.get_clock().now().to_msg()
            msg_marker.id = i
            msg_marker.type = Marker.SPHERE
            msg_marker.pose = Pose(
                position=Point(x=center[0], y=center[1], z=center[2]),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0),
            )
            msg_marker.scale = Vector3(x=radii[0] * 10, y=radii[1] * 10, z=radii[2] * 10)
            msg_marker.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=1.0)
            msg_apple_markers.markers.append(msg_marker)

        # Publish the messages
        self._pub_apples.publish(msg_apples)
        self._pub_apple_markers.publish(msg_apple_markers)

        return

    def run_k_means(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Run k-means on the point cloud data so that"""
        apples = []
        k_means_fit = self.k_means.fit(data)
        predict_data = k_means_fit.predict(data)
        centers = k_means_fit.cluster_centers_
        for cluster_num in range(self.k_means_params["n_clusters"]):
            apple_points = [data[i] for i in range(len(data)) if predict_data[i] == cluster_num]
            apple_dict = {"cluster_num": cluster_num, "points": apple_points, "center": centers[cluster_num]}
            apples.append(apple_dict)

        return apples

    def fit_ellipsoid(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fit ellipsoids to the data
        https://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
        https://github.com/aleksandrbazhin/ellipsoid_fit_python/blob/master/ellipsoid_fit.py

        Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx +
        2Hy + 2Iz + J = 0 and A + B + C = 3 constraint removing one extra
        parameter
        """
        for apple in data:
            points = apple["points"]
            x = points[:][0]
            y = points[:][1]
            z = points[:][2]
            D = np.array(
                [
                    x ** 2 + y ** 2 - 2 * z ** 2,
                    x ** 2 + z ** 2 - 2 * y ** 2,
                    2 * x * y,
                    2 * x * z,
                    2 * y * z,
                    2 * x,
                    2 * y,
                    2 * z,
                    1 - 0 * x,
                ]
            )
            # Solve the normal system of equations
            d2 = np.array(x ** 2 + y ** 2 + z ** 2).T  # the RHS of the llsq problem (y's)
            u = np.linalg.solve(D.dot(D.T), D.dot(d2))
            a = np.array([u[0] + 1 * u[1] - 1])
            b = np.array([u[0] - 2 * u[1] - 1])
            c = np.array([u[1] - 2 * u[0] - 1])
            v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
            A = np.array(
                [[v[0], v[3], v[4], v[6]], [v[3], v[1], v[5], v[7]], [v[4], v[5], v[2], v[8]], [v[6], v[7], v[8], v[9]]]
            )

            center = np.linalg.solve(-A[:3, :3], v[6:9])

            translation_matrix = np.eye(4)
            translation_matrix[3, :3] = center.T

            R = translation_matrix.dot(A).dot(translation_matrix.T)

            evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
            evecs = evecs.T

            radii = np.sqrt(1.0 / np.abs(evals))
            radii *= np.sign(evals)

            apple["ellipsoid"] = {"center": center + apple["center"], "evecs": evecs, "radii": radii, "v": v}

        return data

    def filter_ellipsoid(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out the ellipsoids with bad spherical fits"""
        apples_filtered = []
        for apple in data:
            radii_std = np.std(apple["ellipsoid"]["radii"])
            if radii_std > 0.1:
                continue
            else:
                apples_filtered.append(apple)
        return apples_filtered

    def identify_apples(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Pipe the data through a series of methods to get the apple data"""
        apples = ft.pipe(data, self.run_k_means, self.fit_ellipsoid, self.filter_ellipsoid)
        return apples


def main():
    rclpy.init()
    ai = AppleIdentifier()
    rclpy.spin(ai, executor=MultiThreadedExecutor())
    ai.destroy_node()
    rclpy.shutdown()
    return


if __name__ == "__main__":
    main()
