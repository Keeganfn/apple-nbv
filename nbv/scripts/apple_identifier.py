#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

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
            msg_type=PointCloud2,
            topic="/octomap_point_cloud_centers",
            callback=self._sub_cb_octomap_pc,
            qos_profile=1
        )

        # Publishers
        self._pub_k_means = self.create_publisher(
            msg_type=PointCloud2,
            topic=f"{self.get_name()}/k_means",
            qos_profile=1
        )

        # Timers
        self._timer_k_means = self.create_timer(
            timer_period_sec=1/20,
            callback=self._timer_cb_pub_k_means
        )

        self.k_means = KMeans()
        self.k_means_params = self.k_means.get_params()
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        return
    
    def _sub_cb_octomap_pc(self, msg: PointCloud2):
        pc = pc2.read_points_numpy(msg, ["x", "y", "z"])
        apples = self.identify_apples(pc)
        
        # self.info_logger(apples)
        return
    
    def _timer_cb_pub_k_means(self):
        msg = PointCloud2()
        # self._pub_k_means.publish(msg=apples)
        return

    def run_k_means(self, _data: PointCloud2) -> List[Dict[str, Any]]:
        # fit_data = self.k_means.fit(_data)
        apples = []
        predict_data = self.k_means.fit_predict(_data)
        # zipped_data = zip(_data, predict_data)
        # self.plot_kmeans(_data=_data)
        for cluster_num in range(self.k_means_params["n_clusters"]):
            apple_points = [_data[i] for i in range(len(_data)) if predict_data[i] == cluster_num]
            apple_dict = {"cluster_num": cluster_num, "points": apple_points}
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
            D = np.array([
                x**2 + y**2 - 2 * z**2,
                x**2 + z**2 - 2 * y**2,
                2 * x * y,
                2 * x * z,
                2 * y * z,
                2 * x,
                2 * y,
                2 * z,
                1 - 0 * x
            ])
            # Solve the normal system of equations
            d2 = np.array(x**2 + y**2 + z**2).T  # the RHS of the llsq problem (y's)
            u = np.linalg.solve(D.dot(D.T), D.dot(d2))
            a = np.array([u[0] + 1 * u[1] - 1])
            b = np.array([u[0] - 2 * u[1] - 1])
            c = np.array([u[1] - 2 * u[0] - 1])
            v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
            A = np.array([[v[0], v[3], v[4], v[6]],
                        [v[3], v[1], v[5], v[7]],
                        [v[4], v[5], v[2], v[8]],
                        [v[6], v[7], v[8], v[9]]])

            center = np.linalg.solve(- A[:3, :3], v[6:9])

            translation_matrix = np.eye(4)
            translation_matrix[3, :3] = center.T

            R = translation_matrix.dot(A).dot(translation_matrix.T)

            evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
            evecs = evecs.T

            radii = np.sqrt(1. / np.abs(evals))
            radii *= np.sign(evals)

            apple["ellipsoid"] = {
                "center": center,
                "evecs": evecs,
                "radii": radii,
                "v": v
            }

        return data
    
    def filter_ellipsoid(self, data: List[Dict[str, Any]]):
        """Filter out the ellipsoids with bad spherical fits"""
        apples_filtered = []
        for apple in data:
            radii_std = np.std(apple["ellipsoid"]["radii"])
            if radii_std > 0.1:
                continue
            else:
                apples_filtered.append(apple)            
        return apples_filtered
    
    def identify_apples(self, data: np.ndarray):
        res = ft.pipe(
            data,
            self.run_k_means,
            self.fit_ellipsoid,
            self.filter_ellipsoid
        )
        return res
    
    # def plot_kmeans(self, _data=None):
    #     # plt.axis([0, 10, 0, 1])
    #     data = _data
    #     self.ax.scatter(data[:,0], data[:,1], data[:,2])
    #     plt.pause(0.05)
        
    #     return


def main():
    rclpy.init()
    ai = AppleIdentifier()
    rclpy.spin(ai, executor=MultiThreadedExecutor())
    ai.destroy_node()
    rclpy.shutdown()
    return


if __name__ == "__main__":
    main()

