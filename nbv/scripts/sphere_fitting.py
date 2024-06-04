#!/usr/bin/env python3

from utils.sphere import Sphere

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Float64
from nbv_interfaces.srv import MoveArm

import numpy as np
import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt

from toolz import functoolz as ft


class SphereFitting(Node):
    def __init__(self, num_bins: int = 8) -> None:
        super().__init__(node_name="sphere_processing")
        self.info_logger = lambda x: self.get_logger().info(f"{x}")

        # Subscribers
        self._sub_octomap_pc = self.create_subscription(
            msg_type=PointCloud2, topic="/filtered_apple_points", callback=self._sub_cb_octomap_pc, qos_profile=1
        )
        self.filtered_pc: np.ndarray

        # Service clients
        self._srv_move_arm = self.create_client(
            srv_name="/move_arm",
            srv_type=MoveArm,
        )

        # Timers
        self._timer_pub_apple_bins = self.create_timer(
            timer_period_sec=1.0,
            callback=self._timer_cb_pub_apple_bins
        )

        # NBV bin data
        self.theta_bin = np.linspace(-np.pi, np.pi, num_bins+1)
        self.phi_bin = np.linspace(0, np.pi, 2)
        return
    
    def _sub_cb_octomap_pc(self, msg: PointCloud2) -> None:
        """Subscriber to the filtered point cloud from image_processor.py"""
        self.filtered_pc = pc2.read_points_numpy(msg, ["x", "y", "z"])
        return
    
    def _timer_cb_pub_apple_bins(self) -> None:
        """Timer callback for publishing the binned apple data"""
        # self._pub_apple_bins.publish()
        return
    
    def xy_clustering(self, cloud, graph=False):
        x=cloud[:,0][0::100]
        y=cloud[:,1][0::100]
        data=np.dstack((x,y))[0]
        thresh = .02
        clusters = hcluster.fclusterdata(data, thresh, criterion="distance")
        if graph:
            plt.scatter(*np.transpose(data), c=clusters)
            plt.axis("equal")
            plt.show()
        return data, clusters
    
    def get_cluster_center(self, data, clusters):
        stacked_array=[]
        for i in range(len(data)):
            stacked_array.append([data[i][0],data[i][1], clusters[i]])
        cluster_numbers=np.unique(clusters)
        cluster_centers=[]
        stacked_array=np.array(stacked_array)
        for number in cluster_numbers:
            mask=(stacked_array[:,2]==number)
            x=stacked_array[mask,0]
            y = stacked_array[mask, 1]
            x_average=np.mean(x)
            y_average=np.mean(y)
            #range to use for the rest of the pixels (not sampled originally
            y_min=np.min(y)
            y_max=np.max(y)
            y_range=((y_max-y_min)*1.5)/2
            x_min=np.min(x)
            x_max=np.max(x)
            x_range = ((x_max - x_min) * 1.5) / 2
            cluster_centers.append([x_average,y_average, x_range, y_range])
        return cluster_centers
    
    def upsample(self, cloud, cluster_centers, graph=False):
        cloud_list=[[],[],[],[],[]]
        x=cloud[:,0]
        y=cloud[:,1]
        z = cloud[:, 2]
        data = np.dstack((x, y,z))[0]
        #empty array to put designated cluster in
        clusters=[]
        for point in data:
            for i in range(len(cluster_centers)):
                cluster=cluster_centers[i]
                if point[0]>cluster[0]-cluster[2] and point[0]<cluster[0]+cluster[2] and point[1]>cluster[1]-cluster[3] and point[1]<cluster[1]+cluster[3]:
                    clusters.append(i)
                    cloud_list[i].append(point)
                    break
        #print(cloud_list)
        if graph:
            plt.scatter(*np.transpose(data), c=clusters)
            plt.axis("equal")
            plt.show()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #cloud=np.array(cloud_list[1])
            #x = cloud[:, 0]
            #y = cloud[:, 1]
            #z = cloud[:, 2]
            ax.scatter(x, y, z, zdir='z', c=clusters)
            plt.show()
        return cloud_list

    def sphereFit(self, spX,spY,spZ) -> Sphere:
        #   Assemble the A matrix
        spX = np.array(spX)
        spY = np.array(spY)
        spZ = np.array(spZ)
        A = np.zeros((len(spX),4))
        A[:,0] = spX*2
        A[:,1] = spY*2
        A[:,2] = spZ*2
        A[:,3] = 1

        #   Assemble the f matrix
        f = np.zeros((len(spX),1))
        f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
        C, residules, rank, singval = np.linalg.lstsq(A,f)

        #   solve for the radius
        t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
        radius = np.sqrt(t)

        sphere = Sphere(radius=radius, center_x=C[0], center_y=C[1], center_z=C[2])

        return sphere
    
    def get_spheres(self, cloud_list):
        spheres = []

        for cloud in cloud_list:
            # Fit a sphere to the cloud
            cloud=np.array(cloud)
            sphere=self.sphereFit(cloud[:,0],cloud[:,1],cloud[:,2])

            # Find theta and phi for each point in the cloud with respect to the center of the sphere.
            cloud_xyz = np.subtract(cloud, sphere.center.T)
            cloud_thetas = np.arctan2(cloud_xyz[:,2], cloud_xyz[:,0])

            cloud_phis = np.arctan2(cloud_xyz[:,2], cloud_xyz[:,0])
            cloud_phis = np.where(cloud_phis >= 0, cloud_phis, cloud_phis + np.pi)

            # Get binned theta values
            theta_binned = np.digitize(cloud_thetas, self.theta_bin)
            theta_unique, theta_counts = np.unique(theta_binned, return_counts=True)
            sphere.theta_bin_counts = dict(zip(theta_unique, theta_counts))

            # Get binned phi values
            phi_binned = np.digitize(cloud_phis, self.phi_bin)
            phi_unique, phi_counts = np.unique(phi_binned, return_counts=True)
            sphere.phi_bin_counts = dict(zip(phi_unique, phi_counts))

            spheres.append(sphere)
        return spheres
    
    def get_nbv_vec(self, cloud):
        # ft.pipe(
        #     cloud,
        # )
        data, clusters = self.xy_clustering(cloud)
        cluster_centers = self.get_cluster_center(data, clusters)
        cloud_list = self.upsample(cloud, cluster_centers)
        spheres = self.get_spheres(cloud_list=cloud_list)
        return

def main():
    rclpy.init()
    sphere_fitting = SphereFitting()
    rclpy.spin(node=sphere_fitting, executor=MultiThreadedExecutor())
    sphere_fitting.destroy_node()
    rclpy.shutdown()
    return


if __name__ == "__main__":
    main()