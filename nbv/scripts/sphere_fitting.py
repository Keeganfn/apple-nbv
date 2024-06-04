#!/usr/bin/env python3

from utils.sphere import Sphere

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose, Point, Quaternion
from nbv_interfaces.srv import MoveArm, RunNBV


import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from toolz import functoolz as ft
from typing import Union

from objprint import op


class SphereFitting(Node):
    def __init__(self, num_bins: int = 8) -> None:
        super().__init__(node_name="sphere_processing")
        self.info_logger = lambda x: self.get_logger().info(f"{x}")
        self.warn_logger = lambda x: self.get_logger().warn(f"{x}")

        # Subscribers
        self._sub_octomap_binary = self.create_subscription(
            msg_type=PointCloud2, topic="/octomap_point_cloud_centers", callback=self._sub_cb_octomap_pc, qos_profile=1
        )
        self.filtered_pc: np.ndarray

        # Services servers
        self._svr_run_nbv = self.create_service(
            srv_type=RunNBV,
            srv_name="run_nbv",
            callback=self._svr_cb_run_nbv
        )

        # Service clients
        self._srv_move_arm = self.create_client(
            srv_name="/move_arm",
            srv_type=MoveArm,
        )

        # # Timers
        # self._timer_pub_apple_bins = self.create_timer(
        #     timer_period_sec=1.0,
        #     callback=self._timer_cb_pub_apple_bins
        # )

        # NBV bin data
        self.num_bins = num_bins
        self.theta_bin = np.linspace(-np.pi, np.pi, num_bins + 1)
        self.phi_bin = np.linspace(0, np.pi, 2 + 1)
        return
    
    def _svr_cb_run_nbv(self, request, response):
        self.info_logger("RUN NBV service requested")
        try:
            xyz, quat = self.get_nbv_vec(cloud=self.filtered_pc)
            
            self.warn_logger(f'{xyz}, {quat}')
            pose = Pose(
                position=Point(x=xyz[0], y=xyz[1] ,z=xyz[2]),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            )
            move_arm = MoveArm.Request()
            move_arm.goal = pose

            result = self._srv_move_arm.call(request=move_arm)
            self.warn_logger(result)

            response.success = True
        except Exception as e:
            response.success = False
            self.warn_logger(e)
        return response


    def _sub_cb_octomap_pc(self, msg: PointCloud2) -> None:
        """Subscriber to the filtered point cloud from image_processor.py"""
        self.filtered_pc = pc2.read_points_numpy(msg, ["x", "y", "z"])        
        return

    # def _timer_cb_pub_apple_bins(self) -> None:
    #     """Timer callback for publishing the binned apple data"""
    #     # self._pub_apple_bins.publish()
    #     return

    def xy_clustering(self, cloud, graph=False):
        x = cloud[:, 0]  # [0::100]
        y = cloud[:, 2]  # [0::100]
        data = np.dstack((x, y))[0]
        thresh = 0.015
        clusters = hcluster.fclusterdata(data, thresh, criterion="distance")
        if graph:
            plt.scatter(*np.transpose(data), c=clusters)
            plt.axis("equal")
            plt.show()
        return data, clusters

    def get_cluster_center(self, data, clusters):
        stacked_array = []
        for i in range(len(data)):
            stacked_array.append([data[i][0], data[i][1], clusters[i]])
        cluster_numbers = np.unique(clusters)
        cluster_centers = []
        stacked_array = np.array(stacked_array)
        for number in cluster_numbers:
            mask = stacked_array[:, 2] == number
            x = stacked_array[mask, 0]
            y = stacked_array[mask, 1]
            x_average = np.mean(x)
            y_average = np.mean(y)
            # range to use for the rest of the pixels (not sampled originally
            y_min = np.min(y)
            y_max = np.max(y)
            y_range = ((y_max - y_min) * 1.5) / 2
            x_min = np.min(x)
            x_max = np.max(x)
            x_range = ((x_max - x_min) * 1.5) / 2
            cluster_centers.append([x_average, y_average, x_range, y_range])
        return cluster_centers

    def upsample(self, cloud, cluster_centers):
        cloud_list = [[], [], [], [], []]
        x = cloud[:, 0]
        y = cloud[:, 2]
        z = cloud[:, 1]
        data = np.dstack((x, y, z))[0]

        # empty array to put designated cluster in
        clusters = []
        for point in data:
            for i in range(len(cluster_centers)):
                # self.info_logger(cluster_centers)
                cluster = cluster_centers[i]

                if (
                    point[0] > cluster[0] - cluster[2]
                    and point[0] < cluster[0] + cluster[2]
                    and point[1] > cluster[1] - cluster[3]
                    and point[1] < cluster[1] + cluster[3]
                ):
                    clusters.append(i)
                    cloud_list[i].append(point)
                    break
        return cloud_list

    def sphereFit(self, spX, spY, spZ) -> Sphere:
        #   Assemble the A matrix
        spX = np.array(spX)
        spY = np.array(spY)
        spZ = np.array(spZ)
        A = np.zeros((len(spX), 4))
        A[:, 0] = spX * 2
        A[:, 1] = spY * 2
        A[:, 2] = spZ * 2
        A[:, 3] = 1

        #   Assemble the f matrix
        f = np.zeros((len(spX), 1))
        f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
        C, residules, rank, singval = np.linalg.lstsq(A, f)

        #   solve for the radius
        t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
        radius = np.sqrt(t)

        sphere = Sphere(radius=radius, center_x=C[0], center_y=C[1], center_z=C[2])

        return sphere

    def get_spheres(self, cloud_list) -> Union[list, None]:
        spheres = np.empty(len(cloud_list), dtype=Sphere)

        for i, cloud in enumerate(cloud_list):
            if not cloud:
                continue
            # Fit a sphere to the cloud
            cloud = np.array(cloud)
            sphere = self.sphereFit(cloud[:, 0], cloud[:, 1], cloud[:, 2])
            # Find theta and phi for each point in the cloud with respect to the center of the sphere.
            cloud_xyz = np.subtract(cloud, sphere.center.T)
            cloud_thetas = np.arctan2(cloud_xyz[:, 2], cloud_xyz[:, 0])

            cloud_phis = np.arctan2(cloud_xyz[:, 2], cloud_xyz[:, 0])
            cloud_phis = np.where(cloud_phis >= 0, cloud_phis, cloud_phis + np.pi)

            # Get binned theta values
            theta_binned = np.digitize(cloud_thetas, self.theta_bin)
            theta_unique, theta_counts = np.unique(theta_binned, return_counts=True)
            sphere.theta_bin_counts = dict(zip(theta_unique, theta_counts))

            # Get binned phi values
            phi_binned = np.digitize(cloud_phis, self.phi_bin)
            phi_unique, phi_counts = np.unique(phi_binned, return_counts=True)
            sphere.phi_bin_counts = dict(zip(phi_unique, phi_counts))

            for theta_bin_num in range(1, self.num_bins + 1):
                for phi_bin_num in range(1, 3):
                    current_phi_bin = phi_binned == phi_bin_num
                    current_theta_bin = theta_binned == theta_bin_num
                    current_bin = current_theta_bin & current_phi_bin
                    if phi_bin_num == 1:
                        full_bin_num = theta_bin_num
                    else:
                        full_bin_num = theta_bin_num + self.num_bins
                    # these are left bin angles
                    theta_angle = self.theta_bin[theta_bin_num - 1]
                    phi_angle = self.phi_bin[phi_bin_num - 1]
                    sphere.bins[full_bin_num] = [np.sum(current_bin), theta_angle, phi_angle]
                    # print(np.sum(current_bin), 'phi bin: ',phi_bin_num,'theta bin: ', theta_bin_num)

            # Add the sphere to our list
            spheres[i] = sphere

        return spheres
    
    def get_vector(self, spheres):
        #subject to change (maybe select based on center? or fully explored property?
        sphere=spheres[0]
        #get the front bins only
        #not a prettier way to do it as far as I can tell, wont  let me use sphere.bins[1:4,9:12]
        front_bins={}
        for i in range(1,8+1):
            if i<5:
                _bin=sphere.bins[i]
            else:
                _bin=sphere.bins[i+4]
            front_bins[i]=_bin
        #do we need a threshold for filled bins?? x number of points in each bin before going to next apple
        _bin=min(front_bins, key=front_bins.get)
        full_bin=sphere.bins[_bin]
        #I think this is in global frame?? (z up, x left to right, y into page) need to check this frame
        #from https://stackoverflow.com/questions/30011741/3d-vector-defined-by-2-angles
        theta=full_bin[1]+2*np.pi/self.num_bins
        phi=full_bin[2]+np.pi/4
        x=np.sin(theta)*np.cos(phi)
        y=-np.cos(theta)*np.cos(phi)
        #have to offset to get the bottom half of the sphere to be negative
        z=np.sin(phi-np.pi/2)
        unit_vector=np.array([x,y,z])
        camera_orientation=-1*unit_vector
        #subject to change, uses the y distance to center sphere from scan (may be unreachable, may want to use different approach)
        camera_coords=[sphere.center_x[0], sphere.center_y[0], sphere.center_z[0]]+unit_vector*sphere.center_y
        camera_coords=np.array(camera_coords)
        coord_radius=np.sqrt(camera_coords[0]**2+camera_coords[1]**2+camera_coords[2])
        #if trying to move out of 90% of max reach
        if coord_radius>.85*.9:
            scaling=(.85*.9)/coord_radius
            camera_coords=camera_coords*scaling
            #adjust orienation
            new_coords_to_center=[sphere.center_x[0], sphere.center_y[0], sphere.center_z[0]]-camera_coords
            #new camera orientation is unit vector pointed at center
            #get length of vector
            orientation_len=np.sqrt(new_coords_to_center[0]**2+new_coords_to_center[1]**2+new_coords_to_center[2]**2)
            camera_orientation=[new_coords_to_center]/orientation_len

        camera_orientation = Rotation.from_euler(seq='xyz', angles=camera_orientation, degrees=False).as_quat()[0]
        return camera_coords, camera_orientation

    def get_nbv_vec(self, cloud):
        # ft.pipe(
        #     cloud,
        # )
        data, clusters = self.xy_clustering(cloud)
        cluster_centers = self.get_cluster_center(data, clusters)
        cloud_list = self.upsample(cloud, cluster_centers)
        spheres = self.get_spheres(cloud_list=cloud_list)
        camera_coords, camera_orientation = self.get_vector(spheres)
        return (camera_coords, camera_orientation)


def main():
    rclpy.init()
    sphere_fitting = SphereFitting()
    rclpy.spin(node=sphere_fitting, executor=MultiThreadedExecutor())
    sphere_fitting.destroy_node()
    rclpy.shutdown()
    return


if __name__ == "__main__":
    main()
