#!/usr/bin/env python3

from utils.sphere import Sphere

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Pose, Point, Quaternion
from nbv_interfaces.srv import MoveArm, RunNBV, SetSphereConstraint
from nbv_interfaces.msg import VolumeEstimatesArray

import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from threading import Event

from typing import Union, List

import secrets
import open3d as o3d
import pyransac3d as pyrsc


class SphereFitting(Node):
    def __init__(self, num_bins: int = 8) -> None:
        super().__init__(node_name="sphere_processing")
        self.info_logger = lambda x: self.get_logger().info(f"{x}")
        self.warn_logger = lambda x: self.get_logger().warn(f"{x}")
        self.err_logger = lambda x: self.get_logger().error(f"{x}")

        # Subscribers
        self._sub_octomap_binary = self.create_subscription(
            msg_type=PointCloud2, topic="/octomap_point_cloud_centers", callback=self._sub_cb_octomap_pc, qos_profile=1
        )
        self.filtered_pc: np.ndarray

        # Publishers
        self._spheres: list = []
        self._pub_volume_estimates = self.create_publisher(
            msg_type=VolumeEstimatesArray,
            topic="apples/volume_estimates",
            qos_profile=1
        )

        # Timers
        self._timer_volume_estimates = self.create_timer(
            timer_period_sec=0.1,
            callback=self._timer_cb_volume_estimates
        )

        # Services servers
        # self._srv_cb_group_move_arm_group = ReentrantCallbackGroup()
        self.reentrant_cb_group = ReentrantCallbackGroup()
        self._srv_move_arm_done_event = Event()
        self._srv_set_sphere_constraint_done_event = Event()

        self._svr_run_nbv = self.create_service(
            srv_type=RunNBV,
            srv_name="run_nbv",
            callback=self._svr_cb_run_nbv,
            callback_group=self.reentrant_cb_group
        )

        # Service clients
        self._srv_move_arm = self.create_client(
            srv_name="/move_arm",
            srv_type=MoveArm,
            callback_group=self.reentrant_cb_group
        )
        self._srv_set_sphere_constraint = self.create_client(
            srv_name="/set_sphere_constraint",
            srv_type=SetSphereConstraint,
            callback_group=self.reentrant_cb_group
        )

        # NBV bin data
        self.num_bins = num_bins
        self.theta_bin = np.linspace(-np.pi, np.pi, num_bins + 1)
        self.phi_bin = np.linspace(0, np.pi, 2 + 1)

        self._target_sphere: Sphere = None
        self.nbv_first_run_done = False
        return
    
    def _timer_cb_volume_estimates(self):
        msg = VolumeEstimatesArray()
        msg.data = [sphere.volume_estimate for sphere in self._spheres if sphere is not None]
        self._pub_volume_estimates.publish(msg)
        return
    
    def _svr_cb_run_nbv(self, request, response):
        self.info_logger("RUN NBV service requested")
        
        self._srv_move_arm_done_event.clear()
        self._srv_set_sphere_constraint_done_event.clear()

        self.inner_response = None


        # Step 1: Remove any old collision objects from last run, if any
        if self._spheres:
            for sphere in self._spheres:
                try:
                    sphere_constraint = SetSphereConstraint.Request()
                    sphere_constraint.id = str(sphere.id)
                    sphere_constraint.radius = float(sphere.radius)
                    sphere_constraint.pose = Pose(
                        position=Point(
                            x=float(sphere.center_x),
                            y=float(sphere.center_y),
                            z=float(sphere.center_z)
                        ),
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                    )
                    sphere_constraint.remove_from_scene = True

                    future = self._srv_set_sphere_constraint.call_async(request=sphere_constraint)
                    future.add_done_callback(self.set_sphere_constraint_inner_callback)
                    self._srv_set_sphere_constraint_done_event.wait()

                except Exception as e:
                    response.success = False
                    self.err_logger(f"Failed to remove collision object with sphere.id = {sphere.id}: {e}")

                self.inner_response = None
        

        # Step 2: Add the new collision objects calculated from `get_nbv_vec()`
        xyz, quat = self.get_nbv_vec(cloud=self.filtered_pc)

        self.info_logger(f"TARGET SPHERE: {self._target_sphere.radius}")
        self.info_logger(f"Sphere centers: \n{[sphere.center for sphere in self._spheres]}")

        for sphere in self._spheres:
            try:
                sphere_constraint = SetSphereConstraint.Request()
                sphere_constraint.id = str(sphere.id)
                sphere_constraint.radius = float(sphere.radius)
                sphere_constraint.pose = Pose(
                    position=Point(
                        x=float(sphere.center_x),
                        y=float(sphere.center_y),
                        z=float(sphere.center_z)
                    ),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                )
                sphere_constraint.remove_from_scene = False

                future = self._srv_set_sphere_constraint.call_async(request=sphere_constraint)
                future.add_done_callback(self.set_sphere_constraint_inner_callback)
                self._srv_set_sphere_constraint_done_event.wait()

            except Exception as e:
                response.success = False
                self.err_logger(f"Failed to add collision object with sphere.id = {sphere.id}: {e}")

            self.inner_response = None


        if not self._srv_move_arm.wait_for_service(timeout_sec=5.0):
            self.err_logger("Service unavailable.")
            response.success = False
            return response
        

        # Step 3: Move the arm to the calculated Pose
        try:
            pose = Pose(
                position=Point(x=xyz[0], y=xyz[1] ,z=xyz[2]),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            )
            move_arm = MoveArm.Request()
            move_arm.goal = pose
            future = self._srv_move_arm.call_async(request=move_arm)
            future.add_done_callback(self.move_arm_inner_callback)

            self._srv_move_arm_done_event.wait()
            if self.inner_response:
                response.success = True
            else:
                response.success = False
        except Exception as e:
            response.success = False
            self.warn_logger(e)
        return response
    
    def move_arm_inner_callback(self, future):
        try:
            self.inner_response = future.result()
        except Exception as e:
            self.err_logger(f"Move arm service call failed: {e}")
            self.inner_response = None
        finally:
            self._srv_move_arm_done_event.set()
        return
    
    def set_sphere_constraint_inner_callback(self, future):
        try:
            self.inner_response = future.result()
        except Exception as e:
            self.err_logger(f"Set sphere constraint service call failed: {e}")
            self.inner_response = None
        finally:
            self._srv_set_sphere_constraint_done_event.set()

    def _sub_cb_octomap_pc(self, msg: PointCloud2) -> None:
        """Subscriber to the filtered point cloud from image_processor.py"""
        self.filtered_pc = pc2.read_points_numpy(msg, ["x", "y", "z"])        
        return

    def _xy_clustering(self, cloud, graph=False):
        if self.nbv_first_run_done:
            thresh = 0.009
        else:
            thresh = 0.015
            self.nbv_first_run_done = True
        clusters = hcluster.fclusterdata(cloud, thresh, criterion="distance")
        if graph:
            plt.scatter(*np.transpose(data), c=clusters)
            plt.axis("equal")
            plt.show()
        return clusters
    
    def xy_clustering(self, cloud): 
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0

        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([pcd])

        self.info_logger(labels)

        pc_unlabeled = np.array(pcd.points)
        apple_cluster_points = []
        apple_spheres = []

        for i in range(max_label + 1):
            # Gives a Nx3 numpy array of all the apple points in mask i 
            label_mask = (labels == i)
            apple_pc = pc_unlabeled[label_mask]
            sph = pyrsc.Sphere()
            # center, radius, inliers = sph.fit(apple_pc, thresh=.001)
            threshs = [.01, .005, .001, .0001, .00001]
            for j in threshs:
                center, radius, inliers = sph.fit(apple_pc, thresh=j)
                if radius < .06 and radius > .04:
                    break

            self.info_logger(f"\nCluster: {i}\nCenter: {center}\nRadius: {radius}")

            apple_cluster_points.append(apple_pc)
            apple_spheres.append(Sphere(radius=radius, center_x=center[0], center_y=center[1], center_z=center[2]))
        return apple_cluster_points, apple_spheres

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

    def upsample(self, cloud, clusters):
        cloud_list = [ [] for _ in range(len(np.unique(clusters)))]
        x = cloud[:, 0]
        y = cloud[:, 2]
        z = cloud[:, 1]
        data = np.dstack((x, y, z))[0]

        # empty array to put designated cluster in
        # clusters = []
        for i, point in enumerate(data):
            cloud_list[clusters[i]-1].append(point)
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

    def get_spheres(self, cloud_list, spheres) -> Union[list, None]:
        # spheres = np.empty(len(cloud_list), dtype=Sphere)

        for i, cloud in enumerate(cloud_list):
            # if not cloud:
            #     continue
            # Fit a sphere to the cloud
            cloud = np.array(cloud)
            # sphere = self.sphereFit(cloud[:, 0], cloud[:, 1], cloud[:, 2])
            sphere = spheres[i]
            # Find theta and phi for each point in the cloud with respect to the center of the sphere.
            cloud_xyz = np.subtract(cloud, sphere.center.T)
            cloud_thetas = np.arctan2(cloud_xyz[:, 2], cloud_xyz[:, 0])

            cloud_phis = np.arctan2(cloud_xyz[:, 1], cloud_xyz[:, 0])
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

        if not self._spheres:
            pass
        else:
            old_spheres: List[Sphere] = self._spheres
            new_spheres: List[Sphere] = spheres

            # old_centers = [old_.center for old_ in old_spheres]

            # if np.allclose(old_spheres, new_spheres, rtol=0.05, atol=0.1):
                

            for n_sphere in new_spheres:
                n_center = n_sphere.center
                matched = False
                for i, o_sphere in enumerate(old_spheres):
                    o_center = o_sphere.center
                    # check if they're close, change tolerance if necessary
                    close = np.isclose(o_center, n_center, rtol=.1, atol=.1)
                    if all(close):
                        n_sphere.id = old_spheres[i].id
                        old_spheres[i] = n_sphere
                        matched = True
                if not matched:
                    # don't know a cleaner way to do this, can't use max(old_spheres.id)
                    ids = []
                    for sphere in old_spheres:
                        ids.append(sphere.id)
                    new_id = max(ids) + 1
                    n_sphere.id = new_id
                    old_spheres.append(n_sphere)
            self._spheres = old_spheres

        self._spheres = spheres
        return spheres
    
    def get_vector(self, spheres):
        # try:
        for _sphere in spheres:
            #get the front bins only
            if _sphere is None:
                continue
            front_bins={}
            for bin_id, val in _sphere.bins.items():
                if bin_id in [1,2,3,4,9,10,11,12]:
                    _bin = val
                front_bins[bin_id] = _bin

            #do we need a threshold for filled bins?? x number of points in each bin before going to next apple
            _bin=min(front_bins, key=front_bins.get)
            _sphere.min_bin = _bin
        # except Exception as e:
        #     self.err_logger(e)
            
        sphere = spheres[np.argmin([_sphere.min_bin for _sphere in spheres if _sphere is not None])]
        self._target_sphere = sphere

        # self.info_logger(f"Sphere min bin: {sphere.min_bin}")

        full_bin=sphere.bins[_bin]
        #I think this is in global frame?? (z up, x left to right, y into page) need to check this frame
        #from https://stackoverflow.com/questions/30011741/3d-vector-defined-by-2-angles
        theta = full_bin[1] +  np.pi / self.num_bins
        phi = full_bin[2] + np.pi / 4
        y = np.cos(theta) * np.cos(phi - np.pi / 2)
        x = np.sin(theta) * np.cos(phi - np.pi / 2)
        # have to offset to get the bottom half of the sphere to be negative
        z = np.sin(phi - np.pi / 2)
        unit_vector=np.array([x,y,z])
        unit_vector = unit_vector / np.linalg.norm(unit_vector)
        camera_orientation=-1*unit_vector

        # self.warn_logger(f"bin: {_bin}")
        # self.warn_logger(f"xyz: {[x, y, z]}")
        # self.warn_logger(f"theta: {theta}")
        # self.warn_logger(f"phi: {phi}")
        # self.warn_logger(f"unit_vector: {unit_vector}")
        
        


        #subject to change, uses the y distance to center sphere from scan (may be unreachable, may want to use different approach)
        camera_coords=np.add([sphere.center_x, sphere.center_y, sphere.center_z], unit_vector*sphere.center_x)
        # camera_coords = np.array([0.5, 0.5, 0.5])
        # camera_coords=np.array(camera_coords)

        if camera_coords[2] < 0.2:
            camera_coords[2] = 0.2

        coord_radius=np.sqrt(camera_coords[0]**2+camera_coords[1]**2+camera_coords[2]**2)
        # #if trying to move out of 90% of max reach
        if coord_radius>1*.9:
            self.info_logger("ENTERED IF STATEMENT")
            scaling=(.85*.9)/coord_radius
            camera_coords=camera_coords*scaling

        self.warn_logger(f"camera_coords: {camera_coords}")
        self.warn_logger(f"Sphere center: {sphere.center}")

        # Get the unit vector pointing from desired camera position to the center of the apple
        vec_camera_to_apple = sphere.center - camera_coords
        vec_camera_to_apple = vec_camera_to_apple / np.linalg.norm(vec_camera_to_apple)

        self.info_logger(f"vec: {vec_camera_to_apple}")

        coord_frame = np.identity(3)
        z_axis = coord_frame[2]
        
        # Get rotational vector, convert to quaternion
        rot_axis = np.cross(z_axis, vec_camera_to_apple)
        rot_angle = np.arccos(np.dot(z_axis, vec_camera_to_apple)) # both vectors are unit vectors
        self.info_logger(f"Cross: {rot_axis}")
        rotation = Rotation.from_rotvec(rotvec=rot_angle * rot_axis)
        quat = rotation.as_quat()

        self.warn_logger(f'quat: {quat}')

        return camera_coords, quat

    def get_nbv_vec(self, cloud):
        """Main processing sequence called when run_nbv service is called"""
        apple_cluster_points, apple_spheres = self.xy_clustering(cloud)
        spheres = self.get_spheres(cloud_list=apple_cluster_points, spheres=apple_spheres)
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
