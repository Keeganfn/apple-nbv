#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

class ExtendedNormalAlgorithm(Node):
    def __init__(self) -> None:
        super().__init__(node_name="extended_normal_algorithm")

        return
    
    def count_faces(self, voxel_list):
        x_p_faces = 0
        y_p_faces = 0
        y_n_faces = 0
        z_p_faces = 0
        z_n_faces = 0
        for voxel in voxel_list:
            if voxel.unseen:
                x_faces += 1

        return
    
    # need collective center of apples?
    

    

def main():
    rclpy.init()
    ena = ExtendedNormalAlgorithm()
    rclpy.spin(ena, executor=MultiThreadedExecutor())
    ena.destroy_node()
    rclpy.shutdown()
    return


if __name__ == "__main__":
    main()