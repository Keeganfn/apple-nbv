#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class NextBestViewClientRequest(Node):
    def __init__(self) -> None:
        super().__init__(node_name="next_best_view_client")
        