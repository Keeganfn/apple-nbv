#!/usr/bin/env python3


from dataclasses import dataclass

import numpy as np


@dataclass
class Sphere:
    # center_x: float
    # center_y: float
    # center_z: float
    # radius: float
    # num_bins: int = 8
    # bin_count_list: np.ndarray = np.zeros(num_bins, dtype=float)

    def __init__(self, center_x: float, center_y, center_z, radius, num_bins=8):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.radius = radius
        self.center = np.array([self.center_x, self.center_y, self.center_z])
        self.bin_count_list = np.zeros(num_bins, dtype=float)

