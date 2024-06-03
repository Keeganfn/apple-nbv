#!/usr/bin/env python3


from dataclasses import dataclass

import numpy as np

class Sphere:
    def __init__(self, center_x: float, center_y, center_z, radius, num_bins=8):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.radius = radius
        self.center = np.array([self.center_x, self.center_y, self.center_z])
        self.theta_bin_counts: dict = {}
        self.phi_bin_counts: dict = {}

