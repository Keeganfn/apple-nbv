#!/usr/bin/env python3


from dataclasses import dataclass

import numpy as np

from typing import Union

import itertools


class Sphere:
    id_iter = itertools.count()
    
    def __init__(self, center_x: float, center_y, center_z, radius, num_bins=8):
        self.center_x: float = center_x
        self.center_y: float = center_y
        self.center_z: float = center_z
        self.radius: float = radius
        self.center = np.array([self.center_x, self.center_y, self.center_z])
        self.theta_bin_counts: dict = {}
        self.phi_bin_counts: dict = {}
        self.bins: dict = {}
        self.min_bin: Union[int, None] = None
        self.volume_estimate=(4/3)*np.pi*radius**3
        self.id = next(self.id_iter)