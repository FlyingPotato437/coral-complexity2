import numpy as np


class AABB:
    def __init__(self, min_bound, max_bound):
        self.min_bound = min_bound
        self.max_bound = max_bound

    def intersect(self, ray_origin, ray_direction):
        t_min = np.zeros(3)
        t_max = np.zeros(3)
        for i in range(3):
            if ray_direction[i] != 0:
                t_min[i] = (self.min_bound[i] - ray_origin[i]) / \
                    ray_direction[i]
                t_max[i] = (self.max_bound[i] - ray_origin[i]) / \
                    ray_direction[i]
                if t_min[i] > t_max[i]:
                    t_min[i], t_max[i] = t_max[i], t_min[i]
            else:
                t_min[i] = float(
                    '-inf') if ray_origin[i] >= self.min_bound[i] else float('inf')
                t_max[i] = float(
                    'inf') if ray_origin[i] <= self.max_bound[i] else float('-inf')
        t_enter = max(t_min)
        t_exit = min(t_max)
        return t_enter <= t_exit and t_exit > 0


class BVHNode:
    def __init__(self, start, end, aabb):
        self.start = start
        self.end = end
        self.aabb = aabb
        self.left = None
        self.right = None
