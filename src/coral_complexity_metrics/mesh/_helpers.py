import math
from ._vertex import Vertex


def is_min(x, y):
    return True if x < y else False


def is_max(x, y):
    return True if x > y else False


def get_z_value(vertex):
    return vertex.z


def mean(values):
    return sum(values) / len(values)


def sd(values, sample):
    values_mean = mean(values)
    differences = [x - values_mean for x in values]
    square_differences = [d ** 2 for d in differences]
    sum_of_squares = sum(square_differences)
    variance = sum_of_squares / \
        (len(values) - 1) if sample is True else sum_of_squares / len(values)
    return math.sqrt(variance)


def get_midpoint_of_edge(start, end):
    return Vertex((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2)
