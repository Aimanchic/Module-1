"""Small two-dimensional classification datasets."""

import math as mth
import random as rng
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(n: int) -> List[Tuple[float, float]]:
    """Sample points uniformly from the unit square."""
    return [(rng.random(), rng.random()) for i in range(n)]


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(n: int) -> Graph:
    """Label points to the left of x = 0.5 as class one."""
    xs = make_pts(n)
    ys = [int(x < 0.5) for x, y in xs]
    return Graph(n, xs, ys)


def diag(n: int) -> Graph:
    """Label points below x + y = 0.5 as class one."""
    xs = make_pts(n)
    ys = [int(x + y < 0.5) for x, y in xs]
    return Graph(n, xs, ys)


def split(n: int) -> Graph:
    """Label the left and right strips of width 0.2 as class one."""
    xs = make_pts(n)
    ys = [int(x < 0.2 or x > 0.8) for x, y in xs]
    return Graph(n, xs, ys)


def xor(n: int) -> Graph:
    """Label the upper-left and lower-right quadrants as class one."""
    xs = make_pts(n)
    ys = [int((x < 0.5 and y > 0.5) or (x > 0.5 and y < 0.5)) for x, y in xs]
    return Graph(n, xs, ys)


def circle(n: int) -> Graph:
    """Label points outside the circle of squared radius 0.1 as class one."""
    xs = make_pts(n)
    ys = [int((x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.1) for x, y in xs]
    return Graph(n, xs, ys)


def spiral(n: int) -> Graph:
    """Make two interleaved spiral arms with an equal number of points."""

    def x(t: float) -> float:
        return t * mth.cos(t) / 20.0

    def y(t: float) -> float:
        return t * mth.sin(t) / 20.0

    xs = [
        (x(10.0 * i / (n // 2)) + 0.5, y(10.0 * i / (n // 2)) + 0.5)
        for i in range(5, 5 + n // 2)
    ]
    xs += [
        (y(-10.0 * i / (n // 2)) + 0.5, x(-10.0 * i / (n // 2)) + 0.5)
        for i in range(5, 5 + n // 2)
    ]
    ys = [0] * (n // 2) + [1] * (n // 2)
    return Graph(n, xs, ys)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
