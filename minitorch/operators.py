"""Basic scalar operations and functions on lists."""

import math as mth
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiply two values."""
    return x * y


def id(x: float) -> float:
    """Return the input."""
    return x


def add(x: float, y: float) -> float:
    """Add two values."""
    return x + y


def neg(x: float) -> float:
    """Negate a value."""
    return -x


def lt(x: float, y: float) -> float:
    """Return one when x is smaller than y."""
    return float(x < y)


def eq(x: float, y: float) -> float:
    """Return one when the values are equal."""
    return float(x == y)


def max(x: float, y: float) -> float:
    """Return the larger value."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Compare values with an absolute tolerance of 0.01."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute sigmoid without exponentiating a large positive value."""
    if x >= 0:
        return 1.0 / (1.0 + mth.exp(-x))
    val = mth.exp(x)
    return val / (1.0 + val)


def relu(x: float) -> float:
    """Keep positive values and replace the rest with zero."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Compute the natural logarithm."""
    return mth.log(x)


def exp(x: float) -> float:
    """Compute the exponential."""
    return mth.exp(x)


def inv(x: float) -> float:
    """Compute the reciprocal."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Multiply the incoming derivative by the derivative of log."""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Multiply the incoming derivative by the derivative of reciprocal."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Pass the derivative only for positive inputs."""
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], list[float]]:
    """Build a function that applies fn to each value."""

    def run(xs: Iterable[float]) -> list[float]:
        return [fn(x) for x in xs]

    return run


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], list[float]]:
    """Build a function that combines corresponding values."""

    def run(xs: Iterable[float], ys: Iterable[float]) -> list[float]:
        return [fn(x, y) for x, y in zip(xs, ys)]

    return run


def reduce(
    fn: Callable[[float, float], float],
    ini: float,
) -> Callable[[Iterable[float]], float]:
    """Build a left fold starting at ini."""

    def run(xs: Iterable[float]) -> float:
        val = ini
        for x in xs:
            val = fn(val, x)
        return val

    return run


def negList(xs: Iterable[float]) -> list[float]:
    """Negate every value."""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> list[float]:
    """Add corresponding values."""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Add all values, starting at zero."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Multiply all values, starting at one."""
    return reduce(mul, 1.0)(xs)
