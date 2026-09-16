"""Checks for shared nodes and constants in the computation graph."""

import pytest

import minitorch as mt


def test_shared_node() -> None:
    x = mt.Scalar(3.0)
    y = x * x
    z = y * y + y
    z.backward()
    assert x.derivative == 114.0
    z.backward()
    assert x.derivative == 228.0


def test_constant() -> None:
    x = mt.Scalar(3.0)
    c = mt.Scalar(5.0, back=None)
    z = x * c + c
    z.backward()
    assert x.derivative == 5.0
    assert c.derivative is None
    assert c.unique_id not in [v.unique_id for v in mt.topological_sort(z)]


def test_order() -> None:
    x = mt.Scalar(2.0)
    y = x * x
    z = y + x
    out = list(mt.topological_sort(z))
    idx = {v.unique_id: i for i, v in enumerate(out)}
    assert len(idx) == len(out)
    for v in out:
        for p in v.parents:
            assert idx[v.unique_id] < idx[p.unique_id]


def test_integer_inputs() -> None:
    assert mt.Neg.apply(5).data == -5.0
    assert mt.Mul.apply(2, 3).data == 6.0
    x = mt.Scalar(2.0)
    y = 5 - x
    y.backward()
    assert y.data == 3.0
    assert x.derivative == -1.0


def test_zero_derivatives() -> None:
    x = mt.Scalar(0.0)
    (x.relu() + (x < 1) + (x == 0)).backward()
    assert x.derivative == 0.0


def test_composed_derivative() -> None:
    def f(x: mt.Scalar, y: mt.Scalar) -> mt.Scalar:
        return ((x * y).sigmoid() + x.log()) / y.exp()

    x = mt.Scalar(1.2)
    y = mt.Scalar(0.7)
    z = f(x, y)
    z.backward()
    for i, v in enumerate([x, y]):
        d = mt.central_difference(f, x, y, arg=i).data
        assert v.derivative == pytest.approx(d, rel=1e-5, abs=1e-7)
