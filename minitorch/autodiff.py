from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    xs = list(vals)
    ys = list(vals)
    xs[arg] += epsilon
    ys[arg] -= epsilon
    return (f(*xs) - f(*ys)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add an incoming derivative to a leaf."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the identifier of this node."""
        ...

    def is_leaf(self) -> bool:
        """Check whether the node is a trainable input."""
        ...

    def is_constant(self) -> bool:
        """Check whether the node has no history."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the inputs of the last operation."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Pair each nonconstant input with its local derivative."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    out: list[Variable] = []
    vis: set[int] = set()

    def dfs(var: Variable) -> None:
        if var.is_constant() or var.unique_id in vis:
            return
        vis.add(var.unique_id)
        for par in var.parents:
            dfs(par)
        out.append(var)

    dfs(variable)
    return reversed(out)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Propagate the output derivative to all reachable leaves."""
    ds = {variable.unique_id: deriv}
    for var in topological_sort(variable):
        d = ds[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            for par, val in var.chain_rule(d):
                if not par.is_constant():
                    key = par.unique_id
                    ds[key] = ds.get(key, 0.0) + val


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the values saved during the forward pass."""
        return self.saved_values
