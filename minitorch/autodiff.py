from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
from contextlib import contextmanager
from contextvars import ContextVar

_grad_enabled = ContextVar('minitorch_grad_enabled', default=True)


def is_grad_enabled():
    return _grad_enabled.get()


@contextmanager
def set_grad_enabled(enabled):
    token = _grad_enabled.set(bool(enabled))
    try:
        yield
    finally:
        _grad_enabled.reset(token)


def no_grad():
    """Disable recording new operations; leaves and existing graphs are unchanged."""
    return set_grad_enabled(False)


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    
    Hints:
        1. Ensure that you visit the computation graph in a post-order depth-first search.
        2. When the children nodes of the current node are visited, add the current node 
            at the front of the result order list.
    """
    # BEGIN ASSIGN2_1
    # TODO
    # Iterative DFS avoids a recursive closure cycle retaining the entire graph
    # (and device allocations) until Python's cyclic collector happens to run.
    order: List[Variable] = []
    visited = set()
    stack = [(variable, False)]
    while stack:
        node, expanded = stack.pop()
        ctx = getattr(getattr(node, 'history', None), 'ctx', None)
        if ctx is not None and ctx.released:
            raise RuntimeError('Computation graph was released; use retain_graph=True on the first backward or run a new forward')
        if ctx is not None:
            ctx.check_versions()
        if node.is_constant():
            continue
        if expanded:
            order.append(node)
            continue
        if node.unique_id in visited:
            continue
        visited.add(node.unique_id)
        stack.append((node, True))
        stack.extend((parent, False) for parent in reversed(tuple(node.parents)))
    return reversed(order)
    # END ASSIGN2_1


def backpropagate(variable: Variable, deriv: Any, retain_graph: bool = True) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write its results to the derivative values of each leaf through `accumulate_derivative`.

    Hints:
        1. Traverse nodes in topological order
        2. If the node is a leaf, the derivative should be accumulated
        3. Otherwise, the derivative should be propagated via chain rule
    """
    # BEGIN ASSIGN2_1
    # TODO
    derivatives = {variable.unique_id: deriv}

    # Consume the schedule so completed nodes/gradients do not stay referenced
    # until the end of the entire backward. Fan-out is accumulated before use.
    schedule = list(topological_sort(variable))
    schedule.reverse()
    while schedule:
        node = schedule.pop()
        d_node = derivatives.pop(node.unique_id)
        if node.is_leaf():
            node.accumulate_derivative(d_node)
            continue

        for parent, d_parent in node.chain_rule(d_node):
            if parent.is_constant():
                continue
            
            if parent.unique_id in derivatives:
                derivatives[parent.unique_id] += d_parent
            else:
                derivatives[parent.unique_id] = d_parent
        if not retain_graph and hasattr(node, 'release_history'):
            node.release_history()
    # END ASSIGN2_1


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()
    released: bool = False
    saved_versions: Tuple[Any, ...] = ()

    def release(self):
        self.saved_values = ()
        self.saved_versions = ()
        self.released = True

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values
        versions = []
        for value in values:
            storage = getattr(getattr(value, '_tensor', None), '_storage', None)
            owner = getattr(storage, 'owner', None)
            if owner is not None:
                versions.append((owner, owner.version))
        self.saved_versions = tuple(versions)

    def check_versions(self):
        if any(owner.version != version for owner, version in self.saved_versions):
            raise RuntimeError('Saved tensor was modified by an optimizer step; run a new forward before backward')

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        self.check_versions()
        return self.saved_values
