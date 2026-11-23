"""``calculator`` -- safe arithmetic expression evaluation via an ``ast``
node allowlist. Demonstrates how to write a parser tool that does **not**
call ``eval`` and does **not** expose Python's name resolution.
"""

import ast
import operator as _op
from typing import Any, Callable, Dict

from .core import tool


# Operator table for the safe calculator. Anything not in this table is
# rejected at parse time -- in particular ``Name``, ``Call``, ``Attribute``,
# ``Subscript``, ``Lambda``, ``IfExp``, comprehensions, walrus, etc.
_CALC_BINOPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_CALC_UNARYOPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
}
# Hard cap on the magnitude of the right-hand side of ``**``. ``2**10**10``
# would otherwise hang the agent thread. 1000 is generous for any
# legitimate arithmetic question while remaining trivially fast.
_CALC_MAX_EXPONENT = 1000
_CALC_MAX_EXPR_LEN = 200
# Cap on the size of an integer result. The exponent cap alone does not bound
# it: ``((9**999)**999)**999`` has small exponents and a ~3e9-bit result,
# computed while holding the GIL.
_CALC_MAX_INT_BITS = 100_000


def _int_bits(x: Any) -> int:
    return abs(x).bit_length() if isinstance(x, int) else 0


def _calc_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node from the calculator's allowlist."""
    if isinstance(node, ast.Expression):
        return _calc_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"constant of disallowed type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op_fn = _CALC_BINOPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"disallowed binary operator: {type(node.op).__name__}")
        left = _calc_eval(node.left)
        right = _calc_eval(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > _CALC_MAX_EXPONENT:
            raise ValueError(f"exponent {right!r} exceeds maximum of {_CALC_MAX_EXPONENT}")
        # bits(a**b) <= bits(a) * b and bits(a*b) <= bits(a) + bits(b), estimated before computing
        if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int) and right > 0:
            estimate = _int_bits(left) * right
        elif isinstance(node.op, ast.Mult):
            estimate = _int_bits(left) + _int_bits(right)
        else:
            estimate = 0
        if estimate > _CALC_MAX_INT_BITS:
            raise ValueError(f"result too large (about {estimate} bits; limit {_CALC_MAX_INT_BITS})")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        unary_fn = _CALC_UNARYOPS.get(type(node.op))
        if unary_fn is None:
            raise ValueError(f"disallowed unary operator: {type(node.op).__name__}")
        return unary_fn(_calc_eval(node.operand))
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Supports the operators ``+ - * / // % **`` and unary ``+ -`` over
    integer and float literals, with normal precedence and parentheses.
    Names, function calls, attribute access, subscripting, comparisons,
    and every other Python construct are rejected at parse time -- this
    is **not** ``eval``.

    Args:
        expression: The arithmetic expression to evaluate, e.g.
            ``"(2 + 3) * 4"`` or ``"2 ** 10"``. Limited to 200 characters.

    Returns:
        The result as a string. Integer results render without a decimal
        point; floats use Python's default ``repr`` formatting.
    """
    if len(expression) > _CALC_MAX_EXPR_LEN:
        raise ValueError(f"expression too long ({len(expression)} > {_CALC_MAX_EXPR_LEN} chars)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid arithmetic expression: {e.msg}") from e
    result = _calc_eval(tree)
    return str(result)
