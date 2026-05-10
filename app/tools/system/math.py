"""
Math tool — safely evaluates mathematical expressions.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any

from app.tools import register

logger = logging.getLogger(__name__)

_OPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FNS: dict[str, Any] = {n: getattr(math, n) for n in dir(math) if not n.startswith("_")}
_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        if fn in _FNS:
            return _FNS[fn](*[_safe_eval(a) for a in node.args])
        raise ValueError(f"Unknown function: {fn!r}")
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@register
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression without executing arbitrary code.
    Supports basic arithmetic and math functions (sqrt, log, sin, cos, etc.).
    
    :param expression: The math expression to evaluate (e.g., 'sqrt(16) * 2').
    """
    try:
        logger.info(f"Evaluating math expression: {expression}")
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        logger.error(f"Math evaluator error: {e}")
        return f"Error: {str(e)}"
