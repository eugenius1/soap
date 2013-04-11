#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :


import ast

from ..semantics import mpq
from .common import OPERATORS, ADD_OP, MULTIPLY_OP


def try_to_number(s):
    try:
        return mpq(s)
    except (ValueError, TypeError):
        return s


OPERATOR_MAP = {
    ast.Add: ADD_OP,
    ast.Mult: MULTIPLY_OP,
}


def parse(s):
    from .biop import Expr
    def _parse_r(t):
        try:
            return t.n
        except AttributeError:
            pass
        try:
            return t.id
        except AttributeError:
            op = OPERATOR_MAP[t.op.__class__]
            a1 = _parse_r(t.left)
            a2 = _parse_r(t.right)
            return Expr(op, a1, a2)
    return _parse_r(ast.parse(s, mode='eval').body)
