"""
.. module:: soap.expr.common
    :synopsis: Common definitions for expressions.
"""
ADD_OP = '+'
SUBTRACT_OP = '-'
MULTIPLY_OP = '*'
DIVIDE_OP = '/'
BARRIER_OP = '|'
UNARY_SUBTRACT_OP = '-'
# add3(a, b, c) == (a + b + c)
ADD3_OP = 'add3'
# constMult(z, a) == (z * a), where z is a constant number
CONSTANT_MULTIPLY_OP = 'constMult'

OPERATORS = [ADD_OP, MULTIPLY_OP, ADD3_OP, CONSTANT_MULTIPLY_OP]

OPERATORS_WITH_AREA_INFO = OPERATORS

ASSOCIATIVITY_OPERATORS = [ADD_OP, MULTIPLY_OP]

COMMUTATIVITY_OPERATORS = ASSOCIATIVITY_OPERATORS

COMMUTATIVE_DISTRIBUTIVITY_OPERATOR_PAIRS = [(MULTIPLY_OP, ADD_OP)]
# left-distributive: a * (b + c) == a * b + a * c
LEFT_DISTRIBUTIVITY_OPERATOR_PAIRS = \
    COMMUTATIVE_DISTRIBUTIVITY_OPERATOR_PAIRS
# Note that division '/' is only right-distributive over +
RIGHT_DISTRIBUTIVITY_OPERATOR_PAIRS = \
    COMMUTATIVE_DISTRIBUTIVITY_OPERATOR_PAIRS

LEFT_DISTRIBUTIVITY_OPERATORS, LEFT_DISTRIBUTION_OVER_OPERATORS = \
    list(zip(*LEFT_DISTRIBUTIVITY_OPERATOR_PAIRS))
RIGHT_DISTRIBUTIVITY_OPERATORS, RIGHT_DISTRIBUTION_OVER_OPERATORS = \
    list(zip(*RIGHT_DISTRIBUTIVITY_OPERATOR_PAIRS))

PRIMITIVE_OPERATORS_WITH_2_TERMS = [ADD_OP, SUBTRACT_OP, MULTIPLY_OP, DIVIDE_OP]


def is_expr(e):
    """Check if `e` is an expression."""
    from soap.expr.biop import Expr
    return isinstance(e, Expr)


def concat_multi_expr(*expr_args):
    """Concatenates multiple expressions into a single expression by using the
    barrier operator `|`.
    """
    from soap.expr.biop import Expr
    me = None
    for e in expr_args:
        e = Expr(e)
        me = me | e if me else e
    return me


def split_multi_expr(e):
    """Splits the single expression into multiple expressions."""
    if e.op != BARRIER_OP:
        return [e]
    return split_multi_expr(e.a1) + split_multi_expr(e.a2)
