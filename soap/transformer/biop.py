"""
.. module:: soap.transformer.biop
    :synopsis: Transforms expression instances with binary operators.
"""
import re
import random

import soap.logger as logger
from soap.expr.common import (
    ADD_OP, MULTIPLY_OP, ASSOCIATIVITY_OPERATORS,
    ADD3_OP,
    LEFT_DISTRIBUTIVITY_OPERATORS, LEFT_DISTRIBUTIVITY_OPERATOR_PAIRS,
    RIGHT_DISTRIBUTIVITY_OPERATORS, RIGHT_DISTRIBUTIVITY_OPERATOR_PAIRS,
    is_expr
)
from soap.expr import Expr
from soap.transformer.core import (
    item_to_list, none_to_list, TreeTransformer, ValidationError
)
from soap.semantics import mpq_type


@none_to_list
def associativity(t):
    """Associativity relation between expressions.

    For example:
        (a + b) + c == a + (b + c)

    :param t: The expression tree.
    :type t: :class:`soap.expr.Expr`
    :returns: A list of expressions that are derived with assoicativity from
        the input tree.
    """
    def expr_from_args(args):
        for a in args:
            al = list(args)
            al.remove(a)
            yield Expr(t.op, a, Expr(t.op, al))
    if not t.op in ASSOCIATIVITY_OPERATORS:
        return
    s = []
    # (a + b) + c == a + (b + c)
    if is_expr(t.a1) and t.a1.op == t.op:
        s.extend(list(expr_from_args(t.a1.args + [t.a2])))
    # a + (b + c) == (a + b) + c
    if is_expr(t.a2) and t.a2.op == t.op:
        s.extend(list(expr_from_args(t.a2.args + [t.a1])))
    return s


def distribute_for_distributivity(t):
    """Distributivity relation between expressions by distributing.

    For example:
        (a + b) * c == (a * c) + (b * c)

    :param t: The expression tree.
    :type t: :class:`soap.expr.Expr`
    :returns: A list of expressions that are derived with distributivity from
        the input tree.
    """
    s = []
    # a * (b + c) == (a * b) + (a * c)
    if t.op in LEFT_DISTRIBUTIVITY_OPERATORS and is_expr(t.a2):
        if (t.op, t.a2.op) in LEFT_DISTRIBUTIVITY_OPERATOR_PAIRS:
            s.append(Expr(t.a2.op,
                          Expr(t.op, t.a1, t.a2.a1),
                          Expr(t.op, t.a1, t.a2.a2)))
    # (a + b) * c == (a * c) + (b * c)
    if t.op in RIGHT_DISTRIBUTIVITY_OPERATORS and is_expr(t.a1):
        if (t.op, t.a1.op) in RIGHT_DISTRIBUTIVITY_OPERATOR_PAIRS:
            s.append(Expr(t.a1.op,
                          Expr(t.op, t.a1.a1, t.a2),
                          Expr(t.op, t.a1.a2, t.a2)))
    # logger.debug(distribute_for_distributivity.__name__, t, s)
    return s


@none_to_list
def two_add2_to_one_add3(t):
    if t.op != ADD_OP:
        return
    s = []
    # (a + b) + c == add3(a, b, c)
    if is_expr(t.a1) and t.a1.op == ADD_OP:
        s.extend([Expr(ADD3_OP, *t.a1.args, t.a2)])
    # a + (b + c) == add3(a, b, c)
    if is_expr(t.a2) and t.a2.op == ADD_OP:
        s.extend([Expr(ADD3_OP, t.a1, *t.a2.args)])
    return s

@none_to_list
def collect_for_distributivity(t):
    """Distributivity relation between expressions by collecting common terms.

    For example:
        (a * c) + (b * c) == (a + b) * c

    :param t: The expression tree.
    :type t: :class:`soap.expr.Expr`
    :returns: A list of expressions that are derived with distributivity from
        the input tree.
    """
    def al(a):
        if not is_expr(a):
            return [a, 1]
        if (a.op, t.op) == (MULTIPLY_OP, ADD_OP):
            return a.args
        return [a, 1]

    def sub(l, e):
        l = list(l)
        l.remove(e)
        return l.pop()

    # depth test
    if all(not is_expr(a) for a in t.args):
        return
    # operator tests
    if t.op != ADD_OP:
        return
    if all(a.op != MULTIPLY_OP for a in t.args if is_expr(a)):
        return
    # forming list
    af = [al(arg) for arg in t.args]
    # find common elements
    s = []
    for ac in set.intersection(*(set(a) for a in af)):
        an = [sub(an, ac) for an in af]
        s.append(Expr(MULTIPLY_OP, ac, Expr(ADD_OP, an)))
    return s


def _identity_reduction(t, iop, i):
    if t.op != iop:
        return
    if t.a1 == i:
        return t.a2
    if t.a2 == i:
        return t.a1


@item_to_list
def multiplicative_identity_reduction(t):
    """Multiplicative identity relation from an expression.

    For example:
        a * 1 == a

    :param t: The expression tree.
    :type t: :class:`soap.expr.Expr`
    :returns: A list containing an expression related by this reduction rule.
    """
    return _identity_reduction(t, MULTIPLY_OP, 1)


@item_to_list
def additive_identity_reduction(t):
    """Additive identity relation from an expression.

    For example:
        a + 0 == a

    :param t: The expression tree.
    :type t: :class:`soap.expr.Expr`
    :returns: A list containing an expression related by this reduction rule.
    """
    return _identity_reduction(t, ADD_OP, 0)


@item_to_list
def zero_reduction(t):
    """The zero-property of expressions.

    For example:
        a * 0 == 0

    :param t: The expression tree.
    :type t: :class:`soap.expr.Expr`
    :returns: A list containing an expression related by this reduction rule.
    """
    if t.op != MULTIPLY_OP:
        return
    if t.a1 != 0 and t.a2 != 0:
        return
    return 0


@item_to_list
def constant_reduction(t):
    """Constant propagation.

    For example:
        1 + 2 == 3

    :param t: The expression tree.
    :type t: :class:`soap.expr.Expr`
    :returns: A list containing an expression related by this reduction rule.
    """
    def is_exact(v):
        return isinstance(v, (int, mpq_type))
    if not is_exact(t.a1) or not is_exact(t.a2):
        return
    if t.op == MULTIPLY_OP:
        return t.a1 * t.a2
    if t.op == ADD_OP:
        return t.a1 + t.a2


class BiOpTreeTransformer(TreeTransformer):
    """The class that provides transformation of binary operator expressions.

    It has the same arguments as :class:`soap.transformer.TreeTransformer`,
    which is the class it is derived from.
    """
    transform_methods = [associativity,
                         distribute_for_distributivity,
                         collect_for_distributivity]

    reduction_methods = [multiplicative_identity_reduction,
                         additive_identity_reduction, zero_reduction,
                         constant_reduction]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    VAR_RE = re.compile(r"[^\d\W]\w*", re.UNICODE)

    def validate(t, tn):
        # FIXME: broken after ErrorSemantics
        def vars(tree_str):
            return set(BiOpTreeTransformer.VAR_RE.findall(tree_str))
        to, no = ts, ns = str(t), str(tn)
        tsv, nsv = vars(ts), vars(ns)
        if tsv != nsv:
            raise ValidationError('Variable domain mismatch.')
        vv = {v: random.randint(0, 127) for v in tsv}
        for v, i in vv.items():
            ts = re.sub(r'\b%s\b' % v, str(i), ts)
            ns = re.sub(r'\b%s\b' % v, str(i), ns)
        if eval(ts) != eval(ns):
            raise ValidationError(
                'Failed validation\n'
                'Original: %s %s,\n'
                'Transformed: %s %s' % (to, t, no, tn))


class FusedOnlyBiOpTreeTransformer(TreeTransformer):
    """The class that only makes transformations to and from fused unit expressions.

    It has the same arguments as :class:`soap.transformer.BiOpTreeTransformer`,
    which is the class it is derived from."""
    transform_methods = [two_add2_to_one_add3]

    reduction_methods = BiOpTreeTransformer.reduction_methods

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FusedBiOpTreeTransformer(BiOpTreeTransformer):
    """The class that has transformations for both BiOp and Fused.

    It has the same arguments as :class:`soap.transformer.BiOpTreeTransformer`,
    which is the class it is derived from."""
    transform_methods = BiOpTreeTransformer.transform_methods + \
        FusedOnlyBiOpTreeTransformer.transform_methods

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    import time
    import gmpy2
    from soap.common import profiled, timed, invalidate_cache
    from soap.analysis.utils import plot, analyse, analyse_and_frontier
    from soap.transformer.utils import greedy_frontier_closure, greedy_trace, frontier_trace, martel_trace
    from soap.analysis import Plot
    from soap.analysis.core import pareto_frontier_2d
    
    Expr.__repr__ = Expr.__str__
    logger.set_context(level=logger.levels.info)
    single_prec = gmpy2.ieee(32).precision - 1

    # e = '(a + 1) * b | (b + 1) * a | a * b'
    e = '(a + b) + c'
    v = {'a': ['1', '2'], 'b': ['100', '200'], 'c': ['0.1', '0.2']}
    e2 = '((a + a) + b) * ((a + b) + b)'
    e60 = """
        (
            (
                (   
                    (
                        (
                            ((a + a) + b) 
                            * ((a + b) + b)
                        ) 
                        * ((b + b) + c)
                    ) 
                    * ((b + c) + c)
                ) 
                * ((c + c) + a)
            ) 
            * ((c + a) + a)
        )"""
    e6 = """
        (
            (
                (
                    ((a + a) + b) 
                    * ((a + b) + b)
                )
                * (
                    ((b + b) + c)
                    * ((b + c) + c)
                )
            )
            * (
                ((c + c) + a)
                * ((c + a) + a)
            )
        )"""
    v6 = {
        'a': ['1', '2'],
        'b': ['10', '20'],
        'c': ['100', '200'],
    }
    v6a = v6
    v6a['a'], v6a['b'] = v6a['c'], v6a['c']

    benchmarks = {
        '2d_hydro': {
            'e': 'z + (0.175 * ((((((a*b) + (c*d)) + (e*f)) + (g*h)) + i) + j))',
            'v': {'a':[0,1],'b':[0,1],'c':[0,1],'d':[0,1],'e':[0,1],'f':[0,1],'g':[0,1],'h':[0,1],'i':[0,1],'j':[-1,0],'z':[0,1],}
        },
        'fdtd_1': {
            'e': 'a + (0.5*(c + b))',
            'v': {'a':[0,1],'b':[-1,0],'c':[0,1]}
        },
        'fdtd': {
            'e': 'a + (-0.7)*(b + c + d + e)',
            'v': {'a':[0,1], 'b':[0,1], 'c':[-1,0], 'd':[0,1], 'e':[-1,0]}
        },
        'filter': {
            'e': 'a0 * y0 + a1 * y1 + a2 * y2 + b0 * x0 + b1 * x1 + b2 * x2',
            'v': {
                'x0': [0.0, 1.0],
                'x1': [0.0, 1.0],
                'x2': [0.0, 1.0],
                'y0': [0.0, 1.0],
                'y1': [0.0, 1.0],
                'y2': [0.0, 1.0],
                'a0': [0.2, 0.3],
                'a1': [0.1, 0.2],
                'a2': [0.0, 0.1],
                'b0': [0.2, 0.3],
                'b1': [0.1, 0.2],
                'b2': [0.0, 0.1]
            }
        },
        'gemm': {
            'e': 'C + 32412 * A * B',
            'v': {
                'A': [0, 1],
                'B': [0, 1],
                'C': [0, 1]
            }
        },
        'seidel': {
            'e': '0.2*(a+b+c+d+e)',
            'v': {
                'a': [0, 1],
                'b': [0, 1],
                'c': [0, 1],
                'd': [0, 1],
                'e': [0, 1],
            }
        },
        'symm': {
            'e': 'beta * C + alpha * A * B + alpha * acc',
            'v': { # find ranges
                'alpha': [0, 1],
                'beta': [0, 1],
                'A': [0, 1],
                'B': [0, 1],
                'C': [0, 1],
                'acc': [0, 1],
            }
        },
        'taylor_b': {
            'e': 'b * (2 * i + 1) * (2 * i)',
            'v': {
                'b': [0, 7e48], # ~ product of (4*i^2) from i=1 to 20 is (4^20 * (20!)^2)
                'i': [1, 20],
            }
        },
        'taylor_p': {
            'e': '(x + y) * (x + y)',
            'v': {
                'p': [0, 1.21**40], # (x+y)^20 ~ 2048
                'x': [-0.1, 0.1],
                'y': [0, 1],
            }
        },


    }

    # e, v = e2, v6
    bench_name = 'taylor_p'
    e, v = benchmarks[bench_name]['e'], benchmarks[bench_name]['v']
    t = Expr(e)
    logger.info('Expr:', str(t))
    logger.info('Tree:', t.tree())

    # fused unit = 3-input FP adder(s)
    actions = (
        (BiOpTreeTransformer, 'no transformations to fused units'),
        (FusedBiOpTreeTransformer, 'fused units considered'),
        (FusedOnlyBiOpTreeTransformer, 'only transformations to fused units'))
    plots = []
    # with profiled(), timed():
    for trace_ in (
        (frontier_trace, 3),
        (greedy_trace, None))[1:]:
        for action in (actions, actions[::-1])[:1]:
            z = []
            frontier = []
            p = Plot(depth=3, var_env=v, blocking=False)#,legend_pos='top right')
            for Transformer, label in action:
                invalidate_cache()
                duration = time.time()
                s = Transformer(t, depth=None).closure()
                #s = trace_[0](t, v, depth=trace_[1], transformer=Transformer)
                unfiltered, frontier = analyse_and_frontier(s, v, prec=single_prec)
                duration = time.time() - duration # where to put this?

                logger.info('Transformed:', len(s))
                logger.info('Reduced by ', len(unfiltered)-len(frontier))
                if len(frontier) <= 1:
                    # plot non-frontier points too
                    frontier = unfiltered
                logger.info(frontier)
                # plot(frontier, blocking=False)
                p.add(frontier, legend=label, time=duration, annotate=True, annotate_kwargs={'fontsize': 10})
                z.append(set(map(lambda d:(d['area'], d['error'], d['expression']), frontier)))

            logger.warning(z[0]-z[1])
            logger.warning()
            logger.warning(z[1]-z[0])

            # find min error and area here

            p.add_analysis(t, legend='original expression', s=300)
            p.show()

    input('Press Enter to continue...')