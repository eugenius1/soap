"""
.. module:: soap.transformer.utils
    :synopsis: Useful utility functions to simplify calls to
        BiOpTreeTransformer.
"""
import itertools

import soap.logger as logger
from soap.expr import Expr
from soap.transformer.core import TreeTransformer
from soap.transformer.biop import (
    associativity, distribute_for_distributivity, BiOpTreeTransformer
)
from soap.transformer.martel import MartelBiOpTreeTransformer
from soap.analysis import expr_frontier


def closure(tree, var_env=None, depth=None, prec=None, transformer=None, **kwargs):
    """The full transitive closure."""
    if not transformer:
        transformer = BiOpTreeTransformer
    traces = GreedyTraceExpr(tree).traces(var_env, depth, prec, transformer=transformer, **kwargs)
    traces.add(tree)
    return transformer(traces, depth=depth, **kwargs).closure()


def full_closure(tree, **kwargs):
    """The same as :func:`closure`, ignoring the `kwargs` stuff."""
    return closure(tree)


def greedy_frontier_closure(tree, var_env=None, prec=None, transformer=None, **kwargs):
    """Our greedy transitive closure.

    :param tree: The expression(s) under transform.
    :type tree: :class:`soap.expr.Expr`, set, or str
    :param var_env: The ranges of input variables.
    :type var_env: dictionary containing mappings from variables to
        :class:`soap.semantics.error.Interval`
    :param prec: Precision used to evaluate the expression, defaults to
        single precision.
    :type prec: int
    """
    if var_env:
        func = lambda s: expr_frontier(s, var_env, prec)
    else:
        func = None
    if not transformer:
        transformer = BiOpTreeTransformer
    closure = transformer(tree, step_plugin=func, **kwargs).closure()
    return expr_frontier(closure, var_env, prec)


def transform(tree,
              reduction_methods=None, transform_methods=None,
              step_plugin=None, reduce_plugin=None, depth=None,
              multiprocessing=True):
    """One liner for :class:`soap.transformer.TreeTransformer`."""
    t = TreeTransformer(
        tree, step_plugin=step_plugin, reduce_plugin=reduce_plugin,
        depth=depth, multiprocessing=multiprocessing)
    t.reduction_methods = reduction_methods or []
    t.transform_methods = transform_methods or []
    return t.closure()


def expand(tree):
    """Fully expands the expression tree by distributivity.

    :param tree: The expression tree.
    :type tree: :class:`soap.expr.Expr` or str
    :returns: A fully expanded tree.
    """
    def pop(s):
        if s:
            return [s.pop()]
        return s
    return transform(tree, reduction_methods=[distribute_for_distributivity],
                     reduce_plugin=pop, multiprocessing=False).pop()


def reduce(tree):
    """Transforms tree by reduction rules only.

    :param tree: The expression tree.
    :type tree: :class:`soap.expr.Expr` or str
    :returns: A new expression tree.
    """
    try:
        tree = Expr(tree)
    except TypeError:
        with logger.local_context(level=logger.levels.info):
            return {reduce(t) for t in tree}
    t = transform(tree, BiOpTreeTransformer.reduction_methods,
                  multiprocessing=False)
    s = set(t)
    if len(s) > 1:
        s.remove(tree)
    if len(s) == 1:
        return s.pop()
    raise Exception


def parsings(tree):
    """Generates all possible parsings of the same tree by associativity.

    :param tree: The expression tree.
    :type tree: :class:`soap.expr.Expr` or str
    :returns: A set of trees.
    """
    return transform(tree, None, [associativity])


def collecting_closure(tree, depth=None):
    """Fully closure, sans distributing terms.

    :param tree: The expression tree.
    :type tree: :class:`soap.expr.Expr` or str
    :param depth: The depth limit.
    :type depth: int
    :returns: A set of trees.
    """
    t = BiOpTreeTransformer(tree, depth=depth)
    t.transform_methods.remove(distribute_for_distributivity)
    return t.closure()


class TraceExpr(Expr):
    """A subclass of :class:`soap.expr.Expr` for bottom-up hierarchical
    equivalence finding.

    Implements :member:`traces` that finds equivalnent expressions. Subclasses
    needs to override :member:`closure`.
    """
    def traces(self, var_env=None, depth=None, prec=None, **kwargs):
        _, discovered = self._traces(var_env, depth, prec, **kwargs)
        return discovered

    def _traces(self, var_env=None, depth=None, prec=None, **kwargs):
        subtraces = []
        discovered = []
        for a in self.args:
            try:
                arg_subtraces, arg_discovered = \
                    self.__class__(a)._traces(var_env, depth, prec, **kwargs)
            except (ValueError, TypeError):
                arg_subtraces = arg_discovered = {a}
            subtraces.append(arg_subtraces)
            discovered.append(arg_discovered)
        list_to_expr_set = lambda st: \
            set(Expr(self.op, args) for args in itertools.product(*st))
        logger.debug('Generating traces')
        subtraces = list_to_expr_set(subtraces)
        logger.debug('Generated %d traces for tree: %s' %
                     (len(subtraces), str(self)))
        logger.debug('Finding closure')
        closure = set(self.closure(
            subtraces, depth=depth, var_env=var_env, prec=prec, **kwargs))
        return closure, closure | subtraces | list_to_expr_set(discovered)

    def closure(self, trees, **kwargs):
        raise NotImplementedError


class MartelTraceExpr(TraceExpr):
    """A subclass of :class:`TraceExpr` to generate Martel's results."""
    def closure(self, trees, **kwargs):
        return MartelBiOpTreeTransformer(
            trees, depth=kwargs['depth']).closure()


class GreedyTraceExpr(TraceExpr):
    """A subclass of :class:`TraceExpr` to generate our greedy_trace equivalent
    expressions."""
    def closure(self, trees, **kwargs):
        return greedy_frontier_closure(trees, **kwargs)


class FrontierTraceExpr(TraceExpr):
    """A subclass of :class:`TraceExpr` to generate our frontier_trace
    equivalent expressions."""
    def closure(self, trees, **kwargs):
        return expr_frontier(
            closure(trees,
                depth=kwargs['depth'],
                transformer=kwargs.get('transformer', None)
            ),
            kwargs['var_env'],
            prec=kwargs['prec'])


def martel_trace(tree, var_env=None, depth=2, prec=None, **kwargs):
    """Finds Martel's equivalent expressions.

    :param tree: The original expression.
    :type tree: :class:`soap.expr.Expr` or str
    :param var_env: The ranges of input variables.
    :type var_env: dictionary containing mappings from variables to
        :class:`soap.semantics.error.Interval`
    :param depth: The depth limit.
    :type depth: int
    :param prec: Precision used to evaluate the expression, defaults to
        single precision.
    :type prec: int
    """
    return reduce(MartelTraceExpr(tree).traces(var_env, depth, prec, **kwargs))


def greedy_trace(tree, var_env=None, depth=2, prec=None, **kwargs):
    """Finds our equivalent expressions using :class:`GreedyTraceExpr`.

    :param tree: The original expression.
    :type tree: :class:`soap.expr.Expr` or str
    :param var_env: The ranges of input variables.
    :type var_env: dictionary containing mappings from variables to
        :class:`soap.semantics.error.Interval`
    :param depth: The depth limit.
    :type depth: int
    :param prec: Precision used to evaluate the expression, defaults to
        single precision.
    :type prec: int
    """
    return reduce(GreedyTraceExpr(tree).traces(var_env, depth, prec, **kwargs))


def frontier_trace(tree, var_env=None, depth=2, prec=None, **kwargs):
    """Finds our equivalent expressions using :class:`FrontierTraceExpr`.

    :param tree: The original expression.
    :type tree: :class:`soap.expr.Expr` or str
    :param var_env: The ranges of input variables.
    :type var_env: dictionary containing mappings from variables to
        :class:`soap.semantics.error.Interval`
    :param depth: The depth limit.
    :type depth: int
    :param prec: Precision used to evaluate the expression, defaults to
        single precision.
    :type prec: int
    """
    return reduce(FrontierTraceExpr(tree).traces(
                  var_env, depth, prec, **kwargs))
