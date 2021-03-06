"""
.. module:: soap.transformer.core
    :synopsis: Base class for transforming expression instances.
"""
import sys
import functools
import multiprocessing

import soap.logger as logger
from soap.common import cached
from soap.expr.common import is_expr
from soap.expr import Expr


RECURSION_LIMIT = sys.getrecursionlimit()


def item_to_list(f):
    """Utility decorator for equivalence rules.

    :param f: The transform function.
    :type f: function
    :returns: A new function that returns a list containing the return value of
        the original function.
    """
    def wrapper(t):
        v = f(t)
        return [v] if not v is None else []
    return functools.wraps(f)(wrapper)


def none_to_list(f):
    """Utility decorator for equivalence rules.

    :param f: The transform function.
    :type f: function
    :returns: A new function that returns an empty list if the original
        function returns None.
    """
    def wrapper(t):
        v = f(t)
        return v if not v is None else []
    return functools.wraps(f)(wrapper)


class ValidationError(Exception):
    """Failed to find equivalence."""


class TreeTransformer(object):
    """The base class that finds the transitive closure of transformed trees.

    :param tree_or_trees: An input expression, or a container of equivalent
        expressions.
    :type tree_or_trees: :class:`soap.expr.Expr`
    :param validate: If set, tries to validate equivalence, subclasses
        of :class:`TreeTransformer` should implement a member function
        :member:`TreeTransformer.validate`.
    :type validate: bool
    :param depth: The depth limit for equivalence finding, if not specified, a
        depth limit will not be used.
    :type depth: int or None
    :param step_plugin: A plugin function which is called after one step of
        transitive closure, it should take as argument a set of trees and
        return a new set of trees.
    :type step_plugin: function
    :param reduce_plugin: A plugin function which is called after one step of
        reduction, it should take as argument a set of trees and turn a new set
        of trees.
    :type reduce_plugin: function
    :param multiprocessing: If set, the class will multiprocess when computing
        new equivalent trees.
    :type multiprocessing: bool
    """
    transform_methods = None
    reduction_methods = None

    def __init__(self, tree_or_trees,
                 validate=False, depth=None,
                 step_plugin=None, reduce_plugin=None,
                 multiprocessing=True):
        try:
            self._t = [Expr(tree_or_trees)]
        except TypeError:
            self._t = tree_or_trees
        self._v = validate
        self._m = multiprocessing
        self._d = depth or RECURSION_LIMIT
        self._n = {}
        self._sp = step_plugin
        self._rp = reduce_plugin
        self.transform_methods = list(self.__class__.transform_methods or [])
        self.reduction_methods = list(self.__class__.reduction_methods or [])
        super().__init__()

    def _harvest(self, trees):
        """Crops all trees at the depth limit."""
        if self._d >= RECURSION_LIMIT:
            return trees
        logger.debug('Harvesting trees.')
        cropped = []
        for t in trees:
            try:
                t, e = t.crop(self._d)
            except AttributeError:
                t, e = t, {}
            cropped.append(t)
            self._n.update(e)
        return cropped

    def _seed(self, trees):
        """Stitches all trees."""
        logger.debug('Seeding trees.')
        if not self._n:
            return trees
        seeded = set()
        for t in trees:
            try:
                t = t.stitch(self._n)
            except AttributeError:
                pass
            seeded.add(t)
        return seeded

    def _plugin(self, trees, plugin):
        """Plugin function call setup and cleanup."""
        if not plugin:
            return trees
        trees = self._seed(trees)
        trees = plugin(trees)
        trees = set(self._harvest(trees))
        return trees

    def _step(self, s, c=False, d=None):
        """One step of the transitive closure."""
        fs = self.transform_methods if c else self.reduction_methods
        v = self._validate if self._v else None
        if self._m:
            cpu_count = multiprocessing.cpu_count()
            chunksize = int(len(s) / cpu_count) + 1
            map = pool().imap_unordered
        else:
            cpu_count = chunksize = 1
            map = lambda f, l, _: [f(a) for a in l]
        union = lambda s, _: functools.reduce(lambda x, y: x | y, s)
        r = [(t, fs, v, c, d) for i, t in enumerate(s)]
        b, r = list(zip(*map(_walk, r, chunksize)))
        s = {t for i, t in enumerate(s) if b[i]}
        r = union(r, cpu_count)
        return s, r

    def _closure_r(self, trees, reduced=False):
        """Transitive closure algorithm."""
        if self._d > RECURSION_LIMIT and self.transform_methods:
            logger.warning('Depth limit not set.', self._d)
        done_trees = set()
        todo_trees = set(trees)
        i = 0
        while todo_trees:
            # print set size
            i += 1
            logger.persistent(
                'Iteration' if not reduced else 'Reduction', i)
            logger.persistent('Trees', len(done_trees))
            logger.persistent('Todo', len(todo_trees))
            if not reduced:
                _, step_trees = \
                    self._step(todo_trees, not reduced, None)
                step_trees -= done_trees
                step_trees = self._closure_r(step_trees, True)
                step_trees = self._plugin(step_trees, self._sp)
                done_trees |= todo_trees
                todo_trees = step_trees - done_trees
            else:
                nore_trees, step_trees = \
                    self._step(todo_trees, not reduced, None)
                step_trees = self._plugin(step_trees, self._rp)
                done_trees |= nore_trees
                todo_trees = step_trees - nore_trees
        logger.unpersistent('Iteration', 'Reduction', 'Trees', 'Todo')
        return done_trees

    def closure(self):
        """Perform transforms until transitive closure is reached.

        :returns: A set of trees after transform.
        """
        s = self._harvest(self._t)
        s = self._closure_r(s)
        s = self._seed(s)
        return s

    def validate(self, t, tn):
        """Perform validation of tree, subclasses must implement this method
        if validation of equivalence is needed. This is useful for discovering
        bugs in equivalence rules.

        :param t: An original tree.
        :type t: :class:`soap.expr.Expr`
        :param tn: A transformed tree.
        :type tn: :class:`soap.expr.Expr`
        :raises: ValidationError
        """
        pass

    def _validate(self, t, tn):
        if t == tn:
            return
        if self._v:
            self.validate(t, tn)


def _walk(t_fs_v_c_d):
    t, fs, v, c, d = t_fs_v_c_d
    s = set()
    d = d if c and d else RECURSION_LIMIT
    for f in fs:
        s |= _walk_r(t, f, v, d)
    if c:
        s.add(t)
    return True if not c and not s else False, s


@cached
def _walk_r(t, f, v, d):
    """Tree walker"""
    s = set()
    if d == 0:
        return s
    if not is_expr(t):
        return s
    for e in f(t):
        s.add(e)
    from soap.expr.common import (
        ADD_OP, MULTIPLY_OP,
        ADD3_OP, CONSTANT_MULTIPLY_OP,
        FMA_OP,
    )
    if t.op in (ADD3_OP, MULTIPLY_OP, ADD3_OP, FMA_OP):
        if t.op == FMA_OP:
            # (a * b) + c == (b * a) + c
            commutative_args = t.args[:2]
        else:
            commutative_args = t.args
        for index, a in enumerate(commutative_args):
            for e in _walk_r(a, f, v, d - 1):
                s.add(Expr(t.op, [e]+t.args[:index]+t.args[index+1:]))
    if not v:
        return s
    try:
        for tn in s:
            v(t, tn)
    except ValidationError:
        logger.error('Violating transformation:', f.__name__)
        raise
    return s


def par_union(sl):
    return functools.reduce(lambda s, t: s | t, sl)


_pool = None


def pool():
    global _pool
    if _pool is None:
        _pool = multiprocessing.Pool()
    return _pool


def _iunion(sl, no_processes):
    """Parallel set union, slower than serial implementation."""
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]
    chunksize = int(len(sl) / no_processes) + 2
    while len(sl) > 1:
        sys.stdout.write('\rUnion: %d.' % len(sl))
        sys.stdout.flush()
        sl = list(pool().imap_unordered(par_union, chunks(sl, chunksize)))
    return sl.pop()
