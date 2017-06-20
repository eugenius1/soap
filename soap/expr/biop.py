"""
.. module:: soap.expr.biop
    :synopsis: The class of expressions.
"""
import gmpy2

from soap.common import Comparable, Flyweight, cached, ignored
from soap.expr.common import (
    ADD_OP, MULTIPLY_OP, BARRIER_OP,
    COMMUTATIVITY_OPERATORS, PRIMITIVE_OPERATORS_WITH_2_TERMS,
    ADD3_OP, CONSTANT_MULTIPLY_OP, FMA_OP,
)
from soap.expr.parser import parse
import soap.logger as logger


class Expr(Comparable, Flyweight):
    """The expression class."""

    __slots__ = ('op', 'a1', 'a2', '_hash')

    def __init__(self, *args, **kwargs):
        """Expr allows several ways of instantiation for the expression example
        (a + b)::

            1. ``Expr('+', a, b)``
            2. ``Expr(op='+', a1=a, a2=b)``
            3. ``Expr('+', operands=[a, b])``
        """
        # logger.debug('Expr(*{}, **{})'.format(args, kwargs))
        if not args and not kwargs:
            logger.error('class Expr: no `args`')
            logger.error('  kwargs={}'.format(kwargs))

        if kwargs:
            # Accept `operands` or `al`, in that order of preference
            if 'al' in kwargs and 'operands' not in kwargs:
                kwargs['operands'] = kwargs['al']
            op = kwargs.setdefault('op')
            if op == None and len(args) > 0:
                # Expr(op_, operands=operands_), ie. one positional and some keyword arguments
                op = args[0]
            operands = kwargs.setdefault('operands')
            if operands == None:
                # Set legacy members to None, if no operands nor a1 and a2
                a1 = kwargs.setdefault('a1')
                a2 = kwargs.setdefault('a2')
                operands = (a1, a2)
        # Expr(obj), with obj.op and obj.args available
        elif len(args) == 1:
            expr = list(args).pop()
            try:
                op, operands = expr.op, expr.args
            except AttributeError:
                expr = parse(expr, self.__class__)
            try:
                op, operands = expr.op, expr.args
            except AttributeError:
                raise ValueError('String is not an expression')
        # Expr(op, a1) or Expr(op, [a1, a2, a3,...])
        elif len(args) == 2:
            op, operands = args
            if not isinstance(operands, (list, tuple)):
                # put number into tuple
                operands = (operands,)
        # Expr(op, a1, a2, a3,...)
        else: # len(args) > 2:
            op, *operands = args
        operands = list(operands)
        if op == None:
            logger.error('Expr(*{}, **{}), setting op=None'.format(args, kwargs))
        if op == CONSTANT_MULTIPLY_OP:
            from soap.semantics.common import Label
            from numbers import Number
            if not isinstance(operands[0], (Number, str, Label)):
                logger.error(CONSTANT_MULTIPLY_OP, operands)
                raise TypeError
        self.op = op
        self.operands = operands
        # legacy: self.a1, .a2, .a3, ...
        for index, a in enumerate(operands):
            setattr(self, 'a{}'.format(index + 1), a)
        super().__init__()

    def __getnewargs__(self):
        return (self.op, *(self.operands))

    def tree(self):
        """Produces a tuple tree for the expression."""
        def to_tuple(a):
            if isinstance(a, Expr):
                return a.tree()
            return a
        return (self.op, *tuple([to_tuple(arg) for arg in self.operands]))

    @property
    def args(self):
        """Returns the arguments of the expression"""
        return self.operands

    @property
    def inputs(self):
        """Returns the inputs to the synthesised expression evaluator"""
        if self.op == CONSTANT_MULTIPLY_OP:
            # exclude the constant
            return self.operands[1:]
        return self.operands


    #@cached # caused unexplainable issues with benchmark fdtd_1
    def error(self, var_env, prec):
        """Computes the error bound of its evaulation.

        :param var_env: The ranges of input variables.
        :type var_env: dictionary containing mappings from variables to
            :class:`soap.semantics.error.Interval`
        :param prec: Precision used to evaluate the expression, defaults to
            single precision.
        :type prec: int
        """
        from soap.semantics import precision_context, error_for_operand
        with precision_context(prec):
            errors = [error_for_operand(o, var_env, prec) for o in self.args]
            return errors[0].do_op(self.op, errors[1:])


    def exponent_width(self, var_env, prec, return_bounds=False, output_only=False):
        """Computes the exponent width required for its evaluation so that no
        overflow could occur, at any of the inputs or output.

        :param var_env: The ranges of input variables.
        :type var_env: dictionary containing mappings from variables to
            :class:`soap.semantics.error.Interval`
        :param prec: Precision used to evaluate the expression, defaults to
            single precision.
        :type prec: int
        """
        import math
        from soap.common import (
            standard_exponent_size_for, exponent_size_for_exponent, exponent_for_value,
        )
        from soap.semantics.error import Interval, error_for_operand
        from soap.semantics.flopoco import we_min

        we = None
        exp_lower_bound = None
        bounds = self.error(var_env, prec).v
        if not output_only:
            for arg in self.inputs:
                arg_bounds = error_for_operand(arg, var_env, prec).v
                bounds.min = min(bounds.min, arg_bounds.min)
                bounds.max = max(bounds.max, arg_bounds.max)
        # if bounds cover infinitesimal numbers, fallback to IEEE standard
        if bounds.min <= 0 and bounds.max >= 0:
            we = standard_exponent_size_for(prec)
            exp_lower_bound = -(2**(we-1) -2)
        if we == None or return_bounds:
            exp_bounds_ = list(map(exponent_for_value, bounds))
            exp_bounds = [min(exp_bounds_), max(exp_bounds_)]
            if return_bounds:
                if exp_lower_bound != None:
                    exp_bounds[0] = exp_lower_bound
                return Interval(exp_bounds)
            else:
                we = max(map(exponent_size_for_exponent, exp_bounds))
        return max(we, we_min)


    def exponent_bounds(self, var_env, prec, output_only=True):
        return self.exponent_width(var_env, prec, return_bounds=True, output_only=output_only)


    @cached
    def area(self, var_env, prec):
        """Computes the area estimation of its evaulation.

        :param var_env: The ranges of input variables.
        :type var_env: dictionary containing mappings from variables to
            :class:`soap.semantics.error.Interval`
        :param prec: Precision used to evaluate the expression, defaults to
            single precision.
        :type prec: int
        """
        from soap.semantics import AreaSemantics
        return AreaSemantics(self, var_env, prec)

    @cached
    def real_area(self, var_env, prec):
        """Computes the actual area by synthesising it using XST with flopoco
        cores.

        :param var_env: The ranges of input variables.
        :type var_env: dictionary containing mappings from variables to
            :class:`soap.semantics.error.Interval`
        :param prec: Precision used to evaluate the expression, defaults to
            single precision.
        :type prec: int
        """
        from soap.semantics.flopoco import eval_expr
        return eval_expr(self, var_env, prec)

    @cached
    def as_labels(self):
        """Performs labelling analysis on the expression.

        :returns: dictionary containing the labelling scheme.
        """
        from soap.semantics import Label

        def to_label(e):
            try:
                return e.as_labels()
            except AttributeError:
                l = Label(e)
                return l, {l: e}

        labels, mappings = [], []
        for arg in self.operands:
            l, s = to_label(arg)
            labels.append(l)
            mappings.append(s)
        
        e = BExpr(op=self.op, operands=labels)
        l = Label(e)
        s = {l: e}
        for mapping in mappings:
            s.update(mapping)
        return l, s

    def crop(self, depth):
        """Truncate the tree at a certain depth.

        :param depth: the depth used to truncate the tree.
        :type depth: int
        :returns: the truncated tree and a dictionary containing truncated
            subexpressions.
        """
        def subcrop(a):
            try:
                return a.crop(depth - 1)
            except AttributeError:
                return a, {}
        if depth > 0:
            labels, mappings = [], []
            for arg in self.operands:
                l, s = subcrop(arg)
                labels.append(l)
                mappings.append(s)
            s = mappings[0]
            for mapping in mappings[1:]:
                s.update(mapping)
            return self.__class__(self.op, operands=labels), s
        from soap.semantics import Label
        l = Label(self)
        return l, {l: self}

    def stitch(self, env):
        """Undo truncation by stiching truncated subexpressions back to the
        leaves of the expression.

        :param env: the truncated expressions.
        :type env: dict
        :returns: new expression tree.
        """
        def substitch(a):
            try:
                return a.stitch(env)
            except AttributeError:
                pass
            try:
                return env[a]
            except KeyError:
                return a
        return self.__class__(self.op, [substitch(arg) for arg in self.operands])

    def __iter__(self):
        return iter((self.op, *self.operands))

    def __str__(self):
        if self.op in PRIMITIVE_OPERATORS_WITH_2_TERMS:
            a1, a2 = sorted([str(self.a1), str(self.a2)])
            return '(%s %s %s)' % (a1, self.op, a2)
        else:
            # eg. "add3(a, b, c)" or "op()"
            params = ', '.join(map(str, self.args))
            return '{op}({params})'.format(op=self.op, params=params)

    def __repr__(self, mode=0):
        if mode == 1:
            return "Expr(op='%s', a1=%s, a2=%s)" % \
                (self.op, repr(self.a1), repr(self.a2))
        else:
            return '{cls}({op}, {params})'.format(
                cls=self.__class__.__name__, op=self.op, params=params)

    def do_op(self, op, others=[], **kwargs):
        """Custom operator on Expr
        No need to duplicate self in others.
        """
        # Natural operators
        if op == ADD_OP:
            return self + others[0]
        elif op == SUBTRACT_OP:
            return self - others[0]
        elif op == MULTIPLY_OP:
            return self * others[0]
        elif op == BARRIER_OP:
            return self | others[0]
        
        # Custom operators
        # assuming self is the first operand
        if op == ADD3_OP:
            return Expr(op=op, operands=[self]+others[:2])
        elif op == CONSTANT_MULTIPLY_OP:
            return Expr(op=op, operands=[self]+others[:1])
        elif op == FMA_OP:
            return Expr(op=op, operands=[self]+others[:2])
            
    def __add__(self, other):
        return Expr(op=ADD_OP, a1=self, a2=other)

    def __mul__(self, other):
        return Expr(op=MULTIPLY_OP, a1=self, a2=other)

    def __or__(self, other):
        return Expr(op=BARRIER_OP, a1=self, a2=other)

    def _symmetric_id(self):
        if self.op in COMMUTATIVITY_OPERATORS:
            _sym_id = (self.op, frozenset(self.args))
        elif self.op == FMA_OP:
            _sym_id = (self.op, frozenset(self.args[:2]), self.args[2])
        else:
            _sym_id = tuple(self)
        return _sym_id

    def __eq__(self, other):
        if not isinstance(other, Expr):
            return False
        if self.op != other.op:
            return False
        if id(self) == id(other):
            return True
        return self._symmetric_id() == other._symmetric_id()

    def __lt__(self, other):
        if not isinstance(other, Expr):
            return False
        return self._symmetric_id() < other._symmetric_id()

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            pass
        self._hash = hash(self._symmetric_id())
        return self._hash


class BExpr(Expr):
    """An expression class that only allows non-expression arguments.

    This is a subclass of :class:`Expr`.
    """

    __slots__ = Expr.__slots__

    def __init__(self, **kwargs):
        from soap.semantics import Label
        super().__init__(**kwargs)
        for arg in self.operands:
            if not isinstance(arg, Label):
                raise ValueError(
                    'BExpr allows only binary expressions. Got {obj} of type {typ}'.format(
                        obj=arg, typ=type(obj)))


if __name__ == '__main__':
    logger.set_context(level=logger.levels.debug)
    r = Expr("""(a + a + b) * (a + b + b) * (b + b + c) *
                (b + c + c) * (c + c + a) * (c + a + a)""")
    n, e = r.crop(1)
    print('cropped', n, e)
    print('stitched', n.stitch(e))
    print(r)
    print(repr(r))
    v = {
        'a': ['1', '2'],
        'b': ['10', '20'],
        'c': ['100', '200'],
    }
    print(v)
    prec = gmpy2.ieee(32).precision - 1
    print(r.error(v, prec))
    for l, e in r.as_labels()[1].items():
        print(str(l), ':', str(e))
    print(r.area(v, prec))
    # print(r.real_area(v, prec))
