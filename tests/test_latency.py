import unittest

from soap.context import context
from soap.datatype import int_type, real_type, RealArrayType, ArrayType
from soap.expression import operators, Variable, Subscript, expression_factory
from soap.parser import parse
from soap.semantics.error import IntegerInterval
from soap.semantics.functions.label import label
from soap.semantics.label import Label
from soap.semantics.schedule.extract import (
    ForLoopExtractor, ForLoopNestExtractor
)
from soap.semantics.schedule.common import (
    SEQUENTIAL_LATENCY_TABLE, LOOP_LATENCY_TABLE
)
from soap.semantics.schedule.distance import (
    dependence_vector, dependence_distance, ISLIndependenceException
)
from soap.semantics.schedule.graph import (
    SequentialLatencyDependenceGraph, LoopLatencyDependenceGraph, latency_eval
)
from soap.semantics.state import flow_to_meta_state


class TestDependenceCheck(unittest.TestCase):
    def setUp(self):
        self.x = Variable('x', dtype=int_type)
        self.y = Variable('y', dtype=int_type)
        self.sx = slice(0, 10, 1)
        self.sy = slice(1, 11, 1)

    def test_simple_subscripts(self):
        source = Subscript(
            expression_factory(operators.ADD_OP, self.x, IntegerInterval(1)))
        sink = Subscript(self.x)
        dist_vect = dependence_vector([self.x], [self.sx], source, sink)
        self.assertEqual(dist_vect, (1, ))
        dist = dependence_distance(dist_vect, [self.sx])
        self.assertEqual(dist, 1)

    def test_simple_independence(self):
        source = Subscript(
            expression_factory(operators.ADD_OP, self.x, IntegerInterval(20)))
        sink = Subscript(self.x)
        self.assertRaises(
            ISLIndependenceException, dependence_vector,
            [self.x], [self.sx], source, sink)

    def test_multi_dim_subscripts(self):
        # for (x in 0...9) for (y in 1...10) a[x + 2, y] = ... a[x, y - 1] ...
        expr = expression_factory(operators.ADD_OP, self.x, IntegerInterval(2))
        source = Subscript(expr, self.y)
        expr = expression_factory(
            operators.SUBTRACT_OP, self.y, IntegerInterval(1))
        sink = Subscript(self.x, expr)
        iter_slices = [self.sx, self.sy]
        dist_vect = dependence_vector(
            [self.x, self.y], iter_slices, source, sink)
        self.assertEqual(dist_vect, (2, 1))
        dist = dependence_distance(dist_vect, iter_slices)
        self.assertEqual(dist, 21)

    def test_multi_dim_coupled_subscripts_independence(self):
        # for (x in 0...9) { a[x + 1, x + 2] = a[x, x]; }
        expr_1 = expression_factory(
            operators.ADD_OP, self.x, IntegerInterval(1))
        expr_2 = expression_factory(
            operators.ADD_OP, self.x, IntegerInterval(2))
        source = Subscript(expr_1, expr_2)
        sink = Subscript(self.x, self.x)
        self.assertRaises(
            ISLIndependenceException, dependence_vector,
            [self.x], [self.sx], source, sink)

    def test_multi_dim_coupled_subscripts_dependence(self):
        # for (x in 0...9) { a[x + 1, x + 1] = a[x, x]; }
        expr_1 = expression_factory(
            operators.ADD_OP, self.x, IntegerInterval(1))
        expr_2 = expression_factory(
            operators.ADD_OP, self.x, IntegerInterval(1))
        source = Subscript(expr_1, expr_2)
        sink = Subscript(self.x, self.x)
        dist_vect = dependence_vector([self.x], [self.sx], source, sink)
        self.assertEqual(dist_vect, (1, ))


class _CommonMixin(unittest.TestCase):
    def setUp(self):
        self.ori_ii_prec = context.ii_precision
        context.ii_precision = 40
        self.x = Variable('x', real_type)
        self.y = Variable('y', real_type)
        self.a = Variable('a', RealArrayType([30]))
        self.b = Variable('b', RealArrayType([30, 30]))
        self.i = Variable('i', int_type)

    def tearDown(self):
        context.ii_precision = self.ori_ii_prec


class TestLoopLatencyDependenceGraph(_CommonMixin):
    def test_variable_initiation(self):
        program = """
        def main(real x) {
            for (int i = 0; i < 9; i = i + 1) {
                x = x + 1;
            }
            return x;
        }
        """
        fix_expr = flow_to_meta_state(parse(program))[self.x]
        graph = LoopLatencyDependenceGraph(fix_expr)

        ii = graph.initiation_interval()
        expect_ii = LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        self.assertAlmostEqual(ii, expect_ii)

        trip_count = graph.trip_count()
        self.assertEqual(trip_count, 9)

        latency = graph.latency()
        expect_latency = (trip_count - 1) * ii + graph.depth()
        self.assertAlmostEqual(latency, expect_latency)

    def test_array_independence_initiation(self):
        program = """
        def main(real[30] a) {
            for (int i = 0; i < 9; i = i + 1) {
                a[i] = a[i] + 1;
            }
            return a;
        }
        """
        fix_expr = flow_to_meta_state(parse(program))[self.a]
        graph = LoopLatencyDependenceGraph(fix_expr)
        ii = graph.initiation_interval()
        expect_ii = 1
        self.assertEqual(ii, expect_ii)

    def test_simple_array_initiation(self):
        program = """
        def main(real[30] a) {
            for (int i = 0; i < 9; i = i + 1) {
                a[i] = a[i - 3] + 1;
            }
            return a;
        }
        """
        fix_expr = flow_to_meta_state(parse(program))[self.a]
        graph = LoopLatencyDependenceGraph(fix_expr)
        ii = graph.initiation_interval()
        expect_ii = LOOP_LATENCY_TABLE[real_type, operators.INDEX_ACCESS_OP]
        expect_ii += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        expect_ii += LOOP_LATENCY_TABLE[ArrayType, operators.INDEX_UPDATE_OP]
        expect_ii /= 3
        self.assertAlmostEqual(ii, expect_ii)


class TestLatencyEvaluation(_CommonMixin):
    def test_simple_flow(self):
        program = """
        def main(real[30] a) {
            for (int i = 0; i < 10; i = i + 1) {
                a[i + 3] = a[i] + i;
            }
            return a;
        }
        """
        meta_state = flow_to_meta_state(parse(program))
        distance = 3
        trip_count = 10
        latency = latency_eval(meta_state, [self.a])
        depth = LOOP_LATENCY_TABLE[real_type, operators.INDEX_ACCESS_OP]
        depth += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        depth += LOOP_LATENCY_TABLE[ArrayType, operators.INDEX_UPDATE_OP]
        expect_ii = depth / distance
        expect_latency = expect_ii * (trip_count - 1) + depth
        self.assertAlmostEqual(latency, expect_latency)

    def test_non_pipelineable(self):
        program = """
        def main(real[30] a, int j) {
            for (int i = 0; i < 10; i = i + j) {
                a[i + 3] = a[i] + i;
            }
            return a;
        }
        """
        meta_state = flow_to_meta_state(parse(program))
        latency = latency_eval(meta_state, [self.a])
        expect_latency = float('inf')
        self.assertAlmostEqual(latency, expect_latency)

    def test_loop_nest_flow(self):
        program = """
        def main(real[30] a) {
            int j = 0;
            while (j < 10) {
                int i = 0;
                while (i < 10) {
                    a[i + j + 3] = a[i + j] + i;
                    i = i + 1;
                }
                j = j + 1;
            }
            return a;
        }
        """
        meta_state = flow_to_meta_state(parse(program))
        latency = latency_eval(meta_state[self.a], [self.a])
        distance = 3
        trip_count = 10 * 10
        depth = LOOP_LATENCY_TABLE[real_type, operators.INDEX_ACCESS_OP]
        depth += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        depth += LOOP_LATENCY_TABLE[ArrayType, operators.INDEX_UPDATE_OP]
        expect_ii = depth / distance
        add_latency = LOOP_LATENCY_TABLE[int_type, operators.ADD_OP]
        expect_latency = expect_ii * (trip_count - 1) + depth + add_latency
        self.assertAlmostEqual(latency, expect_latency)

    def test_multi_dim_flow(self):
        program = """
        def main(real[30, 30] b, int j) {
            for (int i = 0; i < 10; i = i + 1) {
                b[i + 3, j] = (b[i, j] + i) + j;
            }
            return b;
        }
        """
        meta_state = flow_to_meta_state(parse(program))
        latency = latency_eval(meta_state[self.b], [self.b])
        distance = 3
        trip_count = 10
        depth = LOOP_LATENCY_TABLE[real_type, operators.INDEX_ACCESS_OP]
        depth += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        depth += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        depth += LOOP_LATENCY_TABLE[ArrayType, operators.INDEX_UPDATE_OP]
        expect_ii = depth / distance
        expect_latency = expect_ii * (trip_count - 1) + depth
        self.assertAlmostEqual(latency, expect_latency)

    def test_resource_constraint_on_ii(self):
        program = """
        def main(real[30] a) {
            for (int i = 0; i < 10; i = i + 1) {
                a[i] = a[i] + a[i + 1] + a[i + 2] + a[i + 3];
            }
            return a;
        }
        """
        meta_state = flow_to_meta_state(parse(program))
        latency = latency_eval(meta_state[self.a], [self.a])
        trip_count = 10
        depth = LOOP_LATENCY_TABLE[int_type, operators.ADD_OP]
        depth += LOOP_LATENCY_TABLE[real_type, operators.INDEX_ACCESS_OP]
        depth += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        depth += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        depth += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        depth += LOOP_LATENCY_TABLE[ArrayType, operators.INDEX_UPDATE_OP]
        expect_ii = 5 / 2
        expect_latency = (trip_count - 1) * expect_ii + depth
        self.assertAlmostEqual(latency, expect_latency)

    def test_sequence_of_loops(self):
        program = """
        def main(real[30] a) {
            for (int i = 0; i < 10; i = i + 1) {
                a[i] = a[i - 1] + i;
            }
            for (int j = 0; j < 20; j = j + 1) {
                a[j] = a[j - 2] + a[j];
            }
            return a;
        }
        """
        meta_state = flow_to_meta_state(parse(program))
        latency = latency_eval(meta_state[self.a], [self.a])
        kernel_lat = LOOP_LATENCY_TABLE[real_type, operators.INDEX_ACCESS_OP]
        kernel_lat += LOOP_LATENCY_TABLE[real_type, operators.ADD_OP]
        kernel_lat += LOOP_LATENCY_TABLE[ArrayType, operators.INDEX_UPDATE_OP]
        depth_1 = LOOP_LATENCY_TABLE[int_type, operators.ADD_OP] + kernel_lat
        depth_2 = max(
            LOOP_LATENCY_TABLE[int_type, operators.ADD_OP],
            LOOP_LATENCY_TABLE[real_type, operators.INDEX_ACCESS_OP])
        depth_2 += kernel_lat
        ii_1 = kernel_lat
        ii_2 = kernel_lat / 2
        trip_count_1 = 10
        trip_count_2 = 20
        latency_1 = (trip_count_1 - 1) * ii_1 + depth_1
        latency_2 = (trip_count_2 - 1) * ii_2 + depth_2
        self.assertAlmostEqual(latency, latency_1 + latency_2)


class TestExtraction(_CommonMixin):
    def compare(self, for_loop, expect_for_loop):
        for key, expect_val in expect_for_loop.items():
            self.assertEqual(getattr(for_loop, key), expect_val)

    def test_simple_loop(self):
        program = """
        def main(real[30] a) {
            for (int i = 1; i < 10; i = i + 1) {
                a[i] = a[i - 1] + 1;
            }
            return a;
        }
        """
        flow = parse(program)
        meta_state = flow_to_meta_state(flow)
        fix_expr = meta_state[flow.outputs[0]]
        for_loop = ForLoopExtractor(fix_expr)
        expect_for_loop = {
            'iter_var': self.i,
            'iter_slice': slice(1, 10, 1),
        }
        self.compare(for_loop, expect_for_loop)

    def test_nested_loop(self):
        program = """
        def main(real[30] a) {
            for (int i = 1; i < 10; i = i + 1) {
                for (int j = 0; j < 20; j = j + 3) {
                    a[i] = a[i - j] + 1;
                }
            }
            return a;
        }
        """
        flow = parse(program)
        meta_state = flow_to_meta_state(flow)

        a = flow.outputs[0]
        i = Variable('i', int_type)
        j = Variable('j', int_type)

        fix_expr = meta_state[a]
        for_loop = ForLoopNestExtractor(fix_expr)
        expect_for_loop = {
            'iter_vars': [i, j],
            'iter_slices': [slice(1, 10, 1), slice(0, 20, 3)],
            'kernel': fix_expr.loop_state[a].loop_state,
        }
        self.compare(for_loop, expect_for_loop)


class TestSequentialLatencyDependenceGraph(_CommonMixin):
    def _simple_dag(self):
        program = """
        def main(real w, real x, int y, int z) {
            x = (w + x) * (y + z) - (w + z);
            return x;
        }
        """
        flow = parse(program)
        meta_state = flow_to_meta_state(flow)
        outputs = flow.outputs
        lab, env = label(meta_state, None, outputs)
        return SequentialLatencyDependenceGraph(env, outputs)

    def test_simple_dag_schedule(self):
        graph = self._simple_dag()
        op_schedule = []
        for node, interval in graph.schedule().items():
            if not isinstance(node, Label):
                continue
            op_schedule.append((node.dtype, node.expr().op, interval))
        op_ints = [
            (int_type, operators.ADD_OP, (0, 1)),
            (real_type, operators.ADD_OP, (0, 4)),
            (real_type, operators.ADD_OP, (0, 4)),
            (real_type, operators.MULTIPLY_OP, (4, 7)),
            (real_type, operators.SUBTRACT_OP, (7, 11)),
        ]
        for op_int in op_ints:
            self.assertIn(op_int, op_schedule)

    def test_simple_dag_latency(self):
        graph = self._simple_dag()
        expect_latency = SEQUENTIAL_LATENCY_TABLE[real_type, operators.ADD_OP]
        expect_latency += SEQUENTIAL_LATENCY_TABLE[
            real_type, operators.MULTIPLY_OP]
        expect_latency += SEQUENTIAL_LATENCY_TABLE[
            real_type, operators.SUBTRACT_OP]
        self.assertEqual(graph.latency(), expect_latency)

    def test_simple_dag_resource(self):
        graph = self._simple_dag()
        print(graph.resource())
