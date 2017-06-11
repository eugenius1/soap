from tests.benchmarks.polybench import dictionary as polybench_dict
from tests.benchmarks.livermore import dictionary as livermore_dict

class BenchmarkExpr(object):
    """docstring for BenchmarkExpr"""
    def __init__(self, e, v, *args, **kwargs):
        super(BenchmarkExpr, self).__init__()
        self.e = e
        self.v = v
        self.name = kwargs.get('name', None)

    @property
    def expr(self):
        return self.e

    @property
    def var_env(self):
        return self.v

    def expr_and_vars(self):
        return (self.e, self.v)

    e_and_v = expr_and_vars

    def __repr__(self):
        return '{cls}(e={e}, v={v}, name={name})'.format(
            cls=self.__class__.__name__, e=repr(self.e), v=self.v, name=repr(self.name))

    __str__ = __repr__
        

custom_benchmarks_dict = {
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
    'taylor_b': {
        'e': 'b * (2 * i + 1) * (2 * i)',
        'v': {
            'b': [0, 7e48], # ~ product of (4*i^2) from i=1 to 20 is (4^20 * (20!)^2)
            'i': [1, 20],
        }
    },
    'taylor_p': {
        'e': 'p * (x + y) * (x + y)',
        'v': {
            'p': [0, 1.21**40], # (x+y)^20 ~ 2048
            'x': [-0.1, 0.1],
            'y': [0, 1],
        }
    },

}

# only include from benchmark suites
# copying
benchmarks_dict = dict(polybench_dict)
benchmarks_dict.update(livermore_dict)

# include custom too in all
all_benchmarks_dict = dict(benchmarks_dict)
all_benchmarks_dict.update(custom_benchmarks_dict)

# dict of dicts to dict of BenchmarkExpr's
benchmarks = {}
custom_benchmarks = {}
for name in benchmarks_dict:
    benchmarks[name] = BenchmarkExpr(**benchmarks_dict[name], name=name)
for name in custom_benchmarks_dict:
    custom_benchmarks[name] = BenchmarkExpr(**custom_benchmarks_dict[name], name=name)
all_benchmarks = dict(benchmarks)
all_benchmarks.update(custom_benchmarks)
