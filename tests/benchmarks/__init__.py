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
            cls=self.__class__.__name__, e=self.e, v=self.v, name=self.name)

    __str__ = __repr__
        

benchmarks_dict = {
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
        'e': 'p * (x + y) * (x + y)',
        'v': {
            'p': [0, 1.21**40], # (x+y)^20 ~ 2048
            'x': [-0.1, 0.1],
            'y': [0, 1],
        }
    },


}

benchmarks = {}
for name, params in benchmarks_dict.items():
    benchmarks[name] = BenchmarkExpr(params['e'], params['v'], name=name)
