from tests.benchmarks.polybench import dictionary as polybench_dict
from tests.benchmarks.livermore import dictionary as livermore_dict
from tests.benchmarks.soap3_results import soap3_results

class BenchmarkExpr(object):
    """Objects of BenchmarkExpr hold the relevant data about a particular benchmark expression."""
    def __init__(self, e, v, max_transformation_depth=None, *args, **kwargs):
        super(BenchmarkExpr, self).__init__()
        self.e = e
        self.v = v
        self.max_transformation_depth = max_transformation_depth
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
        return '{cls}(e={e}, v={v}, name={name}, max_transformation_depth={mtd})'.format(
            cls=self.__class__.__name__, e=repr(self.e), v=self.v, name=repr(self.name),
            mtd=max_transformation_depth)

    __str__ = __repr__
        

# A benchmark names starting with _ (an underscore) means its name will be hidden
custom_benchmarks_dict = {
    '_filter': {
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
    '_taylor_b': {
        'e': 'b * ((2 * i) + 1) * (2 * i)',
        'v': {
            'b': [0, 4], # ~ product of (4*i^2) from i=1 to 20 is (4^20 * (20!)^2) ~ 7e48
            'i': [1, 20],
        }
    },
    '_taylor_p': {
        'e': 'p * (x + y) * (x + y)',
        'v': {
            'p': [0, 1.21**40], # (x+y)^20 ~ 2048
            'x': [-0.1, 0.1],
            'y': [0, 1],
        }
    },
    '_seidel_adds': {
        'e': '(a+b+c+d+e)',
        'v': {
            'a': [0, 1],
            'b': [0, 1],
            'c': [0, 1],
            'd': [0, 1],
            'e': [0, 1],
        }
    },
    '_seidel_var': {
        'e': 'a+b',
        'v': {
            'a': [0, 1],
            'b': [0, 1],
            'c': [0, 1],
            'd': [0, 1],
            'e': [0, 1],
        }
    },
    '_e6': {
        'e': """
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
            )""",
        'v': {
            'a': ['1', '2'],
            'b': ['10', '20'],
            'c': ['100', '200'],
        }
    },
    '_e6_alt': {
        'e': """
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
            )""",
        'v': {
            'a': ['1', '2'],
            'b': ['10', '20'],
            'c': ['100', '200'],
        }
    },
    '_e2': {
        'e': '((a + a) + b) * ((a + b) + b)',
        'v': {
            'a': ['1', '2'],
            'b': ['10', '20'],
            'c': ['100', '200'],
        }
    },
}

basics_dict = {
    '_add': {
        'e': 'a + b',
        'v': {
            'a': [1,100],
            'b': [0.01,1],
        }
    },
    '_add3': {
        'e': 'a + b + c',
        'v': {
            'a': [1,100],
            'b': [0.01,1],
            'c': [0.1,10],
        }
    },
    '_add_3': {
        'e': 'a + b + c',
        'v': {
            'a': [0,1],'b': [0,1],'c': [0,1],'d': [0,1],'e': [0,1],'f': [0,1],'g': [0,1],'h': [0,1],'i': [0,1],'j': [0,1],
        }
    },
    '_add_9': {
        'e': 'a + b + c+d+e+f+g+h+i',
        'v': {
            'a': [0,1],'b': [0,1],'c': [0,1],'d': [0,1],'e': [0,1],'f': [0,1],'g': [0,1],'h': [0,1],'i': [0,1],'j': [0,1],
        }
    },
    '_add_10': {
        'e': 'a + b + c+d+e+f+g+h+i+j',
        'v': {
            'a': [0,1],'b': [0,1],'c': [0,1],'d': [0,1],'e': [0,1],'f': [0,1],'g': [0,1],'h': [0,1],'i': [0,1],'j': [0,1],
        }
    },
    '_add_20': {
        'e': 'a + b + c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t',
        'v': {
            'a': [0,1],'b': [0,1],'c': [0,1],'d': [0,1],'e': [0,1],'f': [0,1],'g': [0,1],'h': [0,1],'i': [0,1],'j': [0,1],
            'k': [0,1],'l': [0,1],'m': [0,1],'n': [0,1],'o': [0,1],'p': [0,1],'q': [0,1],'r': [0,1],'s': [0,1],'t': [0,1],        }
    },
    '_add3_zero_sum': {
        'e': 'a + b + c',
        'v': {
            'a': [100,101],
            'b': [0.01,1],
            'c': [-101,-100],
        }
    },
    '_mul': {
        'e': 'a * b',
        'v': {
            'a': [1,100],
            'b': [0.01,1],
        }
    },
    '_fma': {
        'e': '(a * b) + c',
        'v': {
            'a': [1,100],
            'b': [0.01,1],
            'c': [0.1,10],
        }
    },
    '_doubler': {
        'e': 'a * 2',
        'v': {
            'a': [0, 1],
        }
    },
    '_long_pi': {
        'e': 'a * 3.1415926535897932384626433832795',
        'v': {
            'a': [-1,1]
        }
    },
    '_cmvmc': {
        'e': '3 * a * 2',
        'v': {
            'a': [1,100],
        }
    },
}

# only include from benchmark suites
# copying
benchmarks_dict = dict(polybench_dict)
benchmarks_dict.update(livermore_dict)

# include basics and custom in all
all_benchmarks_dict = dict(benchmarks_dict)
all_benchmarks_dict.update(**custom_benchmarks_dict, **basics_dict)

# dict of dicts to dict of BenchmarkExprs
def _convert_from_dict(dictionary):
    result = {}
    for name in dictionary:
        result[name] = BenchmarkExpr(**dictionary[name], name=name)
    return result

polybench_benchmarks = _convert_from_dict(polybench_dict)
livermore_benchmarks = _convert_from_dict(livermore_dict)
benchmarks = dict(polybench_benchmarks)
benchmarks.update(livermore_benchmarks)
number_in_benchmark_suites = len(benchmarks)

custom_benchmarks = _convert_from_dict(custom_benchmarks_dict)
basics_benchmarks = _convert_from_dict(basics_dict)

all_benchmarks = dict(benchmarks)
all_benchmarks.update(**custom_benchmarks, **basics_benchmarks)

def get_by_name(names, warning_print=print):
    def filter_dict_for_key(dictionary, name):
        return {name: dictionary[name]}

    if isinstance(names, str):
        names = names.split(',')
        # if names is still just a string, return
        if isinstance(names, str):
            return filter_dict_for_key(all_benchmarks, names)
    # lower-case everything
    names = list(map(lambda s:s.lower(), names))
    if 'all' in names or 'a' in names:
        return all_benchmarks
    result = {}
    for name in names:
        if name in ('custom', 'c'):
            result.update(custom_benchmarks)
        elif name in ('polybench', 'p'):
            result.update(polybench_benchmarks)
        elif name in ('livermore', 'l'):
            result.update(livermore_benchmarks)
        elif name in ('basics', 'basic'):
            result.update(basics_benchmarks)
        elif name in ('suite', 'suites', 's', 'benchmarks'):
            result.update(benchmarks)
        else:
            try:
                result.update(filter_dict_for_key(all_benchmarks, name))
            except KeyError:
                error_message = '{!r} not found'.format(name)
                if len(names) == 1:
                    raise KeyError(error_message)
                else:
                    warning_print(error_message)
    return result
