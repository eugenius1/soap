from collections import namedtuple, Mapping

AreaErrorTuple = namedtuple('AreaErrorTuple', ['area', 'error'])

def mins_of_analysis(analysis):
    mins = {}
    for key in ('area', 'error'):
        mins[key] = min(analysis, key=lambda d: d.get(key))

    return mins


def improvements(old, new):
    area_change = 1.0 - float(new['area']['area']) / old['area']['area']
    error_change = 1.0 - float(new['error']['error']) / old['error']['error']
    area_scaling = float(old['area']['area']) / new['area']['area']
    error_scaling = float(old['error']['error']) / new['error']['error']
    # the area cost of the best (lowest) error
    cost_of_error = float(new['error']['area']) / old['error']['area']
    return {
        # 0.75 (75%) less
        'change': AreaErrorTuple(area=area_change, error=error_change), 
        'expression': AreaErrorTuple(area=new['area']['expression'], error=new['error']['expression']),
        'cost_of_error': cost_of_error, # in units of area
        # 4x less
        'scaling': AreaErrorTuple(area=area_scaling, error=error_scaling)
    }


def run(timing=False):
    import time
    import gmpy2
    from pprint import pprint
    
    import soap.logger as logger
    from soap.common import profiled, timed, invalidate_cache
    from soap.analysis.utils import plot, analyse, analyse_and_frontier
    from soap.transformer.utils import greedy_frontier_closure, greedy_trace, frontier_trace, martel_trace
    from soap.analysis import Plot
    from soap.analysis.core import pareto_frontier_2d
    from soap.expr import Expr
    from soap.transformer.biop import (
        BiOpTreeTransformer, FusedBiOpTreeTransformer, FusedOnlyBiOpTreeTransformer,
        Add3TreeTransformer, ConstMultTreeTransformer,
    )

    from tests.benchmarks import benchmarks
    from tests.fused.analysis import improvements, mins_of_analysis

    
    Expr.__repr__ = Expr.__str__
    logger.set_context(level=logger.levels.info)
    single_prec = gmpy2.ieee(32).precision - 1

    # e = '(a + 1) * b | (b + 1) * a | a * b'
    e = '(a + b) + c'
    e_cm = '3 * a * 2'
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

    e, v = e_cm, v
    e, v = benchmarks['seidel'].expr_and_vars()
    t = Expr(e)
    logger.info('Expr:', str(t))
    logger.info('Tree:', t.tree())

    # fused unit = 3-input FP adder(s)
    actions = (
        (BiOpTreeTransformer, 'no fused'),
        (FusedBiOpTreeTransformer, 'any fused'),
        (FusedOnlyBiOpTreeTransformer, 'only fusing'),
        (Add3TreeTransformer, 'only add3 fusing'),
        (ConstMultTreeTransformer, 'only constMult fusing'),
    )[:1]
    plots = []
    transformer_results = []
    # with profiled(), timed():
    for trace_ in (
        (frontier_trace, 3),
        (greedy_trace, None))[1:]:
        for action in (actions, actions[::-1])[:1]: # forwards or backwards
            z = []
            frontier = []
            title = e.replace('\n', '').replace('  ', '').strip()
            p = Plot(depth=3, var_env=v, blocking=False, title=title)#,legend_pos='top right')
            for transformer_index, action_tuple in enumerate(action):
                Transformer, label = action_tuple
                if timing:
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
                    frontier_ = unfiltered
                else:
                    frontier_ = frontier
                logger.info(frontier_)
                # plot(frontier_, blocking=False)
                p.add(frontier_, legend=label, time=duration, annotate=True, annotate_kwargs={'fontsize': 10})
                z.append(set(map(lambda d:(d['area'], d['error'], d['expression']), frontier)))
                if transformer_index == 0:
                    original_mins = mins_of_analysis(frontier)
                else:
                    imp = improvements(original_mins, mins_of_analysis(frontier))
                    transformer_results.append((Transformer.__name__, *imp.items()))

            if len(z) >= 2:
                logger.info('Fused is missing:', z[0]-z[1])
                logger.info()
                logger.info('Fused contains:', z[1]-z[0])

            p.add_analysis(t, legend='original expression', s=300)
            p.show()

    pprint(transformer_results)
    input('Press Enter to continue...')

if __name__ == '__main__':
    run()
