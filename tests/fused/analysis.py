from collections import namedtuple, Mapping

AreaErrorTuple = namedtuple('AreaErrorTuple', ['area', 'error'])

def mins_of_analysis(analysis):
    mins = {}
    for key in ('area', 'error'):
        mins[key] = min(analysis, key=lambda d: d.get(key))

    return mins


def improvements(old, new):
    # 0.75 (75%) less
    area_change = 1.0 - float(new['area']['area']) / old['area']['area']
    error_change = 1.0 - float(new['error']['error']) / old['error']['error']
    # 4x less
    area_scaling = float(old['area']['area']) / new['area']['area']
    error_scaling = float(old['error']['error']) / new['error']['error']
    # the area cost of the best (lowest) error as a scaling
    cost_of_error = float(new['error']['area']) / old['error']['area']
    return {
        'change': AreaErrorTuple(area=area_change, error=error_change), 
        'scaling': AreaErrorTuple(area=area_scaling, error=error_scaling),
        'expression': AreaErrorTuple(area=new['area']['expression'], error=new['error']['expression']),
        'cost_of_error': cost_of_error, # scaling
    }


def run(timing=True):
    import time
    import gmpy2
    from pprint import pprint
    
    import soap.logger as logger
    from soap.common import invalidate_cache
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
    logger.set_context(level=logger.levels.debug)
    single_prec = gmpy2.ieee(32).precision - 1

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

    # fused unit = 3-input FP adder(s)
    actions = (
        (BiOpTreeTransformer, 'no fused'),
        (FusedBiOpTreeTransformer, 'any fused'),
        (FusedOnlyBiOpTreeTransformer, 'only fusing'),
        (Add3TreeTransformer, 'only add3 fusing'),
        (ConstMultTreeTransformer, 'only constMult fusing'),
    )[:2]

    transformer_results = []
    
    #benchmarks = {k: benchmarks[k] for k in ['2mm_1']}
    for benchmark_name in benchmarks:
        logger.error('Running', benchmark_name)
        bench = benchmarks[benchmark_name]
        e, v = bench.expr_and_vars()
        t = Expr(e)
        logger.info('Expr:', str(t))
        logger.info('Tree:', t.tree())

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
                    #s = Transformer(t, depth=None).closure()
                    s = trace_[0](t, v, depth=trace_[1], transformer=Transformer)
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
                    p.add(frontier_,
                        legend=label, time=duration, annotate=True, annotate_kwargs={'fontsize': 10})
                    z.append(set(map(
                        lambda d:(d['area'], d['error'], d['expression']),
                        frontier)))
                    
                    # Analyse the frontier for improvements
                    if transformer_index == 0:
                        original_mins = mins_of_analysis(frontier)
                    else:
                        imp_dict = improvements(original_mins, mins_of_analysis(frontier))
                        transformer_results.append([Transformer.__name__, imp_dict])

                if len(z) >= 2:
                    logger.info('Fused is missing:', z[0]-z[1])
                    logger.info('Fused contains:', z[1]-z[0])

                p.add_analysis(t, legend='original expression', s=300)
                p.show()
                # end for transformer_index, action_tuple
            # end for action
        # end for trace_
    # end for bench

    logger.debug(transformer_results)

    if transformer_results:
        best_area_improvement = max(transformer_results, key=lambda p:p[1]['scaling'].area)
        best_error_improvement = max(transformer_results, key=lambda p:p[1]['scaling'].error)
        worst_area_cost_of_error = max(transformer_results, key=lambda p:p[1]['cost_of_error'])
        
        pprint(best_area_improvement)
        pprint(best_error_improvement)
        pprint(worst_area_cost_of_error)
        print()
        print('best area improvement:', best_area_improvement[1]['scaling'].area)
        print('best error improvement:', best_error_improvement[1]['scaling'].error)
        print('worst area cost of error:', worst_area_cost_of_error[1]['cost_of_error'])
    else:
        logger.error('No transformer comparison made.')

    input('\nPress Enter to continue...')

if __name__ == '__main__':
    run()
