from collections import namedtuple, Mapping

AreaErrorTuple = namedtuple('AreaErrorTuple', ['area', 'error'])

def mins_of_analysis(analysis):
    mins = {}
    for key in ('area', 'error'):
        mins[key] = min(analysis, key=lambda d: d.get(key))

    return mins


def improvements(old, new, old_duration=None, new_duration=None):
    # 0.75 (75%) less
    area_change = 1.0 - float(new['area']['area']) / old['area']['area']
    error_change = 1.0 - float(new['error']['error']) / old['error']['error']
    # 4x less
    area_scaling = float(old['area']['area']) / new['area']['area']
    error_scaling = float(old['error']['error']) / new['error']['error']
    # the area cost of the best (lowest) error as a scaling
    cost_of_error = float(new['error']['area']) / old['error']['area']
    duration_increase = {}
    if old_duration and new_duration:
        duration_increase = {
            'scaling': float(new_duration) / old_duration,
            'change': float(new_duration) / old_duration - 1.0,
        }
    return {
        'change': AreaErrorTuple(area=area_change, error=error_change), 
        'scaling': AreaErrorTuple(area=area_scaling, error=error_scaling),
        'expression': AreaErrorTuple(area=new['area']['expression'], error=new['error']['expression']),
        'cost_of_error': cost_of_error, # scaling
        'duration' : duration_increase,
    }


def is_better_frontier_than(first, second):
    from soap.analysis.core import pareto_frontier_2d
    first, second = set(first), set(second)
    best = set(pareto_frontier_2d(list(first.union(second))))
    better_first = best - second
    better_second = best - first
    return (not better_second), better_first, better_second


def run(timing=True, vary_precision=True, use_area_cache=True, precision_delta=2, annotate=True,
        transformation_depth=100, benchmarks='basics'):
    benchmark_names = benchmarks

    import time
    import gmpy2
    from pprint import pprint
    
    import soap.logger as logger
    from soap.common import invalidate_cache
    from soap.analysis import Plot
    from soap.analysis.utils import plot, analyse, analyse_and_frontier
    from soap.expr import Expr
    import soap.semantics.flopoco as flopoco
    from soap.semantics.flopoco import wf_range
    from soap.transformer.utils import (
        greedy_frontier_closure, greedy_trace, frontier_trace, martel_trace
    )
    from soap.transformer.biop import (
        BiOpTreeTransformer, FusedBiOpTreeTransformer, FusedOnlyBiOpTreeTransformer,
        Add3TreeTransformer, ConstMultTreeTransformer, FMATreeTransformer,
    )

    from tests.benchmarks import number_in_benchmark_suites, get_by_name as get_benchmarks
    from tests.fused.analysis import improvements, mins_of_analysis
    
    Expr.__repr__ = Expr.__str__
    logger.set_context(level=logger.levels.error)
    # wf excludes the leading 1 in the mantissa/significand
    half_prec = 10
    single_prec = gmpy2.ieee(32).precision - 1 # 23
    double_prec = gmpy2.ieee(64).precision - 1 # 52
    quad_prec = gmpy2.ieee(128).precision - 1 # 112
    standard_precs = [half_prec, single_prec, double_prec, quad_prec]
    precision = single_prec

    flopoco.use_area_dynamic_cache = use_area_cache

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
        (FMATreeTransformer, 'only fma fusing'),
    )[:2]

    traces = (
        (frontier_trace, transformation_depth),
        (greedy_frontier_closure, None),
        (greedy_trace, None),
    )[2:]

    transformer_results = []
    
    benchmarks = get_benchmarks(benchmark_names)
    for benchmark_name in benchmarks:
        logger.error('Running', benchmark_name)
        bench = benchmarks[benchmark_name]
        e, v = bench.expr_and_vars()
        t = Expr(e)
        logger.info('Expr:', str(t))
        logger.info('Tree:', t.tree())

        for trace_ in traces:
            
            for action in (actions, actions[::-1])[:1]: # forwards or backwards
                z = []
                frontier = []
                title = e.replace('\n', '').replace('  ', '').strip()
                if benchmark_name and benchmark_name[0] != '_':
                    # eg. '\texttt{2mm\_2}: d + (t * c)'
                    title = '\\texttt{{{name}}}: {expr}'.format(
                        name=benchmark_name.replace('_', '\_'), expr=title)
                p = Plot(var_env=v, blocking=False, title=title)#,legend_pos='top right')
                
                for transformer_index, action_tuple in enumerate(action):
                    Transformer, label = action_tuple
                    if timing:
                        invalidate_cache()
                    duration = time.time()
                    s = Transformer(t, depth=transformation_depth).closure()
                    s = trace_[0](t, v, depth=trace_[1], transformer=Transformer)
                    unfiltered, frontier = analyse_and_frontier(s, v, prec=precision)
                    duration = time.time() - duration # where to put this?

                    logger.info('Transformed:', len(s))
                    logger.info('Reduced by', len(unfiltered)-len(frontier))
                    
                    if len(frontier) <= 1:
                        # plot non-frontier points too
                        frontier_to_plot = unfiltered
                    else:
                        frontier_to_plot = frontier
                    
                    logger.info(frontier_to_plot)
                    linestyle = '--' if transformer_index == 0 else '-'
                    # plot(frontier_to_plot, blocking=False)
                    p.add(frontier_to_plot,
                        legend=label, time=duration, annotate=annotate, linestyle=linestyle,
                        annotate_kwargs={'fontsize': 10})
                    z.append(set(map(
                        lambda d:(d['area'], d['error']),
                        frontier)))
                    
                    # Analyse the frontier for improvements
                    if transformer_index == 0:
                        original_frontier = frontier
                        original_mins = mins_of_analysis(frontier)
                        original_duration = duration
                    else:
                        imp_dict = improvements(original_mins, mins_of_analysis(frontier),
                            original_duration, duration)
                        transformer_results.append([Transformer.__name__, benchmark_name, imp_dict])

                if len(z) >= 2: # or if in loop then transformer_index == 1:
                    fused_success, missing_plots, fused_plots = is_better_frontier_than(
                        z[0], z[1])
                    if missing_plots:
                        logger.error('Fused is missing:', missing_plots)
                    print('Fused discovered:', fused_plots)

                if vary_precision:
                    p.add_analysis(t, legend='varying precision', linestyle=':',
                        precs=list(range(
                            precision-precision_delta,
                            precision+precision_delta+1)
                        )
                    )
                p.add_analysis(t, legend='original expression', s=300, precs=[precision])
                p.show()
                # end for transformer_index, action_tuple
            # end for action
        # end for trace_
    # end for bench

    logger.debug(transformer_results)

    if transformer_results:
        best_area_improvement = max(transformer_results, key=lambda p:p[2]['scaling'].area)
        best_error_improvement = max(transformer_results, key=lambda p:p[2]['scaling'].error)
        worst_area_cost_of_error = max(transformer_results, key=lambda p:p[2]['cost_of_error'])
        worst_duration_increase = max(transformer_results, key=lambda p:p[2]['duration']['scaling'])
        
        pprint(best_area_improvement)
        pprint(best_error_improvement)
        pprint(worst_area_cost_of_error)
        pprint(worst_duration_increase)
        print()
        print('best area improvement:{} (error improved by {})'.format(
            best_area_improvement[2]['scaling'].area, best_area_improvement[2]['scaling'].error))
        print('best error improvement: {} (area cost was {})'.format(
            best_error_improvement[2]['scaling'].error, best_error_improvement[2]['cost_of_error']))
        print('worst area cost of error: {} (error improved by {})'.format(
            worst_area_cost_of_error[2]['cost_of_error'], worst_area_cost_of_error[2]['scaling'].error))
        if timing:
            print('worst duration increase: {} (improvements were {})'.format(
                worst_duration_increase[2]['duration']['scaling'], worst_duration_increase[2]['scaling']))
    else:
        logger.error('No transformer comparison made.')

    print('\n', dict(precision=precision, timing=timing, number_of_benchmarks=len(benchmarks),
        transformation_depth=transformation_depth,
        traces=tuple(map(lambda t:t[0].__name__, traces))))

    if len(benchmarks) > number_in_benchmark_suites:
        print('Heads up! You are running more than just the standard benchmark suites.')

    input('\nPress Enter to continue...')

if __name__ == '__main__':
    run()
