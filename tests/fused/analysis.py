from collections import namedtuple, Mapping

AreaErrorTuple = namedtuple('AreaErrorTuple', ['area', 'error'])

def tex_sanitise(string):
    return string.replace('_', '\_')


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


def run(# Default parameters. Override in the call at the end of the file
        logging='warning',
        benchmarks='suites',
        precision='single',
        algorithm='closure',

        use_area_cache=True, # area_dynamic.pkl
        timing=True, # invalidate internal cache or not (not including the area caches)
        alert_finish=False, # Ubuntu only
        
        # Multiple precisions
        vary_precision=False,
        vary_precision_one_frontier=True, # show one frontier
        precision_step=1, precision_start=22, precision_end=53,

        # FMA
        fma_wf_factor=None,
        single_use_fma=False,

        # Transformation depth
        vary_transformation_depth=False,
        transformation_depth=100,
        
        # Plotting
        annotate=False,
        annotate_size=14,
        expand_singular_frontiers=True,
        expand_all_frontiers=False,

        compare_with_soap3=False, # only for seidel at single precision
    ):
    benchmark_names = benchmarks

    import time
    import gmpy2
    from pprint import pprint
    
    import soap.logger as logger
    import soap.common
    from soap.common import invalidate_cache
    from soap.analysis import Plot
    from soap.analysis.utils import plot, analyse, analyse_and_frontier
    from soap.expr import Expr
    import soap.semantics.flopoco as flopoco
    from soap.semantics.flopoco import wf_range
    from soap.transformer.utils import (
        greedy_frontier_closure, greedy_trace, frontier_trace, closure
    )
    from soap.transformer.biop import (
        BiOpTreeTransformer, FusedBiOpTreeTransformer, FusedOnlyBiOpTreeTransformer,
        Add3TreeTransformer, ConstMultTreeTransformer, FMATreeTransformer,
    )

    from tests.benchmarks import (
        soap3_results, number_in_benchmark_suites, get_by_name as get_benchmarks
    )
    
    # Set logging level
    logging = {'o': 'off', 'e': 'error', 'w': 'warning', 'i': 'info', 'd': 'debug', 'v': 'debug'
        }.get(logging, logging)
    logger.set_context(level=getattr(logger.levels, logging, logger.levels.warning))
    
    Expr.__repr__ = Expr.__str__

    # Set SOAP parameters
    soap.common.fma_is_single_use = single_use_fma
    flopoco.use_area_dynamic_cache = use_area_cache
    flopoco.fma_includes_conversion_to_fp = True
    flopoco.fma_wf_factor = fma_wf_factor

    # wf excludes the leading 1 in the mantissa/significand
    # TODO: replace with IEEE754Standards from soap.common
    standard_precs = {
        'half': 10,
        'single': gmpy2.ieee(32).precision - 1, # 23
        'double': gmpy2.ieee(64).precision - 1, # 52
        'quad': gmpy2.ieee(128).precision - 1, # 112
    }
    try:
        precision = int(precision)
    except (TypeError, ValueError):
        precision = {'h': 'half', 's': 'single', 'd': 'double', 'q': 'quad', 'quadruple': 'quad'
            }.get(precision, precision)
        precision = standard_precs.get(precision, standard_precs['single'])

    if vary_precision:
        precs = list(range(
            precision_start,
            precision_end + 1,
            precision_step)
        )
        timing = False

    if not vary_transformation_depth:
        actions = (
            (BiOpTreeTransformer, ('original frontier' if vary_precision else 'no fused')),
            (FusedBiOpTreeTransformer, ('fused frontier' if vary_precision else 'with fused')),
            (FusedOnlyBiOpTreeTransformer, 'only fusing'),
            (Add3TreeTransformer, 'only add3 fusing'),
            (ConstMultTreeTransformer, 'only constMult fusing'),
            (FMATreeTransformer, 'only fma fusing'),
        )[:2]
    else:
        actions = (
            (FusedBiOpTreeTransformer, 'fused frontier', 1),
            (FusedBiOpTreeTransformer, 'fused frontier', 2),
            (FusedBiOpTreeTransformer, 'fused frontier', 3),
            (FusedBiOpTreeTransformer, 'fused frontier', 4),
            (FusedBiOpTreeTransformer, 'fused frontier', 5),
            (FusedBiOpTreeTransformer, 'fused frontier', 6),
        )

    traces = {
        'frontier': frontier_trace,
        'greedy_frontier': greedy_frontier_closure,
        'greedy': greedy_trace,
        'closure': closure,
    }
    if algorithm not in ('a', 'all'):
        if algorithm not in traces.keys():
            algorithm = {'f': 'frontier', 'gf': 'greedy_frontier', 'fg': 'greedy_frontier', 'g': 'greedy', 'c': 'closure',
                }.get(algorithm, 'closure')
        traces = {algorithm: traces[algorithm]}

    line_styles = ['dashed', 'dashdot', 'dotted', 'solid']

    transformer_results = []
    fused_failures = []
    
    benchmarks = get_benchmarks(benchmark_names, warning_print=logger.error)

    for benchmark_name in benchmarks:
        print('Running ', end='')
        logger.error(benchmark_name)
        bench = benchmarks[benchmark_name]
        e, v = bench.expr_and_vars()
        if bench.max_transformation_depth != None:
            current_depth_limit = min(transformation_depth, bench.max_transformation_depth)
            if current_depth_limit != transformation_depth:
                logger.warning('Capped the input transformation depth from {old} to {new} for {bench}'.format(
                    old=transformation_depth, new=current_depth_limit, bench=benchmark_name))
        else:
            current_depth_limit = transformation_depth

        t = Expr(e)
        logger.info('Expr:', str(t))
        logger.info('Tree:', t.tree())

        for algorithm in traces:
            trace_func = traces[algorithm]

            for action in (actions, actions[::-1])[:1]: # forwards or backwards
                z = []
                frontier = []

                # Format title
                # eg. '$d + (t \times c)$'
                title = e.replace('\n', '').replace('  ', '')
                original_title_length = len(title)
                title = title.replace('*', '\\times ').replace('Sigma', '\\Sigma').strip()
                title = '${}$'.format(title)
                if benchmark_name and benchmark_name[0] != '_':
                    # eg. '\texttt{2mm\_2}: ' + expr
                    title = '\\texttt{{{name}}}: {expr}'.format(
                        name=tex_sanitise(benchmark_name), expr=title)
                if len(traces) > 1:
                    title = '{{\\normalsize {t}}} (\\texttt{{{a}}})'.format(t=title, a=tex_sanitise(algorithm))
                elif original_title_length > 60:
                    title = '{{\\Large {}}}'.format(title)
                elif original_title_length > 40:
                    title = '{{\\LARGE {}}}'.format(title)
                elif original_title_length > 30:
                    title = '{{\\huge {}}}'.format(title)
                p = Plot(var_env=v, blocking=False, title=title)#,legend_pos='top right')
                
                p.add_analysis(t, legend='original expression', s=300, precs=[precision],
                    cycle_marker=False)

                plot_extra_kwargs = {}

                for transformer_index, action_tuple in enumerate(action):
                    if vary_transformation_depth:# and transformer_index > 0:
                        Transformer, label, current_depth_limit = action_tuple
                        logger.warning('Current depth:', current_depth_limit)
                    else:
                        Transformer, label = action_tuple
                    
                    if timing:
                        invalidate_cache()

                    duration = time.time()
                    s = trace_func(t, v, depth=current_depth_limit, prec=precision, transformer=Transformer)
                    unfiltered, frontier = analyse_and_frontier(s, v, prec=precision)
                    duration = time.time() - duration # where to put this?

                    logger.info('Transformed:', len(s))
                    logger.info('Reduced by', len(unfiltered)-len(frontier))
                    
                    if expand_all_frontiers or (expand_singular_frontiers and len(frontier) <= 1):
                        # plot non-frontier points too if there would otherwise only be a single point
                        frontier_to_plot = unfiltered
                    else:
                        frontier_to_plot = frontier
                    
                    logger.info(frontier_to_plot)
                    # don't include the time duration in plots if not timing
                    duration = None if not timing else duration
                    if vary_transformation_depth:
                        plot_extra_kwargs = {'depth': current_depth_limit}
                        linestyle = line_styles[transformer_index % len(line_styles)]
                    else:
                        plot_extra_kwargs.update(color_group=label)
                        linestyle = '--' if transformer_index == 0 else '-'

                    # Plot the frontier
                    p.add(frontier_to_plot,
                        legend=label, time=duration, annotate=annotate, linestyle=linestyle,
                        annotate_kwargs={'fontsize': annotate_size},
                        **plot_extra_kwargs,
                    )
                    z.append(set(map(
                        lambda d:(d['area'], d['error']),
                        frontier))
                    )
                    
                    # Analyse the frontier for improvements
                    if transformer_index == 0:
                        original_frontier = frontier
                        original_duration = duration
                        original_mins = mins_of_analysis(original_frontier)
                    else:
                        imp_dict = improvements(original_mins, mins_of_analysis(frontier),
                            original_duration, duration)
                        transformer_results.append([
                            Transformer.__name__, benchmark_name, imp_dict, algorithm])

                if len(z) >= 2:
                    fused_success, missing_plots, fused_plots = is_better_frontier_than(
                        z[0], z[1])
                    if missing_plots:
                        logger.error('Fused is missing:', missing_plots)
                        fused_failures.append(benchmark_name)
                    print('Fused discovered:', fused_plots)

                if vary_precision:
                    linestyle = ':'

                    for index_fr, frontier_tuple in enumerate([
                        (original_frontier, 'varying precision of original frontier', 80),
                        (frontier, 'varying precision of fused frontier', 30)
                    ]):
                        front, label, marker_size = frontier_tuple
                        frontier_expressions = list(map(lambda d: Expr(d['expression']), front))
                        legend_kwarg = {
                            'legend': label
                        }
                        analysis_kwargs = {
                            #'cycle_marker': (not bool(index_fr)),
                            's': marker_size,
                        }

                        if vary_precision_one_frontier:
                            _, results_vp = analyse_and_frontier(frontier_expressions, v, precs)
                            p.add(results_vp,
                                linestyle=linestyle,
                                color_group=label,
                                annotate=annotate,
                                **legend_kwarg,
                                **analysis_kwargs,
                            )
                            # Analyse the frontier for improvements
                            if index_fr == 0:
                                original_frontier = results_vp
                                original_mins = mins_of_analysis(original_frontier)
                            else:
                                imp_dict = improvements(original_mins, mins_of_analysis(frontier))
                                transformer_results.append([
                                    Transformer.__name__, benchmark_name, imp_dict, algorithm])
                        else:
                            for index_epf, expr in enumerate(frontier_expressions):
                                p.add_analysis(expr,
                                    linestyle=linestyle,
                                    color_group=label,
                                    precision_frontier=True,
                                    precs=precs,
                                    **legend_kwarg,
                                    **analysis_kwargs,
                                )
                                if index_epf == 0:
                                    legend_kwarg = {}
                
                if compare_with_soap3:
                    if benchmark_name in soap3_results:
                        results = soap3_results[benchmark_name]
                        p.add(results['analysis'],
                            legend='SOAP 3', time=results['analysis_duration'], annotate=annotate, linestyle='-.',
                            annotate_kwargs={'fontsize': annotate_size}
                        )
                    else:
                        logger.error('No SOAP3 results for {} found. Available are {}'.format(
                            benchmark_name, list(soap3_results.keys())))
                
                p.show()
                # end for transformer_index, action_tuple
            # end for action
        # end for algorithm in traces:
    #end for benchmark_name in benchmarks

    logger.warning(transformer_results)

    if transformer_results:
        # a visual reminder to not claim these results as from the benchmark suites
        if len(benchmarks) > number_in_benchmark_suites:
            print_func = logger.error
        else:
            print_func = print

        best_area_improvement = max(transformer_results, key=lambda p:p[2]['scaling'].area)
        best_error_improvement = max(transformer_results, key=lambda p:p[2]['scaling'].error)
        worst_area_cost_of_error = max(transformer_results, key=lambda p:p[2]['cost_of_error'])
        
        pprint(best_area_improvement)
        pprint(best_error_improvement)
        pprint(worst_area_cost_of_error)
        if timing:
            worst_duration_increase = max(transformer_results, key=lambda p:p[2]['duration']['scaling'])
            pprint(worst_duration_increase)

        print()
        print_func('best area improvement: {} (error improved by {})'.format(
            best_area_improvement[2]['scaling'].area, best_area_improvement[2]['scaling'].error))
        print_func('best error improvement: {} (area cost was {})'.format(
            best_error_improvement[2]['scaling'].error, best_error_improvement[2]['cost_of_error']))
        print_func('worst area cost of error: {} (error improved by {})'.format(
            worst_area_cost_of_error[2]['cost_of_error'], worst_area_cost_of_error[2]['scaling'].error))
        if timing:
            print_func('worst duration increase: {} (improvements were {})'.format(
                worst_duration_increase[2]['duration']['scaling'], worst_duration_increase[2]['scaling']))
    else:
        logger.error('No transformer comparison made.')

    print()
    pprint(dict(precision=precision, timing=timing, number_of_benchmarks=len(benchmarks),
        transformation_depth=transformation_depth, use_area_cache=use_area_cache,
        algorithm=list(traces.keys()),
        fma_wf_factor=fma_wf_factor, single_use_fma=single_use_fma,
    ))

    if fused_failures:
        logger.error('Missing points in fused frontier of', fused_failures)

    if alert_finish:# and len(benchmarks) >= number_in_benchmark_suites:
        import subprocess
        subprocess.call(['speech-dispatcher'])        #start speech dispatcher
        subprocess.call(['spd-say', '"All done!"'])

    input('\nPress Enter to continue...')


if __name__ == '__main__':
    run()
