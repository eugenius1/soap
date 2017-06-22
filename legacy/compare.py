"""
Plot "Area Usage (LUT Count) for 3-operand FP Addition: Fused vs Discrete"
Requires:
    - area.pkl
    - area.add3.pkl
"""

from soap.semantics.flopoco import load
import soap.logger as logger
from matplotlib import rc

rc('font', family='serif')
rc('text', usetex=True)

def plot(results, 
    transformed=False, title='', multiple=False, labels=None, 
    legend_loc='center left',
    ylabel='', zlabel='Area (Number of LUTs)',
    dim=3, do_on_plt=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    we_label = 'Exponent size (bits)'
    wf_label = 'Mantissa size (bits)'
    fig = plt.figure(id(results))
    if dim == 3:
        ax = Axes3D(fig)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(we_label)
        ax.set_ylabel(wf_label)
        ax.set_zlabel(zlabel)
        graph = ax
    else:
        plt.title(title, fontweight='bold')
        plt.xlabel(wf_label)
        plt.ylabel(ylabel)
        plt.grid(True)
        graph = plt
    if not multiple:
        results = [results]
    
    for index, collection in enumerate(results):
        if transformed:
            vl = collection
        else:
            vl = []
            for item in collection:
                xv, yv, zv = int(item['we']), int(item['wf']), int(item['value'])
                if zv < 0:
                    logger.warning('Warning: At ({x},{y},{z}): zv < 0.'.format(x=xv,y=yv,z=zv))
                    continue
                vl.append((xv, yv, zv))
        label = index
        if labels:
            try:
                label = labels[index]
            except IndexError:
                pass
        if dim == 3:
            # eg. transpose list of 3-element tuples to three lists, of xv, yv and zv
            ax.scatter(*zip(*vl), label=label, marker='.')
        else:
            x, y = list(zip(*vl))
            logger.info(len(x), len(y))
            p = plt.plot(x, y, 'o', linewidth=4.0, label=label)
            x, y = tuple(map(np.array, (x,y)))
            logger.info(type(x), type(y))
            # polynomial fit
            # grad, yintersect = np.polyfit(x, y, 1)
            # plt.plot(x, grad*x + yintersect, '--')
            # exponential fit
            def func(x, a, b, c):
                return a * np.exp(-b * (x-10)) + c
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(func, x, y)
            logger.info('popt={popt}'.format(popt=popt))
            xx = np.linspace(10,20)
            plt.plot(xx, func(xx, *popt), ':', color=p[-1].get_color())
    
    if not vl:
        logger.error('Nothing to plot. vl={}'.format(vl))
    logger.info('vl: len={}, [0]={}'.format(len(vl), vl[0]))
    plt.legend(loc=legend_loc)
    if do_on_plt:
        do_on_plt(plt)
    plt.ion() 
    plt.show()

logger.set_context(level=logger.levels.debug)
directory = 'soap/semantics/'

# for file in ('area.pkl', 'area.pkl.backup', 'area.add3.pkl'):
#     results = load(directory + file)
#     logger.info(file, len(results))
#     plot(results, title=file)

def get_results_dict(results, op):
    dictionary = {}
    for r in results:
        if r['op'] == op:
            dictionary[ (int(r['we']), int(r['wf'])) ] = int(r['value'])

    return dictionary

def plot_lut_comparison():
    add_luts = get_results_dict(load(directory + 'area.pkl'), 'add')
    add3_luts = get_results_dict(load(directory + 'area.add3.pkl'), 'add3')

    logger.info(len(add_luts)) # 997
    logger.info(len(add3_luts)) # 906

    # data aggregate
    agg = {'add3':[], '2add2':[], 'diff': []}
    for key in add3_luts:
        if key in add_luts:
            agg['add3'].append((key[0], key[1], add3_luts[key]))
            agg['2add2'].append((key[0], key[1], 2*add_luts[key]))
            agg['diff'].append((key[0], key[1], (add3_luts[key]-2*add_luts[key])))

    # plot(agg['diff'], transformed=True, title='LUT usage of a 3-input FP Adder compared to two equivalent 2-input')
    plot([agg['add3'], agg['2add2']], transformed=True,
        labels=['One 3-input FP Adder','Two 2-input FP Adders'], 
        title='Area Usage (LUT Count) for 3-operand FP Addition: Fused vs Discrete', 
        multiple=True)

def plot_error_comparison(v=None):
    import copy
    from soap.expr import ADD3_OP, Expr
    from soap.analysis.core import VaryWidthAnalysis
    from soap.semantics.flopoco import wf_range
    logger.set_context(level=logger.levels.info)
    if v == None:
        v = {
            'a': ['5', '10'],
            'b': ['0', '0.001'],
            'c': ['0', '0.000001'],
        }
    e = Expr('((a + b) + c)')
    e2 = copy.deepcopy(e)
    e2.op = ADD3_OP
    e2.operands = ('a','b','c')
    logger.warning(e)
    logger.warning(e2)
    agg = {'add3':[], '2add2':[], 'diff': []}
    for name, expr in zip(['2add2', 'add3'], [e, e2]):
        a = VaryWidthAnalysis(set([expr]), v)
        a, f = a.analyse(), a.frontier()
        logger.info('Results', len(a))
        logger.info(a[0])
        logger.info('Frontier', len(f))
        logger.info(f[0])
        if len(wf_range) != len(a):
            logger.error('len(wf_range) != len(a): {} != {}'.format(len(wf_range), len(a)))
        for index, d in enumerate(a):
            agg[name].append((wf_range[index], d['error']))

    do_loglog = True
    def customise(plt):
        plt.xlim([wf_range[0], 20])
        if do_loglog:
            # plt.xlim([20, 30])
            # max is error value of 2add2 at smallest wf
            m = agg['2add2'][0][1]
            plt.ylim([0.001*m, m])
            # plt.ylim([1e-8, 2e-5]) # wf = [20, 30]
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(loc='top right')

    plot([agg['2add2'], agg['add3']], transformed=True,
        labels=['Two 2-input FP Adders','One 3-input FP Adder'],
        ylabel='Absolute Error',
        title='Error for a 3-operand FP Addition: Fused vs Discrete',
        legend_loc='center right',
        multiple=True, dim=2, do_on_plt=customise)

# plot_lut_comparison()
a = ['1000000000', '1000000000000'] # e9 -> e12
c = ['0.000000000001', '0.000000001'] # e-12 -> e-9
plot_error_comparison()
# plot_error_comparison(v={'a': a, 'b': a, 'c': a})
# plot_error_comparison(v={'a': c, 'b': c, 'c': c})

input('Press Enter to continue...')