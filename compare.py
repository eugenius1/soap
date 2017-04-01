from flopoco import load
import soap.logger as logger
from matplotlib import rc

rc('font', family='serif')
rc('text', usetex=True)

def plot(results, 
    transformed=False, title='', multiple=False, labels=None, 
    legend_loc='center left'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    
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
                    continue
                vl.append((xv, yv, zv))
        
        label = index
        if labels:
            try:
                label = labels[index]
            except IndexError:
                pass
        
        # transpose list of 3-element tuples to three lists, of xv, yv and zv    
        ax.scatter(*zip(*vl), label=label, marker='.')
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Exponent size (bits)')
    ax.set_ylabel('Mantissa size (bits)')
    ax.set_zlabel('Area (Number of LUTs)')
    plt.legend(loc=legend_loc);
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