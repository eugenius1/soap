import os
import shutil
import tempfile
import functools
from contextlib import contextmanager

from soap.common import cached, timeit
import soap.logger as logger
from soap.expr.common import (
    ADD_OP, MULTIPLY_OP, ADD3_OP, CONSTANT_MULTIPLY_OP, FMA_OP,
)
from soap.common import print_return


class FlopocoMissingImplementationError(Exception):
    """Unsynthesizable operator"""


we_min, we_max = 5, 15
wf_min, wf_max = 10, 112
we_range = list(range(we_min, we_max + 1))
wf_range = list(range(wf_min, wf_max + 1))

directory = 'soap/semantics/'
default_file = directory + 'area.pkl'
area_dynamic_file = directory + 'area_dynamic.pkl'
template_file = directory + 'template.vhdl'

device_name = 'Virtex6'
device_model = 'xc6vlx760'

flopoco_2_5_0 = '2.5.0'
flopoco_4_1 = '4.1' # currently fails in xst with most of the ops
flopoco_main_version = flopoco_2_5_0
flopoco_version = flopoco_2_5_0
flopoco_generate_figures = False

use_area_dynamic_cache = True
fma_includes_conversion_to_fp = False

# SOAP-internal representations of flopoco operators
# they don't have to match the flopoco names thanks to the mapping below
F_DotProduct = 'DotProduct'
F_LongAcc = 'LongAcc'
F_LongAcc2FP = 'LongAcc2FP'
F_FPConstMult = 'FPConstMult'
F_FPConstMultRational = 'FPConstMultRational' ##### ? in 2.5.0 ?

flopoco_op_mapping = {
    flopoco_2_5_0: {
        ADD_OP :'FPAdder',
        MULTIPLY_OP: 'FPMultiplier',
        ADD3_OP: 'FPAdder3Input',
        CONSTANT_MULTIPLY_OP: 'FPConstMult',
        F_DotProduct: 'DotProduct',
        F_LongAcc: 'LongAcc',
        F_LongAcc2FP: 'LongAcc2FP',
    },
    flopoco_4_1: {
        ADD_OP: 'FPAdd',
        MULTIPLY_OP: 'FPMult',
        ADD3_OP: 'FPAdd3Input',
        CONSTANT_MULTIPLY_OP: 'FPConstMult',
        F_DotProduct: 'FPDotProduct',
        F_LongAcc: 'FPLargeAcc', # currently broken
        F_LongAcc2FP: 'LargeAccToFP',
    }
}

flopoco_args_mapping = {
    flopoco_2_5_0: {
        # wE wF
        'FPAdder': ('{we}', '{wf}'),
        # wE wF_in wF_out
        'FPMultiplier': ('{we}', '{wf}', '{wf}'),
        'FPAdder3Input': ('{we}', '{wf}'),
        # wE_in wF_in wE_out wF_out wC constant_expr
        'FPConstMult': ('{we}', '{wf}', '{we}', '{wf}', '{wc}', '{constant}'),
        # wE wFX wFY MaxMSB_in LSB_acc MSB_acc DSPThreshold
        'DotProduct': ('{we}', '{wf}', '{wf}', '{MaxMSB_in}', '{LSB_acc}', '{MSB_acc}', '{DSPThreshold}'),
        # wE_in wF_in MaxMSB_in LSB_acc MSB_acc
        'LongAcc': ('{we}', '{wf}', '{MaxMSB_in}', '{LSB_acc}', '{MSB_acc}'),
        # LSB_acc MSB_acc wE_out wF_out
        'LongAcc2FP': ('{LSB_acc}', '{MSB_acc}', '{we}', '{wf}'),
    },
    flopoco_4_1: {
        # wE wF [sub dualPath]
        'FPAdd': ('wE={we}', 'wF={wf}'),
        # wE wF [wFout]
        'FPMult': ('wE={we}', 'wF={wf}', 'wFout={wf}'),
        'FPAdd3Input': ('wE={we}', 'wF={wf}'),
        # wE_in wF_in wE_out wF_out constant [cst_width]
        'FPConstMult': (
            'wE_in={we}', 'wF_in={wf}', 'wE_out={we}', 'wF_out={wf}', 'constant={constant}', 'cst_width={wc}'),
        'FPDotProduct': (
            'wE={we}', 'wFX={wf}', 'wFY={wf}', 'MaxMSBX={MaxMSB_in}', 'MSBA={MSB_acc}', 'LSBA={LSB_acc}'),
        'FPLargeAcc': (
            'wEX={we}', 'wFX={wf}', 'MaxMSBX={MaxMSB_in}', 'MSBA={MaxMSB_acc}', 'LSBA={LSB_acc}'),
        'LargeAccToFP': (
            'wE_out={we}', 'wF_out={wf}', 'MSBA={MSB_acc}', 'LSBA={LSB_acc}'),
    }
}


def sh_flopoco(*args, version=None, **kwargs):
    import sh
    version = version or flopoco_version
    if version == flopoco_4_1:
        return sh.flopoco_4_1(*args, **kwargs)
    else:
        return sh.flopoco(*args, **kwargs)


@contextmanager
def cd(d):
    import sh
    p = os.path.abspath(os.curdir)
    if d:
        sh.mkdir('-p', d)
        sh.cd(d)
    try:
        yield
    except Exception:
        raise
    finally:
        sh.cd(p)


def get_luts(file_name):
    from bs4 import BeautifulSoup
    with open(file_name, 'r') as f:
        f = BeautifulSoup(f.read(), "lxml")
        app = f.document.application
        util = app.find('section', stringid='XST_DEVICE_UTILIZATION_SUMMARY')
        luts = util.find('item', stringid='XST_NUMBER_OF_SLICE_LUTS')
        return int(luts.get('value'))


def flopoco_command_args(fop, **kwargs):
    """Returns a list of arguments.
    """
    if fop not in flopoco_op_mapping[flopoco_version].values():
        raise ValueError('Unrecognised op {!r} for flopoco {}'.format(fop, flopoco_version))

    kwargs.setdefault('DSPThreshold', 0.9)
    if fop == flopoco_op_mapping[flopoco_version][CONSTANT_MULTIPLY_OP]:
        kwargs.setdefault('wc', 0)

        # alternatively, use rational constant multiplier
        # from soap.semantics.common import mpq
        # rational = mpq(constant)
        # # FPConstMultRational wE_in wF_in wE_out wF_out a b
        # flopoco_cmd += ['FPConstMultRational', we, wf, we, wf,
        #     str(rational.numerator), str(rational.denominator)]
    
    return [fop, *map(lambda a:a.format(**kwargs), flopoco_args_mapping[flopoco_version][fop])]
    

def flopoco(op, we=None, wf=None, f=None, dir=None, op_params={}, op_args=None):
    import sh
    # copy we and wf to the objects if given inside of op_params
    if we == None and  'we' in op_params:
        we = op_params['we']
    if wf == None and  'wf' in op_params:
        wf = op_params['wf']

    option_prefix = '-' if flopoco_version == flopoco_2_5_0 else ''
    flopoco_cmd = []
    flopoco_cmd += [option_prefix + 'target=' + device_name]
    dir = dir or tempfile.mkdtemp(prefix='soap_', suffix='/')
    logger.debug(dir)
    with cd(dir):
        if f is None:
            _, f = tempfile.mkstemp(suffix='.vhdl', dir=dir)
            logger.debug(type(f), f)
        # `outputfile` and other option names are case-insensitive
        flopoco_cmd += [option_prefix + 'outputfile={filepath}'.format(filepath=f)]
        if flopoco_version == flopoco_4_1:
            # generateFigures=<0|1>:generate SVG graphics
            flopoco_cmd += ['generateFigures={}'.format(int(flopoco_generate_figures))]
            flopoco_cmd += ['plainVHDL=0']
        if op == 'add' or op == ADD_OP:
            flopoco_cmd += ['FPAdder', we, wf]
        elif op == 'mul' or op == MULTIPLY_OP:
            flopoco_cmd += ['FPMultiplier', we, wf, wf]
        elif op == ADD3_OP:
            flopoco_cmd += ['FPAdder3Input', we, wf]
        elif op == CONSTANT_MULTIPLY_OP:
            flopoco_cmd += flopoco_command_args(F_FPConstMult, **op_params)
        elif op in flopoco_op_mapping[flopoco_version].values():
            if op_args == None:
                raise ValueError('Expecting a Sequence of op_args for the op {}'.format(op))
            flopoco_cmd += [op, *op_args]
        else:
            raise ValueError('Unrecognised operator %s' % str(op))
        logger.debug('Flopoco', flopoco_cmd)
        logger.debug(sh_flopoco(*flopoco_cmd, _err_to_out=False))
        try:
            with open(f) as fh:
                if not fh.read():
                    raise IOError()
        except (IOError, FileNotFoundError):
            logger.error('Flopoco failed to generate file %s' % f)
            raise
    return dir, f


def xilinx(f, dir=None):
    import sh
    file_base = os.path.split(f)[1]
    file_base = os.path.splitext(file_base)[0]
    g = file_base + '.ngc'
    cmd = ['run', '-p', device_model]
    cmd += ['-ifn', f, '-ifmt', 'VHDL']
    cmd += ['-ofn', g, '-ofmt', 'NGC']
    dir = dir or tempfile.mkdtemp(prefix='soap_', suffix='/')
    with cd(dir):
        logger.debug('Xilinx', repr(cmd))
        sh.xst(sh.echo(*cmd), _out='out.log', _err='err.log')
        return get_luts(file_base + '.ngc_xst.xrpt')


def eval_operator(op, we=None, wf=None, f=None, dir=None, op_params={}, op_args=None):
    """TODO: what if op is not dynamic and running this as part of _para_synth in batch_synth"""
    if op_args == None:
        if op_params:
            flopoco_args = flopoco_command_args(op, **op_params)
            op = flopoco_args[0]
            op_args = flopoco_args[1:]
        else:
            logger.error('Warning: flopoco.eval_operator given empty op_args and empty op_params'.format(
                op_args, op_params))
    
    if use_area_dynamic_cache:
        if flopoco_version != flopoco_main_version:
            version = [flopoco_version]
        else:
            version = []
        cache_key = (*version, op, *op_args)

    # check if in dynamic cache
    if use_area_dynamic_cache and cache_key in area_dynamic_cache:
        luts = area_dynamic_cache[cache_key]
    else: # evaluate
        dir, f = flopoco(op, we, wf, f, dir, op_params=op_params, op_args=op_args)
        # add we and wf to op_params if given
        for string, obj in (('we', we), ('wf', wf)):
            if obj != None:
                op_params[string] = obj
        # if args are in just op_args and not in op_params
        if not op_params:
            # set the key as the index it takes in the flopoco command, with 0 being the op
            # append a letter to a string of the number
            op_params.update(
                map(
                    lambda t:'a{}'.format(t[0]),
                    enumerate(op_args, start=1)))

        luts = xilinx(f, dir)
        if use_area_dynamic_cache:
            # add to dynamic cache
            area_dynamic_cache[cache_key] = luts
            save(area_dynamic_file, area_dynamic_cache)

    return dict(op=op, value=luts, **op_params)


@timeit
def _para_synth(op_we_wf):
    import sh
    op, we, wf = op_we_wf
    try:
        item = eval_operator(op, we, wf, f=None, dir=None)
        logger.info('Processed', item)
        return item
    except sh.ErrorReturnCode:
        logger.error('Error processing %s, %d, %d' % op_we_wf)


_pool = None


def pool():
    global _pool
    if _pool is None:
        import multiprocessing
        _pool = multiprocessing.Pool()
    return _pool


@timeit
def batch_synth(we_range, wf_range):
    import itertools
    ops = ['add', 'mul', 'add3'][2:]
    args = itertools.product(ops, we_range, wf_range)
    return list(pool().imap_unordered(_para_synth, args))


def load(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        return pickle.loads(f.read())


def save(file_name, results, do_format=False):
    import pickle
    if do_format:
        results = [i for i in results if not i is None]
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)


def plot(results):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    vl = []
    for i in results:
        xv, yv, zv = int(i['we']), int(i['wf']), int(i['value'])
        if zv < 0:
            continue
        vl.append((xv, yv, zv))
    ax.scatter(*zip(*vl))
    plt.show()



_op_luts = {ADD_OP: {}, MULTIPLY_OP: {}, ADD3_OP: {}}
if os.path.isfile(default_file):
    for i in load(default_file):
        xv, yv, zv = int(i['we']), int(i['wf']), int(i['value'])
        if i['op'] == 'add':
            _op_luts[ADD_OP][xv, yv] = zv
        elif i['op'] == 'mul':
            _op_luts[MULTIPLY_OP][xv, yv] = zv
        elif i['op'] == 'add3':
            _op_luts[ADD3_OP][xv, yv] = zv

_add = _op_luts[ADD_OP]
_mul = _op_luts[MULTIPLY_OP]

# Area dynamic cache
if os.path.isfile(area_dynamic_file):
    area_dynamic_cache = load(area_dynamic_file)
else: # create file
    area_dynamic_cache = {}
    save(area_dynamic_file, area_dynamic_cache)


def _impl(_dict, we, wf):
    try:
        return _dict[we, wf]
    except KeyError:
        if not wf in wf_range:
            raise FlopocoMissingImplementationError(
                'Precision %d out of range' % wf)
        elif not we in we_range:
            raise FlopocoMissingImplementationError(
                'Exponent width %d out of range' % we)
        return _impl(_dict, we + 1, wf)


def adder(we, wf):
    return _impl(_add, we, wf)


def multiplier(we, wf):
    return _impl(_mul, we, wf)


@cached
@print_return('flopoco.')
def luts_for_op(op, we=None, wf=None, **kwargs):
    if flopoco_version == flopoco_main_version and op in _op_luts:
        return _impl(_op_luts[op], we, wf)

    kwargs.update(we=we, wf=wf)
    if op == FMA_OP:
        MaxMSB_in = 0
        LSB_acc = -wf-1
        MSB_acc = 1

        luts = eval_operator(flopoco_op_mapping[flopoco_version][F_DotProduct],
            op_params=dict(
                we=we, 
                wf=wf,
                MaxMSB_in=MaxMSB_in,
                LSB_acc=LSB_acc,
                MSB_acc=MSB_acc)
            ).get('value')
        if fma_includes_conversion_to_fp:
            luts += eval_operator(flopoco_op_mapping[flopoco_version][F_LongAcc2FP],
                op_params=dict(
                    LSB_acc=LSB_acc, 
                    MSB_acc=MSB_acc,
                    we=we,
                    wf=wf)
                ).get('value')
    # ops that only run one flopoco command
    elif op in (ADD_OP, MULTIPLY_OP, ADD3_OP, CONSTANT_MULTIPLY_OP):
        return eval_operator(flopoco_op_mapping[flopoco_version][op],
            op_params=kwargs
            ).get('value')
    else:
        raise ValueError('Area info for {!r} not found'.format(op))
    return luts


@cached
def keys():
    return sorted(list(set(_add.keys()) & set(_mul.keys())))


class CodeGenerator(object):

    def __init__(self, expr, var_env, prec, file_name=None, dir=None):
        from soap.expr import Expr
        self.expr = Expr(expr)
        self.var_env = var_env
        self.wf = prec
        self.we = self.expr.exponent_width(var_env, prec)
        self.dir = dir or tempfile.mkdtemp(prefix='soap_', suffix='/')
        with cd(self.dir):
            self.f = file_name or tempfile.mkstemp(suffix='.vhdl', dir=dir)[1]

    def generate(self):
        from akpytemp import Template

        ops = set()
        in_ports = set()
        out_port, ls = self.expr.as_labels()
        wires = set()
        signals = set()

        def wire(op, in1, in2, out):
            def wire_name(i):
                if i in signals:
                    return i.signal_name()
                if i in in_ports:
                    return i.port_name()
                if i == out_port:
                    return 'p_out'
            for i in [in1, in2, out]:
                # a variable represented as a string is a port
                if isinstance(i.e, str):
                    in_ports.add(i)
                    continue
                # a number is a port
                try:
                    float(i.e)
                    in_ports.add(i)
                    continue
                except (TypeError, ValueError):
                    pass
                # a range is a port
                try:
                    a, b = i.e
                    float(a), float(b)
                    in_ports.add(i)
                    continue
                except (TypeError, ValueError):
                    pass
                # an expression, need a signal for its output
                try:
                    i.e.op
                    if i != out_port:
                        signals.add(i)
                except AttributeError:
                    pass
            wires.add((op, wire_name(in1), wire_name(in2), wire_name(out)))

        for out, e in ls.items():
            try:
                op, in1, in2 = e.op, e.a1, e.a2
                wire(op, in1, in2, out)
                ops.add(e.op)
            except AttributeError:
                pass
        in_ports = [i.port_name() for i in in_ports]
        out_port = 'p_out'
        signals = [i.signal_name() for i in signals]
        logger.debug(in_ports, signals, wires)
        Template(path=template_file).save(
            path=self.f, directory=self.dir, flopoco=flopoco,
            ops=ops, e=self.expr,
            we=self.we, wf=self.wf,
            in_ports=in_ports, out_port=out_port,
            signals=signals, wires=wires)
        return self.f


def eval_expr(expr, var_env, prec):
    import sh
    dir = tempfile.mkdtemp(prefix='soap_', suffix='/')
    f = CodeGenerator(expr, var_env, prec, dir=dir).generate()
    logger.debug('Synthesising', str(expr), 'with precision', prec, 'in', f)
    try:
        return xilinx(f, dir=dir)
    except (sh.ErrorReturnCode, KeyboardInterrupt):
        raise
    finally:
        shutil.rmtree(dir)


if __name__ == '__main__':
    import sys
    from soap.expr import Expr
    logger.set_context(level=logger.levels.debug)
    if 'synth' in sys.argv:
        save(directory + 'area.add3.pkl', batch_synth(we_range, wf_range), do_format=True)
    else:
        p = 23
        e = Expr('a + b + c')
        v = {'a': ['0', '1'], 'b': ['0', '100'], 'c': ['0', '100000']}
        logger.info(e.area(v, p).area)
        logger.info(e.real_area(v, p))
        plot(load(directory + 'area.add3.pkl'))
