# Fused Arithmetic Analysis for Efficient Hardware Datapath [MEng Final-Year Project]

Eusebius M. Ngemera, Department of Electrical & Electronic Engineering, Imperial College London.

## Introduction

This is a fork of SOAP version 1 that considers the floating-point fused arithmetic units: 3-input adder, constant multiplier and fused multiply-add.

Version 1 of SOAP takes in a numerical expression (additions and multiplications of variables and constants), value ranges of the input variables and returns a set of rewritten equivalent expressions such that when synthesised onto an FPGA, the area in number of LUTs (look-up tables) and numerical accuracy in the form maximum absolute error are both minimised.

Forked version 1 from <https://github.com/admk/soap> at commit `b1bd173bb47f3ca8afbb0e0bb0b440f88bcf69a5`.

More details including my report are [here](http://eugenius1.github.io/fyp).

## Install

Instructions are given for Ubuntu and are expected to work for other major Linux OS's.

- Install [matplotlib](http://matplotlib.org/users/installing.html#build-requirements).

- Install [Python3](https://www.python.org/downloads/), but this pretty much always already installed.

- Install dependencies:
```bash
pip3 install -r requirements.txt
```

- Install gmpy2 outside of pip:
```bash
sudo apt-get install python3-gmpy2
```

### Optional

Optionally, in order to allow fetching of area information beyond that already stored (the included cache is sufficient for the default benchmark parameters):

- Install [FloPoCo 2.5.0](http://flopoco.gforge.inria.fr/flopoco_installation.html)
- Install [ISE Design Suite](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/design-tools.html) (version 14.7; needs a license file)

Either add the locations of these binaries to your `$PATH`, or make a symbolic link and add it to an existing `$PATH` location like `/usr/bin/`:

```
/path/to/your/flopoco-2.5.0/flopoco 
/path/to/your/Xilinx/14.7/ISE_DS/ISE/bin/lin64/xst 
```

## Usage

While in the project directory, run the following command to run the default parameters and see graphs.

```bash
PYTHONPATH=. python3 tests/fused/analysis.py 
```

### Parameters

The function call `run()` at the end of `tests/fused/analysis.py` can take keyword arguments, as shown below.

Defaults are to run the benchmark expressions available (a subset from PolyBench and Livermore Loops) at single-precision.
Area dynamic cache is used by default and full closure is performed with a maximum transformation depth of 100.
The multiple-use FMA type is default and singular frontiers are expanded when plotted.

```python
logging='warning',
# ('o'|'off') | ('e'|'error') | ('w'|'warning') | ('i'|'info') | ('v'|'d'|'debug')
benchmarks='suites',
# comma-separated list of the names or ('a'|'all') | ('s'|'suite'|'suites')
precision='single',
# wF integer or ('h'|'half') | ('s'|'single') | ('d'|'double') | ('q'|'quad'|'quadruple)
algorithm='closure',
# ('f'|'frontier') | ('gf'|'fg'|'greedy_frontier') | ('g'|'greedy') | ('c'|'closure')

use_area_cache=True, # area_dynamic.pkl
timing=True, # invalidate internal cache or not (not including the area caches)
alert_finish=False, # Ubuntu only

# Multiple precisions
vary_precision=False,
vary_precision_one_frontier=True, # show one frontier
precision_step=1, precision_start=22, precision_end=53,
# range(precision_start, precision_end + 1, precision_step)

# Fused Multiply-Add (FMA)
# fma_wf_factor overrides LSB_acc (LSBA) set by single_use_fma
fma_wf_factor=None,
# LSB_acc = MSB_acc - int(fma_wf_factor * wf) - 1
single_use_fma=False, # 
# True: LSB_acc = max(a_mul_b_exp_bounds.min, c_exp_bounds.min) - wF -1
# False: LSB_acc = min(a_mul_b_exp_bounds.min, c_exp_bounds.min) - wF

# Transformation depth
vary_transformation_depth=False, # 1 to 6
transformation_depth=100,

# Plotting
annotate=False,
annotate_size=14,
expand_singular_frontiers=True,
expand_all_frontiers=False,

compare_with_soap3=False, # only for `seidel` at single precision
```
