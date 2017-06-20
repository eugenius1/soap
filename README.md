# Fused Arithmetic Analysis for Efficient Hardware Datapath [MEng Final-Year Project]

Eusebius M. Ngemera, Department of Electrical & Electronic Engineering, Imperial College London.

## Introduction

This is a fork of SOAP version 1 that considers the floating-point fused arithmetic units: 3-input adder, constant multiplier and fused multiply-add.

Version 1 of SOAP takes in a numerical expression (additions and multiplications of variables and constants), value ranges of the input variables and returns a set of rewritten equivalent expressions such that when synthesised onto an FPGA, the area in number of LUTs (look-up tables) and numerical accuracy in the form maximum absolute error are both minimised.

Forked version 1 from <https://github.com/admk/soap> at commit `b1bd173bb47f3ca8afbb0e0bb0b440f88bcf69a5`.

More details including my report are [here](http://eugenius1.github.io/fyp).

## Install

Instructions are given for Ubuntu and are expected to work for other major Linux OS's.

- Install [Matplotlib](http://matplotlib.org/users/installing.html#build-requirements).

- Install Python3, but this usually always already installed.

- Install dependencies:
```bash
pip3 install -r requirements.txt
```

- Install gmpy3 outside of pip:
```bash
sudo apt-get install python3-gmpy2
```

### Optional

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

Defaults are to run the benchmark expressions available (a subset from PolyBench and Livermore Loops) at single-precision, and varying the precision of the frontiers.
Area dynamic cache is used by default and full closure is perfomed with a maximum transormation depth of 1000.
