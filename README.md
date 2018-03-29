# Hungarian Algorithm C++ Library

**DON'T USE, CURRENTLY BROKEN**

The Hungarian algorithm, also known as Munkres algorithm or Kuhn-Munkres
algorithm, is a method for solving the assignment problem, for example
assigning workers to jobs. It finds the assignment that minimizes the total
cost.

This is a C++ version (with slight modifications) of a hungarian algorithm
implementation by Markus Buehren. The original code is a few mex-functions for
use in MATLAB, found here: 
http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem

## Build

    mkdir build
    cd build
    cmake ..
    make

## Test

    cd build
    ctest -V

## Install

TODO
