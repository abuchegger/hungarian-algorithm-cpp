# Hungarian Algorithm C++ Library

The Hungarian algorithm, also known as Munkres algorithm or Kuhn-Munkres
algorithm, is an efficient method for solving the assignment problem, for
example, assigning workers to jobs while minimizing the total cost. As the
number of possible assignments is the factorial of the number of workers and
jobs, finding the optimal assignment using brute force quickly becomes
infeasible. While the Hungarian algorithm (among others) can easily find
solutions for 100x100 problems, brute force dwindles above 10x10 problems.

This is a C++98, template-based implementation of the Hungarian
algorithm, based on the C++ implementation by Cong Ma (mcximing). The original
MATLAB code was written by Markus Buehren and can be found here:
http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem

## Use

To add the package as a dependency of your CMake package, add the following
lines to your CMakeLists.txt:

```CMake
find_package(hungarian-algorithm-cpp REQUIRED)
include_directories(${hungarian-algorithm-cpp_INCLUDE_DIRS})
target_link_libraries(your_target ${hungarian-algorithm-cpp_LIBRARIES})
```

Using the solver in your code is quite simple (see tests for more examples):

```C++
#include <boost/numeric/ublas/matrix.hpp>
#include <hungarian_algorithm.h>
#include <iostream>

void helloHungarian()
{
  boost::numeric::ublas::matrix<int> cost_matrix(6, 7, 0);
  cost_matrix(3, 4) = 42;
  ...
  std::vector<std::size_t> assignment;
  const int total_cost = hungarian_algorithm::solve(
    cost_matrix, cost_matrix.size1(), cost_matrix.size2(), assignment);
  std::cout << "Total cost = " << total_cost << std::endl;
}
```

Each item in the resulting assignment vector corresponds to one row in the cost
matrix, and contains the index of the assigned column. If no assignment was
possible (which happens for non-quadratic cost matrices), the item contains a
value greater than or equal to the number of columns.

Note that although we use a Boost uBLAS matrix in the example above (and also
in the tests), anything that supports being used like `cost_fn(row, col)`,
like most matrix classes, can be used as a cost function.

## License

The C++ code as well as the original MATLAB code is licensed under a BSD
2-clause license. See LICENSE.txt for details.

## Dependencies

The algorithm only depends on the C++98 standard library. The tests depend on
the Boost library (tested with version 1.58.0).

## Build

```
mkdir build
cd build
cmake ..
make
```

## Test

After building, run the following command from the `build` directory:

```
ctest -V
```

## Install

After building, run the following command from the `build` directory:

```
sudo make install
```

