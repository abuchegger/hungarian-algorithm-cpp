///////////////////////////////////////////////////////////////////////////////
// A C++ implementation of the Hungarian algorithm for solving the assignment
// problem.
//
// Both this code and the original MATLAB code by Markus Buehren are published
// under the BSD license (see LICENSE.txt).
//
// Copyright (c) 2016, Cong Ma (mcximing)
// Copyright (c) 2018, Alexander Buchegger (abuchegger)
//
#define BOOST_TEST_MODULE hungarian_algorithm

#include "hungarian_algorithm.h"
#include <boost/chrono.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace ublas = boost::numeric::ublas;

class GroupingNumPunct : public std::numpunct<char>
{
protected:
  virtual char do_thousands_sep() const
  {
    return ',';
  }

  virtual std::string do_grouping() const
  {
    return "\x03"; // Group every 3 digits
  }
};

struct GroupingNumPunctFixture
{
  GroupingNumPunctFixture()
  {
    std::cout.imbue(std::locale(std::locale::classic(), new GroupingNumPunct));
  }
};

struct DurationFormatter
{
  explicit DurationFormatter(const double d)
    : d_(d)
  {}

  friend std::ostream& operator<<(std::ostream& out, const DurationFormatter& d)
  {
    static const char* const prefixes = " munpfa";
    static const double factors[] = {1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18};
    const char* prefix = &prefixes[0];
    for (const double* factor = &factors[0]; *prefix != 0; ++prefix, ++factor)
    {
      const double value = d.d_ * *factor;
      if (value >= 1.0)
      {
        out << std::fixed << std::setprecision(2) << value;
        if (std::isalpha(*prefix) != 0)
        {
          out << *prefix;
        }
        out << "s";
        return out;
      }
    }
    out << std::scientific << std::setprecision(2) << d.d_ << "s";
    return out;
  }

  const double d_;
};

BOOST_GLOBAL_FIXTURE(GroupingNumPunctFixture);

template<typename T>
void print_matrix(std::ostream& out, const ublas::matrix<T>& matrix, const std::string& indent = std::string(),
                  const std::string& colsep = ", ", const std::string& rowsep = ";\n")
{
  // Save formatting info for later:
  const std::streamsize width = out.width(0);
  const std::streamsize precision = out.precision();
  for (std::size_t row = 0; row < matrix.size1(); ++row)
  {
    out << indent;
    for (std::size_t col = 0; col < matrix.size2(); ++col)
    {
      if (col != 0)
      {
        out << colsep;
      }
      out.width(width);
      out.precision(precision);
      out << matrix(row, col);
    }
    out << rowsep;
  }
}

template<typename Cost>
void print_assignment(std::ostream& out, const std::vector<std::size_t>& assignment,
                      const ublas::matrix<Cost>& cost_matrix, const std::string& indent = std::string())
{
  for (std::size_t row = 0; row < assignment.size(); ++row)
  {
    out << indent << row << " -> ";
    const std::size_t col = assignment[row];
    if (col < cost_matrix.size2())
    {
      out << col << " (cost: " << cost_matrix(row, col) << ")" << std::endl;
    }
    else
    {
      out << "invalid" << std::endl;
    }
  }
}

template<typename T>
ublas::matrix<T> make_matrix(const T* const data, const std::size_t num_rows, const std::size_t num_cols)
{
  ublas::matrix <T> matrix(num_rows, num_cols);
  std::copy(&data[0], &data[num_rows * num_cols], matrix.data().begin());
  return matrix;
}

template<typename T>
std::vector<T> make_vector(const T* const data, const std::size_t size)
{
  return std::vector<T>(&data[0], &data[size]);
}

template<typename SolvingFunction, typename Cost, typename CostComparator>
void assert_solver_result(const SolvingFunction& solving_function, const ublas::matrix<Cost>& cost_matrix,
                          const Cost expected_cost, const std::vector<std::size_t>& expected_assignment,
                          const CostComparator& cost_comparator)
{
  std::vector<std::size_t> assignment;
  const Cost total_cost = solving_function(cost_matrix, cost_matrix.size1(), cost_matrix.size2(), assignment,
                                           cost_comparator);
  BOOST_CHECK_EQUAL(total_cost, expected_cost);
  BOOST_CHECK_EQUAL(assignment.size(), cost_matrix.size1());
  std::size_t num_invalid_assignments = 0;
  for (std::size_t row = 0; row < cost_matrix.size1(); ++row)
  {
    if (assignment.at(row) >= cost_matrix.size2())
    {
      ++num_invalid_assignments;
    }
  }
  const std::size_t expected_num_invalid_assignments = (cost_matrix.size1() > cost_matrix.size2())
                                                       ? cost_matrix.size1() - cost_matrix.size2() : 0;
  BOOST_CHECK_EQUAL(num_invalid_assignments, expected_num_invalid_assignments);
  BOOST_CHECK_EQUAL_COLLECTIONS(assignment.begin(), assignment.end(),
                                expected_assignment.begin(), expected_assignment.end());
}

template<typename Cost, typename CostComparator>
void assert_solvers_result(const ublas::matrix<Cost>& cost_matrix, const Cost expected_cost,
                           const std::vector<std::size_t>& expected_assignment,
                           const CostComparator& cost_comparator)
{
  typedef Cost SolvingFunction(const ublas::matrix<Cost>&, const std::size_t, const std::size_t,
                               std::vector<std::size_t>&, const CostComparator&);
  std::cout << "  Testing Hungarian solver" << std::endl;
  assert_solver_result(static_cast<SolvingFunction*>(&hungarian_algorithm::solve), cost_matrix, expected_cost,
                       expected_assignment, cost_comparator);
  std::cout << "  Testing brute force solver" << std::endl;
  assert_solver_result(static_cast<SolvingFunction*>(&hungarian_algorithm::solveBruteForce), cost_matrix, expected_cost,
                       expected_assignment, cost_comparator);
}

template<typename Cost>
void assert_solvers_result(const ublas::matrix<Cost>& cost_matrix, const Cost expected_cost,
                           const std::vector<std::size_t>& expected_assignment)
{
  assert_solvers_result<Cost>(cost_matrix, expected_cost, expected_assignment, std::less<Cost>());
}

template<typename Cost>
void assert_solvers_return_equal_cost(const ublas::matrix<Cost>& cost_matrix)
{
  std::vector<std::size_t> assignment, assignment_brute_force;
  const Cost total_cost = hungarian_algorithm::solve<Cost>(
    cost_matrix, cost_matrix.size1(), cost_matrix.size2(), assignment);
  const Cost total_cost_brute_force = hungarian_algorithm::solveBruteForce<Cost>(
    cost_matrix, cost_matrix.size1(), cost_matrix.size2(), assignment_brute_force);
  if (total_cost != total_cost_brute_force)
  {
    std::cerr << "total_cost != total_cost_brute_force:" << std::endl;
    std::cerr << "  total_cost: " << total_cost << std::endl;
    std::cerr << "  total_cost_brute_force: " << total_cost_brute_force << std::endl;
    std::cerr << "  cost_matrix: [" << std::endl << std::setw(3);
    print_matrix(std::cerr, cost_matrix, "    ");
    std::cerr << "  ]" << std::endl;
    std::cerr << "  Hungarian assignment:" << std::endl;
    print_assignment(std::cerr, assignment, cost_matrix, "    ");
    std::cerr << "  brute force assignment:" << std::endl;
    print_assignment(std::cerr, assignment_brute_force, cost_matrix, "    ");
  }
  BOOST_CHECK_EQUAL(total_cost, total_cost_brute_force);
}

void fill_matrix_with_random_ints(ublas::matrix<int>& cost_matrix, const int min_cost, const int max_cost)
{
  // Note: to prevent overflows when summing, max_cost should not exceed RAND_MAX / cost_matrix.size1().
  for (std::size_t row = 0; row < cost_matrix.size1(); ++row)
  {
    for (std::size_t col = 0; col < cost_matrix.size2(); ++col)
    {
      cost_matrix(row, col) = min_cost + std::rand() % (max_cost - min_cost);
    }
  }
}

void test_solvers_random_square(const std::size_t num_rows, const std::size_t num_runs, const int min_cost,
                                const int max_cost)
{
  ublas::matrix<int> cost_matrix(num_rows, num_rows, 0);
  for (std::size_t i = 0; i < num_runs; ++i)
  {
    std::cout << ".";
    std::cout.flush();
    fill_matrix_with_random_ints(cost_matrix, min_cost, max_cost);
    assert_solvers_return_equal_cost(cost_matrix);
  }
}

template<typename SolvingFunction, typename Cost>
double measure_solver_timing(const SolvingFunction& solving_function, const ublas::matrix<Cost>& cost_matrix)
{
  typedef boost::chrono::high_resolution_clock Clock;
  std::vector<std::size_t> assignment;
  const Clock::time_point start_time(Clock::now());
  solving_function(cost_matrix, cost_matrix.size1(), cost_matrix.size2(), assignment);
  const boost::chrono::duration<double> dt(Clock::now() - start_time);

  // Compute number of combinations = factorial of larger dimension of matrix:
  double c = 1;  // size_t gets too small fast
  for (std::size_t row = 1; row <= std::max(cost_matrix.size1(), cost_matrix.size2()); ++row)
  {
    c *= row;
  }

  std::cout << "  Time to solve " << cost_matrix.size1() << "x" << cost_matrix.size2() << " assignment problem ("
            << std::resetiosflags(std::ios::floatfield) << std::setprecision(8) << c << " combinations): "
            << DurationFormatter(dt.count()) << " (" << DurationFormatter(dt.count() / c) << " per combination)"
            << std::endl;
  return dt.count();
}

template<typename SolvingFunction>
double test_solver_timing(const SolvingFunction& solving_function, const std::size_t num_rows)
{
  ublas::matrix<int> cost_matrix(num_rows, num_rows, 0);
  fill_matrix_with_random_ints(cost_matrix, 0, 1000);
  return measure_solver_timing(solving_function, cost_matrix);
}

BOOST_AUTO_TEST_CASE(test_matrix)
{
  // Tests whether the matrix class works as we expect it to, so that the solvers and tests work as expected.
  const double costs[] = {10, 19, 8, 15, 0,
                          10, 18, 7, 17, 0,
                          13, 16, 9, 14, 0,
                          12, 19, 8, 18, 0};
  const ublas::matrix<double> cost_matrix(make_matrix(costs, 4, 5));

  BOOST_CHECK_EQUAL(cost_matrix.size1(), 4);
  BOOST_CHECK_EQUAL(cost_matrix.size2(), 5);

  std::size_t i = 0;
  for (std::size_t row = 0; row < cost_matrix.size1(); ++row)
  {
    for (std::size_t col = 0; col < cost_matrix.size2(); ++col)
    {
      BOOST_CHECK_EQUAL(cost_matrix(row, col), costs[i++]);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_solvers_0x0)
{
  const boost::numeric::ublas::matrix<int> cost_matrix(0, 0, 0);
  const std::vector<std::size_t> expected_assignment;
  assert_solvers_result(cost_matrix, 0, expected_assignment);
}

BOOST_AUTO_TEST_CASE(test_solvers_0x2)
{
  const boost::numeric::ublas::matrix<int> cost_matrix(0, 2, 0);
  const std::vector<std::size_t> expected_assignment;
  assert_solvers_result(cost_matrix, 0, expected_assignment);
}

BOOST_AUTO_TEST_CASE(test_solvers_2x0)
{
  const boost::numeric::ublas::matrix<int> cost_matrix(2, 0, 0);
  const std::vector<std::size_t> expected_assignment(2, 0);
  assert_solvers_result(cost_matrix, 0, expected_assignment);
}

BOOST_AUTO_TEST_CASE(test_solvers_1x1)
{
  const boost::numeric::ublas::matrix<int> cost_matrix(1, 1, 1);
  const std::vector<std::size_t> expected_assignment(1, 0);
  assert_solvers_result(cost_matrix, 1, expected_assignment);
}

BOOST_AUTO_TEST_CASE(test_solvers_1x2)
{
  const int costs[] = {1, 0};
  ublas::matrix<int> cost_matrix(1, 2);
  std::copy(&costs[0], &costs[2], cost_matrix.data().begin());
  const std::vector<std::size_t> expected_assignment(1, 1);
  assert_solvers_result(cost_matrix, 0, expected_assignment);
}

BOOST_AUTO_TEST_CASE(test_solvers_2x1)
{
  const int costs[] = {1,
                       0};
  const std::size_t expected_assignment[] = {1, 0};
  assert_solvers_result(make_matrix(costs, 2, 1), 0, make_vector(expected_assignment, 2));
}

BOOST_AUTO_TEST_CASE(test_solvers_2x2)
{
  const int costs[] = {2, 4,
                       0, 1};
  const std::size_t expected_assignment[] = {0, 1};
  assert_solvers_result(make_matrix(costs, 2, 2), 3, make_vector(expected_assignment, 2));
}

BOOST_AUTO_TEST_CASE(test_solvers_2x3)
{
  const int costs[] = {1, 2, 0,
                       1, 3, 0};
  const std::size_t expected_assignment[] = {0, 2};
  assert_solvers_result(make_matrix(costs, 2, 3), 1, make_vector(expected_assignment, 2));
}

BOOST_AUTO_TEST_CASE(test_solvers_3x2)
{
  const int costs[] = {1, 2,
                       0, 1,
                       3, 0};
  const std::size_t expected_assignment[] = {2, 0, 1};
  assert_solvers_result(make_matrix(costs, 3, 2), 0, make_vector(expected_assignment, 3));
}

BOOST_AUTO_TEST_CASE(test_solvers_3x3)
{
  const int costs[] = {1, 2, 4,
                       0, 2, -1,
                       3, 0, -3};
  const std::size_t expected_assignment[] = {1, 0, 2};
  assert_solvers_result(make_matrix(costs, 3, 3), -1, make_vector(expected_assignment, 3));
}

BOOST_AUTO_TEST_CASE(test_solvers_3x3_nonnumeric)
{
  using boost::chrono::seconds;
  const seconds costs[] = {seconds(1), seconds(2), seconds(4),
                           seconds(0), seconds(2), seconds(-1),
                           seconds(3), seconds(0), seconds(-3)};
  const std::size_t expected_assignment[] = {1, 0, 2};
  assert_solvers_result(make_matrix(costs, 3, 3), seconds(-1), make_vector(expected_assignment, 3));
}

BOOST_AUTO_TEST_CASE(test_solvers_3x3_nonnumeric_maximize)
{
  using boost::chrono::seconds;
  const seconds costs[] = {seconds(1), seconds(2), seconds(4),
                           seconds(0), seconds(2), seconds(-1),
                           seconds(3), seconds(0), seconds(-3)};
  const std::size_t expected_assignment[] = {2, 1, 0};
  assert_solvers_result(make_matrix(costs, 3, 3), seconds(9), make_vector(expected_assignment, 3),
                        std::greater<seconds>());
}

BOOST_AUTO_TEST_CASE(test_solvers_4x5)
{
  const double costs[] = {10, 19, 8, 15, 0,
                          10, 18, 7, 17, 0,
                          13, 16, 9, 14, 0,
                          12, 19, 8, 18, 0};
  const std::size_t expected_assignment[] = {0, 2, 3, 4};
  assert_solvers_result(make_matrix(costs, 4, 5), 31.0, make_vector(expected_assignment, 4));
}

BOOST_AUTO_TEST_CASE(test_solvers_random_nonnegative_int_squares)
{
  std::cout << "  ";
  test_solvers_random_square(8, 10, 0, 1000);
  std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_solvers_random_int_squares)
{
  std::cout << "  ";
  test_solvers_random_square(8, 10, -1000, 1000);
  std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_solvers_timings)
{
  // These are informative only, and to check for any severe errors (segfaults, endless loops, ...):
  typedef int SolvingFunction(const ublas::matrix<int>&, const std::size_t, const std::size_t,
                              std::vector<std::size_t>&);
  std::cout << "Brute force method:" << std::endl;
  const double brute_force_timing = test_solver_timing(
    static_cast<SolvingFunction*>(&hungarian_algorithm::solveBruteForce), 10);
  std::cout << "Hungarian method:" << std::endl;
  const double hungarian_timing = test_solver_timing(static_cast<SolvingFunction*>(&hungarian_algorithm::solve), 10);
  BOOST_CHECK_LT(hungarian_timing, brute_force_timing);
  test_solver_timing(static_cast<SolvingFunction*>(&hungarian_algorithm::solve), 100);
  test_solver_timing(static_cast<SolvingFunction*>(&hungarian_algorithm::solve), 200);
}
