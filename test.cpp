#define BOOST_TEST_MODULE test

#include "hungarian_algorithm.h"
#include <boost/chrono.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace ublas = boost::numeric::ublas;

class MyNumPunct : public std::numpunct<char>
{
protected:
  virtual char do_thousands_sep() const
  { return ','; }
  virtual std::string do_grouping() const
  { return "\03"; } // Group every 3 digits
};

struct MyNumPunctFixture
{
  MyNumPunctFixture()
  {
    std::cout.imbue(std::locale(std::locale::classic(), new MyNumPunct));
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

BOOST_GLOBAL_FIXTURE(MyNumPunctFixture);

template<typename T>
ublas::matrix<T> make_matrix(const T* const data, const std::size_t num_rows, const std::size_t num_cols)
{
  ublas::matrix<T> matrix(num_rows, num_cols);
  std::copy(&data[0], &data[num_rows * num_cols], matrix.data().begin());
  return matrix;
}

template<typename T>
std::vector<T> make_vector(const T* const data, const std::size_t size)
{
  return std::vector<T>(&data[0], &data[size]);
}

template<typename Cost>
void print_cost_matrix(std::ostream& out, const ublas::matrix<Cost>& cost_matrix,
                       const std::string& indent = std::string())
{
  out << indent << "cost_matrix: [" << std::endl << std::setw(3) << cost_matrix << indent << "]" << std::endl;
}

template<typename Size, typename Cost>
void print_assignment(std::ostream& out, const std::vector<Size>& assignment,
                      const ublas::matrix<Cost>& cost_matrix, const std::string& indent = std::string())
{
  out << "assignment:" << std::endl;
  // TODO: move assignment printing into utility function
  //cost_matrix.printAssignment(out, assignment, indent);
}

template<typename SolvingMethod, typename Cost>
void assert_solver_result(const ublas::matrix<Cost>& cost_matrix,
                          const std::size_t expected_num_invalid_assignments, const Cost expected_cost,
                          const std::vector<std::size_t>& expected_assignment)
{
  hungarian_algorithm::Solver<Cost, hungarian_algorithm::MinimizeCosts, SolvingMethod> solver;
  std::vector<std::size_t> assignment;
  const typename hungarian_algorithm::Solver<Cost>::CombinedCost total_cost
    = solver.solve(cost_matrix, cost_matrix.size1(), cost_matrix.size2(), assignment);
  BOOST_CHECK_EQUAL(total_cost.first, expected_num_invalid_assignments);
  BOOST_CHECK_EQUAL(total_cost.second, expected_cost);
  BOOST_CHECK_EQUAL_COLLECTIONS(assignment.begin(), assignment.end(),
                                expected_assignment.begin(), expected_assignment.end());
}

template<typename Cost>
void assert_solvers_result(const ublas::matrix<Cost>& cost_matrix,
                           const std::size_t expected_num_invalid_assignments, const Cost expected_cost,
                           const std::vector<std::size_t>& expected_assignment)
{
  std::cout << "  Testing Hungarian solver" << std::endl;
  assert_solver_result<hungarian_algorithm::HungarianMethod>(cost_matrix, expected_num_invalid_assignments,
                                                             expected_cost, expected_assignment);
  std::cout << "  Testing brute force solver" << std::endl;
  assert_solver_result<hungarian_algorithm::BruteForceMethod>(cost_matrix, expected_num_invalid_assignments,
                                                              expected_cost, expected_assignment);
  std::cout << "  Testing new Hungarian solver" << std::endl;
  hungarian_algorithm::HungarianSolver solver;
  std::vector<std::size_t> assignment;
  const Cost total_cost = solver.solve<Cost>(cost_matrix, cost_matrix.size1(), cost_matrix.size2(), assignment);
  BOOST_CHECK_EQUAL(total_cost, expected_cost);
  BOOST_CHECK_EQUAL_COLLECTIONS(assignment.begin(), assignment.end(),
                                expected_assignment.begin(), expected_assignment.end());
}

template<typename Cost>
void assert_solvers_return_equal_cost(const ublas::matrix<Cost>& cost_matrix)
{
  hungarian_algorithm::Solver<Cost> munkres;
  hungarian_algorithm::Solver<Cost, hungarian_algorithm::MinimizeCosts, hungarian_algorithm::BruteForceMethod>
    brute_force_solver;
  typedef typename hungarian_algorithm::Solver<Cost>::CombinedCost CombinedCost;
  std::vector<std::size_t> assignment, assignment_brute_force;
  const CombinedCost total_cost = munkres.solve(cost_matrix, cost_matrix.size1(), cost_matrix.size2(), assignment);
  const CombinedCost total_cost_brute_force = brute_force_solver.solve(cost_matrix, cost_matrix.size1(),
                                                                       cost_matrix.size2(), assignment_brute_force);
  if (total_cost != total_cost_brute_force)
  {
    std::cerr << "total_cost != total_cost_brute_force:" << std::endl;
    std::cerr << "  total_cost: {ninv: " << total_cost.first << ", cost:" << total_cost.second << "}" << std::endl;
    std::cerr << "  total_cost_brute_force: {ninv: " << total_cost_brute_force.first << ", cost:"
              << total_cost_brute_force.second << "}" << std::endl;
    print_cost_matrix(std::cerr, cost_matrix, "  ");
    std::cerr << "  Hungarian ";
    print_assignment(std::cerr, assignment, cost_matrix, "    ");
    std::cerr << "  brute force ";
    print_assignment(std::cerr, assignment_brute_force, cost_matrix, "    ");
  }
  BOOST_CHECK_EQUAL(total_cost.first, total_cost_brute_force.first);
  BOOST_CHECK_EQUAL(total_cost.second, total_cost_brute_force.second);
}

void fill_matrix_with_random_ints(ublas::matrix<int>& cost_matrix)
{
  // To prevent overflows when summing, and to have comprehendable numbers for debugging:
  const int max_cost = std::min(1000, RAND_MAX / static_cast<int>(cost_matrix.size1()));
  for (std::size_t row = 0; row < cost_matrix.size1(); ++row)
  {
    for (std::size_t col = 0; col < cost_matrix.size2(); ++col)
    {
      cost_matrix(row, col) = std::rand() % max_cost;
    }
  }
}

void test_solver_random_square(const std::size_t num_rows, const std::size_t num_runs)
{
  ublas::matrix<int> cost_matrix(num_rows, num_rows, 0);
  for (std::size_t i = 0; i < num_runs; ++i)
  {
    fill_matrix_with_random_ints(cost_matrix);
    assert_solvers_return_equal_cost(cost_matrix);
  }
}

template<typename Solver, typename Cost>
void measure_solver_timing(Solver& solver, const ublas::matrix<Cost>& cost_matrix)
{
  typedef boost::chrono::high_resolution_clock Clock;
  std::vector<std::size_t> assignment;
  const Clock::time_point start_time(Clock::now());
  solver.solve(cost_matrix, assignment);
  const boost::chrono::duration<double> dt(Clock::now() - start_time);

  double c = 1;  // size_t gets too small fast
  for (std::size_t row = 1; row <= std::max(cost_matrix.size1(), cost_matrix.size2()); ++row)
  {
    c *= row;
  }

  std::cout << "Time to solve " << cost_matrix.size1() << "x" << cost_matrix.size2() << " assignment problem ("
            << std::resetiosflags(std::ios::floatfield) << std::setprecision(8) << c << " combinations): "
            << DurationFormatter(dt.count()) << " (" << DurationFormatter(dt.count() / c) << " per combination)"
            << std::endl;
}

template<typename SolvingMethod>
void test_solver_timing(const std::size_t num_rows)
{
  ublas::matrix<int> cost_matrix(num_rows, num_rows, 0);
  fill_matrix_with_random_ints(cost_matrix);
  hungarian_algorithm::Solver<int, hungarian_algorithm::MinimizeCosts, SolvingMethod> solver;
  measure_solver_timing(solver, cost_matrix);
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

BOOST_AUTO_TEST_CASE(test_solver_construction)
{
  hungarian_algorithm::Solver<double> munkres;
  static_cast<void>(munkres);  // Use variable so it's actually created
}

BOOST_AUTO_TEST_CASE(test_solver_1x1)
{
  const boost::numeric::ublas::matrix<int> cost_matrix(1, 1, 1);
  const std::vector<std::size_t> expected_assignment(1, 0);
  assert_solvers_result(cost_matrix, 0, 1, expected_assignment);
}

BOOST_AUTO_TEST_CASE(test_solver_1x2)
{
  const int costs[] = {1, 0};
  ublas::matrix<int> cost_matrix(1, 2);
  std::copy(&costs[0], &costs[2], cost_matrix.data().begin());
  const std::vector<std::size_t> expected_assignment(1, 1);
  assert_solvers_result(cost_matrix, 0, 0, expected_assignment);
}

//BOOST_AUTO_TEST_CASE(test_solver_2x1)
//{
//  const int costs[] = {1,
//                       0};
//  const std::size_t expected_assignment[] = {std::numeric_limits<std::size_t>::max(), 0};
//  assert_solvers_result(make_matrix(costs, 2, 1), 1, 0, make_vector(expected_assignment, 2));
//}

BOOST_AUTO_TEST_CASE(test_solver_2x2)
{
  const int costs[] = {2, 4,
                       0, 1};
  const std::size_t expected_assignment[] = {0, 1};
  assert_solvers_result(make_matrix(costs, 2, 2), 0, 3, make_vector(expected_assignment, 2));
}

BOOST_AUTO_TEST_CASE(test_solver_2x3)
{
  const int costs[] = {1, 2, 0,
                       1, 3, 0};
  const std::size_t expected_assignment[] = {0, 2};
  assert_solvers_result(make_matrix(costs, 2, 3), 0, 1, make_vector(expected_assignment, 2));
}

//BOOST_AUTO_TEST_CASE(test_solver_3x2)
//{
//  const int costs[] = {1, 2,
//                       0, 1,
//                       3, 0};
//  const std::size_t expected_assignment[] = {std::numeric_limits<std::size_t>::max(), 0, 1};
//  assert_solvers_result(make_matrix(costs, 2, 3), 1, 0, make_vector(expected_assignment, 3));
//}

BOOST_AUTO_TEST_CASE(test_solver_4x5)
{
  const double costs[] = {10, 19, 8, 15, 0,
                          10, 18, 7, 17, 0,
                          13, 16, 9, 14, 0,
                          12, 19, 8, 18, 0};
  const std::size_t expected_assignment[] = {0, 2, 3, 4};
  assert_solvers_result(make_matrix(costs, 4, 5), 0, 31.0, make_vector(expected_assignment, 4));
}

//BOOST_AUTO_TEST_CASE(test_solver_random_squares)
//{
//  test_solver_random_square(8, 10);
//}

//BOOST_AUTO_TEST_CASE(test_solver_timings)
//{
//  std::cout << "Brute force method:" << std::endl;
//  test_solver_timing<hungarian_algorithm::BruteForceMethod>(10);
//  std::cout << "Hungarian method:" << std::endl;
//  test_solver_timing<hungarian_algorithm::HungarianMethod>(10);
//  test_solver_timing<hungarian_algorithm::HungarianMethod>(100);
//  test_solver_timing<hungarian_algorithm::HungarianMethod>(110);
//}
