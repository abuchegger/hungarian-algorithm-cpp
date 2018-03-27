#include <boost/chrono.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "hungarian_algorithm.h"

template<typename Cost>
void printCostMatrix(std::ostream& out, const hungarian_algorithm::Matrix<Cost>& cost_matrix,
                     const std::string& indent = std::string())
{
  out << indent << "cost_matrix: [" << std::endl << std::setw(3);
  cost_matrix.print(out, indent + "  [", ", ", "],\n");
  out << indent << "]" << std::endl;
}

template<typename Size, typename Cost>
void printAssignment(std::ostream& out, const std::vector<Size>& assignment,
                     const hungarian_algorithm::Matrix<Cost>& cost_matrix, const std::string& indent = std::string())
{
  out << "assignment:" << std::endl;
  cost_matrix.printAssignment(out, assignment, indent);
}

template<typename Cost, typename Size>
Cost solveAssignmentBruteForce(const hungarian_algorithm::Matrix<Cost>& cost_matrix, std::vector<Size>& assignment)
{
  hungarian_algorithm::Solver<Cost, hungarian_algorithm::MinimizeCosts, hungarian_algorithm::BruteForceMethod> solver;
  return solver.solve(cost_matrix, assignment);
}

template<typename Cost>
void assertSolversReturnEqualCost(const hungarian_algorithm::Matrix<Cost>& cost_matrix)
{
  hungarian_algorithm::Solver<Cost> munkres;
  std::vector<std::size_t> assignment, assignment_brute_force;
  const int total_cost = munkres.solve(cost_matrix, assignment);
  const int total_cost_brute_force = solveAssignmentBruteForce(cost_matrix, assignment_brute_force);
  if (total_cost != total_cost_brute_force)
  {
    std::cerr << "total_cost != total_cost_brute_force:" << std::endl;
    std::cerr << "  total_cost: " << total_cost << std::endl;
    std::cerr << "  total_cost_brute_force: " << total_cost_brute_force << std::endl;
    printCostMatrix(std::cerr, cost_matrix, "  ");
    std::cerr << "  Hungarian ";
    printAssignment(std::cerr, assignment, cost_matrix, "    ");
    std::cerr << "  brute force ";
    printAssignment(std::cerr, assignment_brute_force, cost_matrix, "    ");
  }
  assert(total_cost == total_cost_brute_force);
}

void test_matrix()
{
  const double costs[] = {10, 19, 8, 15, 0,
                          10, 18, 7, 17, 0,
                          13, 16, 9, 14, 0,
                          12, 19, 8, 18, 0};
  const hungarian_algorithm::Matrix<double> cost_matrix(4, 5, std::vector<double>(&costs[0], &costs[20]));

  assert(cost_matrix.numRows() == 4);
  assert(cost_matrix.numCols() == 5);

  std::size_t i = 0;
  for (std::size_t row = 0; row < cost_matrix.numRows(); ++row)
  {
    for (std::size_t col = 0; col < cost_matrix.numCols(); ++col)
    {
      assert(cost_matrix(row, col) == costs[i++]);
    }
  }
}

void test_solver_construction()
{
  hungarian_algorithm::Solver<double> munkres;
  static_cast<void>(munkres);  // Use variable so it's actually created
}

void test_solver_1x1()
{
  const hungarian_algorithm::Matrix<int> cost_matrix(1, 1, 1);
  std::vector<std::size_t> assignment;
  hungarian_algorithm::Solver<int> munkres;
  const int total_cost = munkres.solve(cost_matrix, assignment);
  assert(total_cost == 1);
}

void test_solver_1x2()
{
  const int costs[] = {1, 0};
  const hungarian_algorithm::Matrix<int> cost_matrix(1, 2, std::vector<int>(&costs[0], &costs[2]));
  assertSolversReturnEqualCost(cost_matrix);
}

void test_solver_2x3()
{
  const int costs[] = {1, 2, 0,
                       1, 3, 0};
  const hungarian_algorithm::Matrix<int> cost_matrix(2, 3, std::vector<int>(&costs[0], &costs[6]));
  assertSolversReturnEqualCost(cost_matrix);
}

void test_solver_4x5()
{
  const double costs[] = {10, 19, 8, 15, 0,
                          10, 18, 7, 17, 0,
                          13, 16, 9, 14, 0,
                          12, 19, 8, 18, 0};
  const hungarian_algorithm::Matrix<double> cost_matrix(4, 5, std::vector<double>(&costs[0], &costs[20]));
  assertSolversReturnEqualCost(cost_matrix);
}

void fill_matrix_with_random_ints(hungarian_algorithm::Matrix<int>& cost_matrix)
{
  // To prevent overflows when summing, and to have comprehendable numbers for debugging:
  const int max_cost = std::min(1000, RAND_MAX / static_cast<int>(cost_matrix.numRows()));
  for (std::size_t row = 0; row < cost_matrix.numRows(); ++row)
  {
    for (std::size_t col = 0; col < cost_matrix.numCols(); ++col)
    {
      cost_matrix(row, col) = std::rand() % max_cost;
    }
  }
}

void test_solver_random_square(const std::size_t num_rows, const std::size_t num_runs)
{
  hungarian_algorithm::Matrix<int> cost_matrix(num_rows, num_rows, 0);
  for (std::size_t i = 0; i < num_runs; ++i)
  {
    fill_matrix_with_random_ints(cost_matrix);
    assertSolversReturnEqualCost(cost_matrix);
  }
}

void printDuration(std::ostream& out, const double d)
{
  static const char* const prefixes = " munpfa";
  static const double factors[] = {1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18};
  const char* prefix = &prefixes[0];
  for (const double* factor = &factors[0]; *prefix != 0; ++prefix, ++factor)
  {
    const double value = d * *factor;
    if (value >= 1.0)
    {
      out << std::fixed << std::setprecision(2) << value;
      if (std::isalpha(*prefix) != 0)
      {
        out << *prefix;
      }
      out << "s";
      return;
    }
  }
  out << std::scientific << std::setprecision(2) << d << "s";
}

template<typename Solver, typename Cost>
void measure_solver_timing(Solver& solver, const hungarian_algorithm::Matrix<Cost>& cost_matrix)
{
  typedef boost::chrono::high_resolution_clock Clock;
  std::vector<std::size_t> assignment;
  const Clock::time_point start_time(Clock::now());
  solver.solve(cost_matrix, assignment);
  const boost::chrono::duration<double> dt(Clock::now() - start_time);

  double c = 1;  // size_t gets too small fast
  for (std::size_t row = 1; row <= std::max(cost_matrix.numRows(), cost_matrix.numCols()); ++row)
  {
    c *= row;
  }

  std::cout << "Time to solve " << cost_matrix.numRows() << "x" << cost_matrix.numCols() << " assignment problem ("
            << std::resetiosflags(std::ios::floatfield) << std::setprecision(8) << c << " combinations): ";
  printDuration(std::cout, dt.count());
  std::cout << " (";
  printDuration(std::cout, dt.count() / c);
  std::cout << " per combination)" << std::endl;
}

class MyNumPunct : public std::numpunct<char>
{
protected:
  virtual char do_thousands_sep() const
  { return ','; }
  virtual std::string do_grouping() const
  { return "\03"; } // Group every 3 digits
};

template<typename SolvingMethod>
void test_solver_timing(const std::size_t num_rows)
{
  hungarian_algorithm::Matrix<int> cost_matrix(num_rows, num_rows, 0);
  fill_matrix_with_random_ints(cost_matrix);
//  hungarian_algorithm::Solver<int> solver;
  hungarian_algorithm::Solver<int, hungarian_algorithm::MinimizeCosts, SolvingMethod> solver;
  measure_solver_timing(solver, cost_matrix);
}

void test_solver_timings()
{
  std::cout << "Brute force method:" << std::endl;
  test_solver_timing<hungarian_algorithm::BruteForceMethod>(10);
  std::cout << "Hungarian method:" << std::endl;
  test_solver_timing<hungarian_algorithm::HungarianMethod>(10);
  test_solver_timing<hungarian_algorithm::HungarianMethod>(100);
  test_solver_timing<hungarian_algorithm::HungarianMethod>(110);
}

int main()
{
  std::cout.imbue(std::locale(std::locale::classic(), new MyNumPunct));
  test_matrix();
  test_solver_construction();
  test_solver_1x1();
  test_solver_1x2();
  test_solver_2x3();
  test_solver_4x5();
//  test_solver_random_square(5, 20);
  test_solver_timings();
  return 0;
}
