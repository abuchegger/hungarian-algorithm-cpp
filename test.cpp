#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "hungarian_algorithm.h"


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

template<typename Cost>
void printCostMatrix(std::ostream& out, const hungarian_algorithm::Matrix<Cost>& cost_matrix)
{
  out << "cost_matrix:" << std::endl << std::setw(3);
  cost_matrix.print(out, "  ");
}

template<typename Size, typename Cost>
void printAssignment(std::ostream& out, const std::vector<Size>& assignment,
                     const hungarian_algorithm::Matrix<Cost>& cost_matrix)
{
  out << "assignment:" << std::endl;
  cost_matrix.printAssignment(out, assignment, "  ");
}

template<typename Cost, typename Size>
Cost solveAssignmentBruteForce(const hungarian_algorithm::Matrix<Cost>& cost_matrix, std::vector<Size>& assignment)
{
  const std::size_t size_diff = std::max(cost_matrix.numRows(), cost_matrix.numCols())
    - std::min(cost_matrix.numRows(), cost_matrix.numCols());
  const Cost invalid_cost = std::numeric_limits<Cost>::max() / Cost(size_diff + 1);  // Prevent overflow
  hungarian_algorithm::Solver<Cost, hungarian_algorithm::MinimizeCosts<Cost>, hungarian_algorithm::BruteForceMethod>
    solver(invalid_cost);
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
    std::cerr << total_cost << " != " << total_cost_brute_force << ":" << std::endl;
    printCostMatrix(std::cerr, cost_matrix);
    std::cerr << "Hungarian ";
    printAssignment(std::cerr, assignment, cost_matrix);
    std::cerr << "brute force ";
    printAssignment(std::cerr, assignment_brute_force, cost_matrix);
  }
  assert(total_cost == total_cost_brute_force);
}

void test_solver()
{
  const double costs[] = {10, 19, 8, 15, 0,
                          10, 18, 7, 17, 0,
                          13, 16, 9, 14, 0,
                          12, 19, 8, 18, 0};
  const hungarian_algorithm::Matrix<double> cost_matrix(4, 5, std::vector<double>(&costs[0], &costs[20]));
  assertSolversReturnEqualCost(cost_matrix);
}

void test_solver_random_square(const std::size_t num_rows, const std::size_t num_runs)
{
  hungarian_algorithm::Matrix<int> cost_matrix(num_rows, num_rows, 0);
  const int max_cost = std::min(1000, RAND_MAX / static_cast<int>(num_rows));  // To prevent overflows when summing
  for (std::size_t i = 0; i < num_runs; ++i)
  {
    for (std::size_t row = 0; row < cost_matrix.numRows(); ++row)
    {
      for (std::size_t col = 0; col < cost_matrix.numCols(); ++col)
      {
        cost_matrix(row, col) = std::rand() % max_cost;
      }
    }
    assertSolversReturnEqualCost(cost_matrix);
  }
}

int main()
{
  test_matrix();
  test_solver_construction();
  test_solver_1x1();
//  test_solver();
  test_solver_random_square(5, 20);
  return 0;
}
