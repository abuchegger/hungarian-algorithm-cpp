#include <cassert>
#include <cstdlib>
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
  hungarian_algorithm::Solver<int> munkres;

  std::vector<std::size_t> assignment;
  const int total_cost = munkres.solve(cost_matrix, assignment);
  std::cout << std::endl << "total cost: " << total_cost << std::endl;
}

/**
 * Computes minimal cost by iterating over all possible assignments (combinations).
 */
template<typename Cost>
Cost computeMinimalCostBruteForce(const hungarian_algorithm::Matrix<Cost> cost_matrix)
{
  std::vector<std::size_t> assignment(cost_matrix.numRows(), 0);
  Cost min_cost = std::numeric_limits<Cost>::max();
  std::size_t row = 0;
  while (true)
  {
    bool valid = true;
    for (std::size_t r = 0; r < row; ++r)
    {
      if (assignment[r] == assignment[row])
      {
        valid = false;
        break;
      }
    }
    if (!valid)
    {
      ++assignment[row];
    }
    else if (assignment[row] < cost_matrix.numCols())
    {
      if (row < (cost_matrix.numRows() - 1))
      {
        ++row;
        assignment[row] = 0;
      }
      else
      {
        Cost cost = 0;
        for (std::size_t r = 0; r < cost_matrix.numRows(); ++r)
        {
          cost += cost_matrix(r, assignment[r]);
        }
        min_cost = std::min(min_cost, cost);
        ++assignment[row];
      }
    }
    else if (row > 0)
    {
      --row;
      ++assignment[row];
    }
    else
    {
      return min_cost;
    }
  }
}

void test_solver()
{
  const double costs[] = {10, 19, 8, 15, 0,
                          10, 18, 7, 17, 0,
                          13, 16, 9, 14, 0,
                          12, 19, 8, 18, 0};
  const hungarian_algorithm::Matrix<double> cost_matrix(4, 5, std::vector<double>(&costs[0], &costs[20]));

  hungarian_algorithm::Solver<double> munkres;
  std::vector<std::size_t> assignment;
  const double total_cost = munkres.solve(cost_matrix, assignment);
  assert(total_cost == computeMinimalCostBruteForce(cost_matrix));

  for (std::size_t i = 0; i < assignment.size(); ++i)
  {
    std::cout << i << " -> " << assignment[i] << " (cost = " << cost_matrix(i, assignment[i]) << ")" << std::endl;
  }

  std::cout << std::endl << "total cost: " << total_cost << std::endl;
}

void test_solver_random_square(const std::size_t num_rows, const std::size_t num_runs)
{
  hungarian_algorithm::Matrix<int> cost_matrix(num_rows, num_rows, 0);
  std::vector<std::size_t> assignment;
  const int max_cost = RAND_MAX / static_cast<int>(num_rows);  // To prevent overflows when summing
  for (std::size_t i = 0; i < num_runs; ++i)
  {
    for (std::size_t row = 0; row < cost_matrix.numRows(); ++row)
    {
      for (std::size_t col = 0; col < cost_matrix.numCols(); ++col)
      {
        cost_matrix(row, col) = std::rand() % max_cost;
      }
    }

    hungarian_algorithm::Solver<int> munkres;
    const int total_cost = munkres.solve(cost_matrix, assignment);
    const int total_cost_brute_force = computeMinimalCostBruteForce(cost_matrix);
    if (total_cost != total_cost_brute_force)
    {
      std::cerr << "total_cost != total_cost_brute_force:" << std::endl;
      std::cerr << total_cost << " != " << total_cost_brute_force << ":" << std::endl;
      std::cerr << "cost_matrix:" << std::endl;
      for (std::size_t row = 0; row < cost_matrix.numRows(); ++row)
      {
        std::cerr << "  ";
        for (std::size_t col = 0; col < cost_matrix.numCols(); ++col)
        {
          std::cerr << (col != 0 ? ", " : "") << cost_matrix(row, col);
        }
        std::cerr << std::endl;
      }
    }
    assert(total_cost == total_cost_brute_force);
  }
}

int main()
{
  test_matrix();
  test_solver_construction();
  test_solver_1x1();
  test_solver();
  test_solver_random_square(5, 20);
  return 0;
}
