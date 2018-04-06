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
#ifndef HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_ALGORITHM_H
#define HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_ALGORITHM_H

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hungarian_algorithm
{
template<typename Size, typename AssignmentMap>
struct AssignmentMapAdapter
{
  AssignmentMapAdapter(AssignmentMap& assignment_map, const Size /*num_rows*/, const Size /*num_cols*/)
    : assignment_map_(assignment_map)
  {
  }

  void insert(const Size row, const Size col) const
  {
    assignment_map_[row] = col;
  }

  AssignmentMap& assignment_map_;
};

template<typename Size>
struct AssignmentMapAdapter<Size, std::vector<Size> >
{
  AssignmentMapAdapter(std::vector<Size>& assignment_map, const Size num_rows, const Size num_cols)
    : assignment_map_(assignment_map)
  {
    assignment_map_.assign(static_cast<std::size_t>(num_rows), num_cols);
  }

  void insert(const Size row, const Size col) const
  {
    assignment_map_[row] = col;
  }

  std::vector<Size>& assignment_map_;
};

/**
 * Base class for HungarianSolver that contains functionality independent from template arguments.
 */
class HungarianSolverBase
{
protected:
  HungarianSolverBase(const std::size_t num_rows, const std::size_t num_cols);

  inline std::size_t getIndex(const std::size_t row, const std::size_t col) const
  {
    return row * num_cols_ + col;
  }

  std::size_t findInCol(const std::vector<bool>& matrix, const std::size_t col) const;
  std::size_t findInRow(const std::vector<bool>& matrix, const std::size_t row) const;
  void coverStarredColumns();
  bool areAllColumnsCovered() const;
  bool findAssignment();
  void permute(const std::size_t row, const std::size_t col);

  const std::size_t num_rows_;
  const std::size_t num_cols_;
  std::vector<bool> covered_rows_;
  std::vector<bool> covered_cols_;
  std::vector<bool> zero_matrix_;
  std::vector<bool> star_matrix_;
  std::vector<bool> prime_matrix_;
};

/**
 * Computes optimal assignment by iteratively looking for minimal costs and trying out permutations of the assignment.
 * Also known as Munkres or Kuhn-Munkres algorithm.
 */
template<typename Cost, typename Size, typename CostComparator>
class HungarianSolver : public HungarianSolverBase
{
public:
  HungarianSolver(const Size num_rows, const Size num_cols, const CostComparator& cost_comparator,
                  const Cost& zero_cost)
    : HungarianSolverBase(static_cast<std::size_t>(num_rows), static_cast<std::size_t>(num_cols)),
      cost_comparator_(cost_comparator), zero_cost_(zero_cost), cost_matrix_(num_rows_ * num_cols_, zero_cost)
  {
    if (num_rows < 0 || num_cols < 0)
    {
      throw std::invalid_argument("Hungarian algorithm: num_rows and num_cols must be non-negative");
    }
  }

  template<typename CostFunction>
  void solve(const CostFunction& cost_function)
  {
    if (num_rows_ != 0 && num_cols_ != 0)
    {
      copyAndNormalizeCosts<CostFunction>(cost_function);
      while (!areAllColumnsCovered())
      {
        while (!findAssignment())
        {
          // No zeros found => generate more zeros by subtracting smallest remaining element:
          subtractSmallestUncoveredElement();
        }
      }
    }
  }

  template<typename CostFunction, typename AssignmentMap>
  Cost getAssignment(const CostFunction& cost_function, AssignmentMap& assignment_map)
  {
    // Fill assignment map and compute total cost of assignment:
    AssignmentMapAdapter<Size, AssignmentMap> assignment_map_adapter(assignment_map, num_rows_, num_cols_);
    Cost total_cost(zero_cost_);
    for (std::size_t row = 0; row < num_rows_; ++row)
    {
      const std::size_t col(findInRow(star_matrix_, row));
      if (col < num_cols_)
      {
        assignment_map_adapter.insert(Size(row), Size(col));
        total_cost += cost_function(Size(row), Size(col));
      }
    }
    return total_cost;
  }

protected:
  template<typename CostFunction>
  void copyAndNormalizeCosts(const CostFunction& cost_function)
  {
    if (num_rows_ <= num_cols_)
    {
      // Copy costs from cost function and find best cost per row:
      std::size_t i = 0;
      for (std::size_t row = 0; row < num_rows_; ++row)
      {
        Cost best_cost(zero_cost_);
        for (std::size_t col = 0; col < num_cols_; ++col, ++i)
        {
          const Cost cost(cost_function(Size(row), Size(col)));
          cost_matrix_[i] = cost;
          if (col == 0 || cost_comparator_(cost, best_cost))
          {
            best_cost = cost;
          }
        }

        i -= num_cols_; // Reset index to beginning of row

        // Subtract best cost in row from each element (or vice-versa, depending on whether costs are maximized or
        // minimized) and star first zero found in row if column is not already covered:
        for (std::size_t col = 0; col < num_cols_; ++col, ++i)
        {
          Cost& cost = cost_matrix_[i];
          cost = std::max(cost, best_cost) - std::min(cost, best_cost);
          if (cost == zero_cost_)
          {
            zero_matrix_[i] = true;
            if (!covered_rows_[row] && !covered_cols_[col])
            {
              star_matrix_[i] = true;
              covered_rows_[row] = true;
              covered_cols_[col] = true;
            }
          }
        }
      }
    }
    else
    {
      // Copy costs from cost function and find best cost per column:
      for (std::size_t col = 0; col < num_cols_; ++col)
      {
        Cost best_cost(zero_cost_);
        for (std::size_t row = 0, i = col; row < num_rows_; ++row, i += num_cols_)
        {
          const Cost cost(cost_function(Size(row), Size(col)));
          cost_matrix_[i] = cost;
          if (row == 0 || cost_comparator_(cost, best_cost))
          {
            best_cost = cost;
          }
        }

        // Subtract best cost in column from each element (or vice-versa, depending on whether costs are maximized or
        // minimized) and star first zero found in column if column is not already covered:
        for (std::size_t row = 0, i = col; row < num_rows_; ++row, i += num_cols_)
        {
          Cost& cost = cost_matrix_[i];
          cost = std::max(cost, best_cost) - std::min(cost, best_cost);
          if (cost == zero_cost_)
          {
            zero_matrix_[i] = true;
            if (!covered_rows_[row] && !covered_cols_[col])
            {
              star_matrix_[i] = true;
              covered_rows_[row] = true;
              covered_cols_[col] = true;
            }
          }
        }
      }
    }
    covered_rows_.assign(num_rows_, false);
  }

  void subtractSmallestUncoveredElement()
  {
    // Find smallest uncovered element:
    bool found_minimum = false;
    Cost h = zero_cost_;
    for (std::size_t row = 0; row < num_rows_; ++row)
    {
      if (!covered_rows_[row])
      {
        for (std::size_t col = 0; col < num_cols_; ++col)
        {
          if (!covered_cols_[col])
          {
            if (!found_minimum)
            {
              h = cost_matrix_[getIndex(row, col)];
              found_minimum = true;
            }
            else
            {
              h = std::min(h, cost_matrix_[getIndex(row, col)]);
            }
          }
        }
      }
    }

    if (!found_minimum)
    {
      throw std::logic_error("Hungarian algorithm: !found_minimum; this should never happen");
    }

    // Subtract h from each uncovered element, and add h to each doubly covered element:
    std::size_t i = 0;
    for (std::size_t row = 0; row < num_rows_; ++row)
    {
      for (std::size_t col = 0; col < num_cols_; ++col, ++i)
      {
        if (covered_rows_[row] && covered_cols_[col])
        {
          cost_matrix_[i] += h;
          zero_matrix_[i] = (cost_matrix_[i] == zero_cost_);
        }
        else if (!covered_rows_[row] && !covered_cols_[col])
        {
          cost_matrix_[i] -= h;
          zero_matrix_[i] = (cost_matrix_[i] == zero_cost_);
        }
      }
    }
  }

  const CostComparator& cost_comparator_;
  const Cost zero_cost_;
  std::vector<Cost> cost_matrix_;
};

/**
 * Solves an assignment problem.
 *
 * @tparam NormalizationStrategy strategy for bringing the costs returned by the cost function into the setup required
 *                               for the algorithm. Should be either MinimizeCosts or MaximizeCosts.
 * @tparam CostFunction type of the cost_function parameter.
 * @tparam AssignmentMap type of the assignment parameter.
 * @param cost_function anything that takes a row and a column index and returns a cost.
 *                      Used like Cost c = cost_function(row, col). In particular, the Matrix utility class, and the
 *                      matrix classes of several libraries like Eigen or OpenCV provide this interface.
 * @param num_rows number of rows (e.g., workers) in the assignment problem.
 * @param num_cols number of columns (e.g. jobs) in the assignment problem.
 * @param assignment_map anything that can be indexed and then assigned an index. Used like assignment[row] = col. In
 *                       particular, std::vector and std::map provide this interface. Must accept a row index in the
 *                       range 0 ... num_rows - 1. Only valid assignments are set; other entries are left unchanged
 *                       (and should thus probably be initialized to something invalid, like SIZE_MAX).
 * @return the cost of the optimal assignment, as a pair consisting of the number of invalid assignments, and the sum
 *         of the costs of the valid assignments.
 */
template<typename Cost, typename Size, typename CostFunction, typename AssignmentMap, typename CostComparator>
Cost solve(const CostFunction& cost_function, const Size num_rows, const Size num_cols, AssignmentMap& assignment_map,
           const CostComparator& cost_comparator, const Cost& zero_cost)
{
  HungarianSolver<Cost, Size, CostComparator> solver(num_rows, num_cols, cost_comparator, zero_cost);
  solver.solve(cost_function);
  return solver.getAssignment(cost_function, assignment_map);
}

template<typename Cost, typename Size, typename CostFunction, typename AssignmentMap, typename CostComparator>
Cost solve(const CostFunction& cost_function, const Size num_rows, const Size num_cols, AssignmentMap& assignment_map,
           const CostComparator& cost_comparator)
{
  HungarianSolver<Cost, Size, CostComparator> solver(num_rows, num_cols, cost_comparator, Cost());
  solver.solve(cost_function);
  return solver.getAssignment(cost_function, assignment_map);
}

template<typename Cost, typename Size, typename CostFunction, typename AssignmentMap>
Cost solve(const CostFunction& cost_function, const Size num_rows, const Size num_cols, AssignmentMap& assignment_map,
           const Cost& zero_cost)
{
  return solve<Cost, Size, CostFunction, AssignmentMap>(
    cost_function, num_rows, num_cols, assignment_map, std::less<Cost>(), zero_cost);
}

template<typename Cost, typename Size, typename CostFunction, typename AssignmentMap>
Cost solve(const CostFunction& cost_function, const Size num_rows, const Size num_cols, AssignmentMap& assignment_map)
{
  return solve<Cost, Size, CostFunction, AssignmentMap>(
    cost_function, num_rows, num_cols, assignment_map, std::less<Cost>(), Cost());
}

/**
 * Finds optimal assignment by iterating over all possible combinations.
 */
template<typename Cost, typename CostFunction, typename CostComparator, typename Size>
class BruteForceSolver
{
public:
  BruteForceSolver(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                   const CostComparator& cost_comparator, const Cost& zero_cost)
    : cost_function_(cost_function), cost_comparator_(cost_comparator), zero_cost_(zero_cost),
      num_rows_(static_cast<std::size_t>(num_rows)), num_cols_(static_cast<std::size_t>(num_cols)),
      current_assignment_(num_rows_, num_cols_), optimal_assignment_(num_rows_, num_cols_),
      covered_cols_(num_cols_, false), optimal_cost_(num_rows_, zero_cost)
  {
  }

  /**
   * Solves an assignment problem.
   *
   * @tparam NormalizationStrategy strategy for bringing the costs returned by the cost function into the setup required
   *                               for the algorithm. Should be either MinimizeCosts or MaximizeCosts.
   * @tparam CostFunction type of the cost_function parameter.
   * @tparam AssignmentMap type of the assignment parameter.
   * @param cost_function anything that takes a row and a column index and returns a cost.
   *                      Used like Cost c = cost_function(row, col). In particular, the Matrix utility class, and the
   *                      matrix classes of several libraries like Eigen or OpenCV provide this interface.
   * @param num_rows number of rows (e.g., workers) in the assignment problem.
   * @param num_cols number of columns (e.g. jobs) in the assignment problem.
   * @param assignment_map anything that can be indexed and then assigned an index. Used like assignment[row] = col. In
   *                       particular, std::vector and std::map provide this interface. Must accept a row index in the
   *                       range 0 ... num_rows - 1. Only valid assignments are set; other entries are left unchanged
   *                       (and should thus probably be initialized to something invalid, like SIZE_MAX).
   * @return the cost of the optimal assignment (the sum of the costs of the valid assignments).
   */
  void solve()
  {
    doSolve(0, CombinedCost(0, zero_cost_));
  }

  template<typename AssignmentMap>
  Cost getAssignment(AssignmentMap& assignment_map)
  {
    // Fill assignment map and compute total cost of assignment:
    AssignmentMapAdapter<Size, AssignmentMap> assignment_map_adapter(assignment_map, Size(num_rows_), Size(num_cols_));
    for (std::size_t row = 0; row < num_rows_; ++row)
    {
      const std::size_t col = optimal_assignment_[row];
      if (col < num_cols_)
      {
        assignment_map_adapter.insert(Size(row), Size(col));
      }
    }
    return optimal_cost_.second;
  }

protected:
  // We count rows containing invalid assignments separately from the sum of valid assignments to mitigate some
  // overflow / loss of significance problems found during testing, affecting both integral and floating-point types.
  // We thus use a tuple of (Number of invalid assignments, sum of valid assignments) for keeping track of cost.
  typedef std::pair<std::size_t, Cost> CombinedCost;

  bool isBetterThan(const CombinedCost& a, const CombinedCost& b) const
  {
    return a.first < b.first || (a.first == b.first && cost_comparator_(a.second, b.second));
  }

  void doSolve(const std::size_t row, const CombinedCost& accumulated_cost)
  {
    if (row < num_rows_)
    {
      for (std::size_t col = 0; col < num_cols_; ++col)
      {
        if (!covered_cols_[col])
        {
          const CombinedCost new_accumulated_cost(accumulated_cost.first,
                                                  accumulated_cost.second + cost_function_(Size(row), Size(col)));
          current_assignment_[row] = col;
          covered_cols_[col] = true;
          doSolve(row + 1, new_accumulated_cost);
          covered_cols_[col] = false;
        }
      }

      if (num_cols_ < num_rows_)
      {
        // One possibility is also an invalid assignment:
        const CombinedCost new_accumulated_cost(accumulated_cost.first + 1, accumulated_cost.second);
        current_assignment_[static_cast<std::size_t>(row)] = num_cols_;
        doSolve(row + 1, new_accumulated_cost);
      }
    }
    else if (isBetterThan(accumulated_cost, optimal_cost_))
    {
      optimal_assignment_ = current_assignment_;
      optimal_cost_ = accumulated_cost;
    }
  }

  const CostFunction& cost_function_;
  const CostComparator& cost_comparator_;
  const Cost zero_cost_;
  const std::size_t num_rows_;
  const std::size_t num_cols_;
  std::vector<std::size_t> current_assignment_;
  std::vector<std::size_t> optimal_assignment_;
  std::vector<bool> covered_cols_;
  CombinedCost optimal_cost_;
};

/**
 * Finds optimal assignment by iterating over all possible combinations.
 */
template<typename Cost, typename CostFunction, typename Size, typename AssignmentMap, typename CostComparator>
Cost solveBruteForce(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                     AssignmentMap& assignment_map, const CostComparator& cost_comparator, const Cost& zero_cost)
{
  BruteForceSolver<Cost, CostFunction, CostComparator, Size> solver(
    cost_function, num_rows, num_cols, cost_comparator, zero_cost);
  solver.solve();
  return solver.getAssignment(assignment_map);
}

template<typename Cost, typename CostFunction, typename Size, typename AssignmentMap, typename CostComparator>
Cost solveBruteForce(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                     AssignmentMap& assignment_map, const CostComparator& cost_comparator)
{
  return solveBruteForce<Cost, CostFunction, Size, AssignmentMap>(
    cost_function, num_rows, num_cols, assignment_map, cost_comparator, Cost());
}

/**
 * Finds optimal assignment by iterating over all possible combinations.
 */
template<typename Cost, typename CostFunction, typename Size, typename AssignmentMap>
Cost solveBruteForce(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                     AssignmentMap& assignment_map, const Cost& zero_cost)
{
  return solveBruteForce<Cost, CostFunction, Size, AssignmentMap>(
    cost_function, num_rows, num_cols, assignment_map, std::less<Cost>(), zero_cost);
}

/**
 * Finds optimal assignment by iterating over all possible combinations.
 */
template<typename Cost, typename CostFunction, typename Size, typename AssignmentMap>
Cost solveBruteForce(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                     AssignmentMap& assignment_map)
{
  return solveBruteForce<Cost, CostFunction, Size, AssignmentMap>(
    cost_function, num_rows, num_cols, assignment_map, std::less<Cost>());
}
}

#endif // HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_ALGORITHM_H
