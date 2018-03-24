///////////////////////////////////////////////////////////////////////////////
// Hungarian.h: Header file for Class HungarianAlgorithm.
//
// This is a C++ wrapper with slight modification of a hungarian algorithm implementation by Markus Buehren.
// The original implementation is a few mex-functions for use in MATLAB, found here:
// http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem
//
// Both this code and the orignal code are published under the BSD license.
// by Cong Ma, 2016
//

#ifndef HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_H
#define HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_H

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>
#include <deque>
#include <queue>

namespace hungarian_algorithm
{
// A simple helper class that saves a row-major ordered matrix in a vector.
template<class T>
class Matrix
{
public:
  // Declare a few types like the STL containers:
  typedef std::vector<T> container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_reference const_reference;

  Matrix()
    : data_(), num_rows_(0), num_cols_(0)
  {}

  Matrix(const std::size_t num_rows, const std::size_t num_cols, const value_type& initial_value)
    : data_(num_rows * num_cols, initial_value), num_rows_(num_rows), num_cols_(num_cols)
  {}

  Matrix(const std::size_t num_rows, const std::size_t num_cols, const container_type& initial_value)
    : data_(initial_value), num_rows_(num_rows), num_cols_(num_cols)
  {
    assert(data_.size() == (num_rows_ * num_cols_));
  }

  void assign(const std::size_t num_rows, const std::size_t num_cols, const value_type& value)
  {
    data_.assign(num_rows * num_cols, value);
    num_rows_ = num_rows;
    num_cols_ = num_cols;
  }

  void assign(const value_type& value)
  {
    data_.assign(data_.size(), value);
  }

  inline reference operator()(const std::size_t row, const std::size_t col)
  {
    return data_[row * num_cols_ + col];
  }

  inline const_reference operator()(const std::size_t row, const std::size_t col) const
  {
    return data_[row * num_cols_ + col];
  }

  inline std::size_t numRows() const
  {
    return num_rows_;
  }

  inline std::size_t numCols() const
  {
    return num_cols_;
  }

  std::size_t findInRow(const std::size_t row, const value_type& value) const
  {
    const typename container_type::const_iterator row_begin = data_.begin() + row * num_cols_;
    return static_cast<std::size_t>(std::distance(row_begin, std::find(row_begin, row_begin + num_cols_, value)));
  }

  std::size_t findInCol(const std::size_t col, const value_type& value) const
  {
    std::size_t row = 0;
    for (; row < num_rows_; ++row)
    {
      if ((*this)(row, col) == value)
      {
        break;
      }
    }
    return row;
  }

protected:
  container_type data_;
  std::size_t num_rows_;
  std::size_t num_cols_;
};

template<typename Cost>
struct CostNormalizationStrategy
{
  static Cost getBetterCost(const Cost& a, const Cost& b);
  static Cost normalizeCost(const Cost& cost, const Cost& best_cost);
};

template<typename Cost>
struct MinimizeCosts : public CostNormalizationStrategy<Cost>
{
  static Cost getBetterCost(const Cost& a, const Cost& b)
  {
    return std::min(a, b);
  }

  static Cost normalizeCost(const Cost& cost, const Cost& best_cost)
  {
    return cost - best_cost;
  }
};

template<typename Cost>
struct MaximizeCosts : public CostNormalizationStrategy<Cost>
{
  static Cost getBetterCost(const Cost& a, const Cost& b)
  {
    return std::max(a, b);
  }

  static Cost normalizeCost(const Cost& cost, const Cost& best_cost)
  {
    return best_cost - cost;
  }
};

template <typename Size, typename AssignmentMap>
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

template <typename Size>
struct AssignmentMapAdapter<Size, std::vector<Size> >
{
  AssignmentMapAdapter(std::vector<Size>& assignment_map, const Size num_rows, const Size /*num_cols*/)
    : assignment_map_(assignment_map)
  {
    assignment_map_.assign(static_cast<std::size_t>(num_rows), std::numeric_limits<Size>::max());
  }

  void insert(const Size row, const Size col) const
  {
    assignment_map_[row] = col;
  }

  std::vector<Size>& assignment_map_;
};

template<typename Cost, typename CostNormalizationStrategy = MinimizeCosts<Cost> >
class Solver
{
public:
  typedef Cost CostType;

  explicit Solver(const Cost invalid_cost, const Cost epsilon_cost = std::numeric_limits<Cost>::epsilon())
    : invalid_cost_(invalid_cost), epsilon_cost_(epsilon_cost)
  {
  }

  explicit Solver(const Cost epsilon_cost = std::numeric_limits<Cost>::epsilon())
    : invalid_cost_(std::numeric_limits<Cost>::max()), epsilon_cost_(epsilon_cost)
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
   * @return the cost of the optimal assignment.
   */
  template<typename CostFunction, typename Size, typename AssignmentMap>
  Cost solve(const CostFunction& cost_function, const Size num_rows, const Size num_cols, AssignmentMap& assignment_map)
  {
    assert(num_rows > Size(0));
    assert(num_cols > Size(0));

    copyAndNormalizeCosts<CostFunction, Size>(cost_function, num_rows, num_cols);

    covered_rows_.assign(cost_matrix_.numRows(), false);
    covered_cols_.assign(cost_matrix_.numCols(), false);
    star_matrix_.assign(cost_matrix_.numRows(), cost_matrix_.numCols(), false);
    prime_matrix_.assign(cost_matrix_.numRows(), cost_matrix_.numCols(), false);

    // Steps 1 and 2a:
    for (std::size_t row = 0; row < cost_matrix_.numRows(); ++row)
    {
      for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
      {
        if (!covered_cols_[col] && cost_matrix_(row, col) <= epsilon_cost_)
        {
          star_matrix_(row, col) = true;
          covered_cols_[col] = true;
          break;
        }
      }
    }

    if (!areAllColumnsCovered())
    {
      /* algorithm not finished */
      /* move to step 3 */
      while (true)
      {
        while (!step3())
        {
        }
        step5();
        if (step3())
        {
          break;
        }
      }
    }

    return fillAssignmentMap<CostFunction, Size, AssignmentMap>(
      cost_function, num_rows, num_cols, AssignmentMapAdapter<Size, AssignmentMap>(assignment_map, num_rows, num_cols));
  }

  template <typename AssignmentMap>
  Cost solve(const Matrix<Cost>& cost_matrix, AssignmentMap& assignment_map)
  {
    return solve(cost_matrix, cost_matrix.numRows(), cost_matrix.numCols(), assignment_map);
  }

protected:
  template<typename CostFunction, typename Size>
  void copyAndNormalizeCosts(const CostFunction& cost_function, const Size num_rows, const Size num_cols)
  {
    // Make cost matrix square, filling extraneous elements with invalid cost:
    const std::size_t max_size(static_cast<std::size_t>(std::max(num_rows, num_cols)));
    cost_matrix_.assign(max_size, max_size, invalid_cost_);

    // Copy costs, making sure optimal element in each row is zero, and every other one is larger; note that this does
    // not touch the extraneous elements:
    std::size_t s_row = 0;
    for (Size row(0); row < num_rows; ++row, ++s_row)
    {
      std::size_t s_col = 0;
      Size col(0);
      Cost best_cost = cost_matrix_(s_row, s_col) = cost_function(row, col);
      for (++col, ++s_col; col < num_cols; ++col, ++s_col)
      {
        const Cost cost(cost_function(row, col));
        cost_matrix_(s_row, s_col) = cost;
        best_cost = CostNormalizationStrategy::getBetterCost(best_cost, cost);
      }

      for (col = Size(0), s_col = 0; col < num_cols; ++col, ++s_col)
      {
        cost_matrix_(s_row, s_col) = CostNormalizationStrategy::normalizeCost(cost_matrix_(s_row, s_col), best_cost);
      }
    }
  }

  template<typename CostFunction, typename Size, typename AssignmentMap>
  Cost fillAssignmentMap(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                         const AssignmentMapAdapter<Size, AssignmentMap>& assignment_map)
  {
    // Fill assignment vector and compute total cost of assignment:
    Cost total_cost(0);
    std::size_t s_row = 0;
    for (Size row(0); row < num_rows; ++row, ++s_row)
    {
      std::size_t s_col = 0;
      for (Size col(0); col < num_cols; ++col, ++s_col)
      {
        if (star_matrix_(s_row, s_col))
        {
          assignment_map.insert(row, col);
          total_cost += cost_function(row, col);
          break;
        }
      }
    }
    return total_cost;
  }

  void coverStarredColumns()
  {
    // Cover every column containing a starred zero:
    for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
    {
      if (star_matrix_.findInCol(col, true) < cost_matrix_.numRows())
      {
        covered_cols_[col] = true;
      }
    }
  }

  bool areAllColumnsCovered()
  {
    return static_cast<std::size_t>(std::count(covered_cols_.begin(), covered_cols_.end(), true))
      == covered_cols_.size();
  }

  bool step3()
  {
    bool zeros_found = true;
    while (zeros_found)
    {
      // Find zeros in uncovered cells ("prime" zeros):
      zeros_found = false;
      for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
      {
        if (!covered_cols_[col])
        {
          for (std::size_t row = 0; row < cost_matrix_.numRows(); ++row)
          {
            if (!covered_rows_[row] && cost_matrix_(row, col) <= epsilon_cost_)
            {
              // Found a prime zero:
              prime_matrix_(row, col) = true;

              // Find starred zero in current row:
              const std::size_t star_col = star_matrix_.findInRow(row, true);
              if (star_col < cost_matrix_.numCols())
              {
                // Found a starred zero in current row => cover row, uncover column, and proceed with next column:
                covered_rows_[row] = true;
                covered_cols_[star_col] = false;
                zeros_found = true;
                break;
              }

              // No starred zero found => move to step 4:
              step4(row, col);
              coverStarredColumns();
              return areAllColumnsCovered(); // Continue searching if not all columns are starred
            }
          }
        }
      }
    }
    return true;
  }

  void step4(const std::size_t row, const std::size_t col)
  {
    // Generate temporary copy of star matrix:
    Matrix<bool> new_star_matrix(star_matrix_);

    // Star current zero:
    new_star_matrix(row, col) = true;

    // Find starred zero in current column:
    std::size_t star_row = star_matrix_.findInCol(col, true);
    while (star_row < cost_matrix_.numRows())
    {
      // Unstar the starred zero:
      new_star_matrix(star_row, col) = false;  // FIXME: is col really right in subsequent iterations?

      // Find primed zero in current row:
      const std::size_t prime_col = prime_matrix_.findInRow(star_row, true);

      // Star the primed zero:
      // FIXME: can't prime_col be >= numCols here??
      new_star_matrix(star_row, prime_col) = true;

      // Find starred zero in current column:
      star_row = star_matrix_.findInCol(prime_col, true);
    }

    // Use temporary copy as new star matrix:
    star_matrix_ = new_star_matrix;

    // Delete all primes, uncover all rows:
    prime_matrix_.assign(false);
    covered_rows_.assign(covered_rows_.size(), false);
  }

  void step5()
  {
    // Find smallest uncovered element:
    Cost h = invalid_cost_;
    for (std::size_t row = 0; row < cost_matrix_.numRows(); ++row)
    {
      if (!covered_rows_[row])
      {
        for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
        {
          if (!covered_cols_[col])
          {
            h = std::min(h, cost_matrix_(row, col));
          }
        }
      }
    }

    // Add h to each covered row:
    for (std::size_t row = 0; row < cost_matrix_.numRows(); ++row)
    {
      if (covered_rows_[row])
      {
        for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
        {
          cost_matrix_(row, col) += h;
        }
      }
    }

    // Subtract h from each uncovered column:
    for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
    {
      if (!covered_cols_[col])
      {
        for (std::size_t row = 0; row < cost_matrix_.numRows(); ++row)
        {
          cost_matrix_(row, col) -= h;
        }
      }
    }
  }

  const Cost invalid_cost_;
  const Cost epsilon_cost_;
  Matrix<Cost> cost_matrix_;
  std::vector<bool> covered_rows_;
  std::vector<bool> covered_cols_;
  Matrix<bool> star_matrix_;
  Matrix<bool> prime_matrix_;
};
}

#endif // HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_H
