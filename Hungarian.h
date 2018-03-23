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

namespace hungarian_algorithm
{
// A simple helper class that saves a row-major ordered matrix in a vector.
template<class T>
class Matrix
{
public:
  Matrix()
    : data_(), num_rows_(0), num_cols_(0)
  {}

  Matrix(std::size_t num_rows, std::size_t num_cols, const T& initial_value)
    : data_(num_rows * num_cols, initial_value), num_rows_(num_rows), num_cols_(num_cols)
  {}

  Matrix(std::size_t num_rows, std::size_t num_cols, const std::vector<T>& initial_value)
    : data_(initial_value), num_rows_(num_rows), num_cols_(num_cols)
  {
    assert(data_.size() == (num_rows_ * num_cols_));
  }

  inline T& at(std::size_t row, std::size_t col)
  {
    return data_.at(row * num_cols_ + col);
  }

  inline T& operator()(std::size_t row, std::size_t col)
  {
    return data_[row * num_cols_ + col];
  }

  inline const T& at(std::size_t row, std::size_t col) const
  {
    return data_.at(row * num_cols_ + col);
  }

  inline const T& operator()(std::size_t row, std::size_t col) const
  {
    return data_[row * num_cols_ + col];
  }

  inline std::size_t getNumRows() const
  {
    return num_rows_;
  }

  inline std::size_t getNumCols() const
  {
    return num_cols_;
  }

protected:
  std::vector<T> data_;
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

template<typename Cost, typename Size = std::size_t>
class Solver
{
public:
  typedef Cost Cost;
  typedef Size Size;

  explicit Solver(const Cost invalid_cost, const Cost epsilon_cost = std::numeric_limits<Cost>::epsilon())
    : invalid_cost_(invalid_cost), epsilon_cost_(epsilon_cost)
  {
  }

  explicit Solver(const Cost epsilon_cost = std::numeric_limits<Cost>::epsilon())
    : invalid_cost_(std::numeric_limits<Cost>::max()), epsilon_cost_(epsilon_cost)
  {
  }

  template<typename NormalizationStrategy = MinimizeCosts, typename CostFunction>
  void solve(const CostFunction& cost_function, const Size num_rows, const Size num_cols)
  {
    assert(num_rows > Size(0));
    assert(num_cols > Size(0));

    copyAndNormalizeCosts<NormalizationStrategy, CostFunction>(cost_function, num_rows, num_cols);

    covered_rows_.assign(cost_matrix_.getNumRows(), false);
    covered_cols_.assign(cost_matrix_.getNumCols(), false);
    star_matrix_ = Matrix<bool>(cost_matrix_.getNumRows(), cost_matrix_.getNumCols(), false);
    prime_matrix_ = Matrix<bool>(cost_matrix_.getNumRows(), cost_matrix_.getNumCols(), false);

    // Steps 1 and 2a:
    for (std::size_t row = 0; row < cost_matrix_.getNumRows(); ++row)
    {
      for (std::size_t col = 0; col < cost_matrix_.getNumCols(); ++col)
      {
        if (cost_matrix_(row, col) <= epsilon_cost_ && !covered_cols_[col])
        {
          star_matrix_(row, col) = true;
          covered_cols_[col] = true;
          break;
        }
      }
    }

    if (!step2b())
    {
      /* algorithm not finished */
      /* move to step 3 */
      step3();
    }

    /* algorithm finished */
    buildassignmentvector(assignment, star_matrix);

    // Compute cost and remove invalid assignments:
    computeassignmentcost(assignment, total_cost, costs_, num_rows_);
  }

protected:
  template<typename NormalizationStrategy, typename CostFunction>
  void copyAndNormalizeCosts(const CostFunction& cost_function, const Size num_rows, const Size num_cols)
  {
    // Make cost matrix square, filling extraneous elements with invalid cost:
    const std::size_t max_size(static_cast<std::size_t>(std::max(num_rows, num_cols)));
    cost_matrix_ = Matrix<Cost>(max_size, max_size, invalid_cost_);

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
        best_cost = NormalizationStrategy::getBetterCost(best_cost, cost);
      }

      for (col = Size(0), s_col = 0; col < num_cols; ++col, ++s_col)
      {
        cost_matrix_(s_row, s_col) = NormalizationStrategy::normalizeCost(cost_matrix_(s_row, s_col), best_cost);
      }
    }
  }

  bool step2b()
  {
    // Count covered columns:
    const std::vector<bool>::difference_type num_covered_cols
      = std::count(covered_cols_.begin(), covered_cols_.end(), true);

    return num_covered_cols == covered_cols_.size();
  }

  void buildassignmentvector(int* assignment, bool* starMatrix);
  void computeassignmentcost(int* assignment, double* cost, double* distMatrix, int nOfRows);
  void step2a(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
              bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
  void step3(int* assignment, double* distMatrix, bool* starMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
  void step4(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
  void step5(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);

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
