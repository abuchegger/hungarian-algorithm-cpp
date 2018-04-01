///////////////////////////////////////////////////////////////////////////////
// Hungarian.cpp: Implementation file for Class HungarianAlgorithm.
//
// This is a C++ wrapper with slight modification of a hungarian algorithm implementation by Markus Buehren.
// The original implementation is a few mex-functions for use in MATLAB, found here:
// http://www.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem
//
// Both this code and the orignal code are published under the BSD license.
// by Cong Ma, 2016
//
#include "hungarian_algorithm.h"
#include <cmath>  // for std::abs()
#include <limits>
#include <stdexcept>

namespace hungarian_algorithm
{
std::size_t HungarianSolver::findInCol(const std::vector<bool>& matrix, const std::size_t col) const
{
  std::size_t row = 0;
  for (std::size_t i = col; i < matrix.size() && !matrix[i]; i += num_cols_)
  {
    ++row;
  }
  return row;
}

std::size_t HungarianSolver::findInRow(const std::vector<bool>& matrix, const std::size_t row) const
{
  const typename std::vector<bool>::const_iterator row_begin = matrix.begin() + row * num_cols_;
  return static_cast<std::size_t>(std::distance(row_begin, std::find(row_begin, row_begin + num_cols_, true)));
}

void HungarianSolver::doSolve()
{
  covered_rows_.assign(num_rows_, false);
  covered_cols_.assign(num_cols_, false);
  star_matrix_.assign(cost_matrix_.size(), false);
  prime_matrix_.assign(cost_matrix_.size(), false);

  // Steps 1 and 2a:
  std::size_t i = 0;
  for (std::size_t row = 0; row < num_rows_; ++row)
  {
    // Find minimum
    std::size_t min_cost = std::numeric_limits<std::size_t>::max();
    for (std::size_t col = 0; col < num_cols_; ++col, ++i)
    {
      min_cost = std::min(min_cost, cost_matrix_[i]);
    }
    i -= num_cols_; // Reset index to beginning of row
    for (std::size_t col = 0; col < num_cols_; ++col, ++i)
    {
      cost_matrix_[i] -= min_cost;
      if (!covered_rows_[row] && !covered_cols_[col] && cost_matrix_[i] == 0)
      {
        star_matrix_[i] = true;
        covered_rows_[row] = true;
        covered_cols_[col] = true;
      }
    }
  }
  covered_rows_.assign(num_rows_, false);

  while (!areAllColumnsCovered())
  {
    step3();
  }
}

void HungarianSolver::coverStarredColumns()
{
  // Cover every column containing a starred zero:
  for (std::size_t col = 0; col < num_cols_; ++col)
  {
    if (findInCol(star_matrix_, col) < num_rows_)
    {
      covered_cols_[col] = true;
    }
  }
}

bool HungarianSolver::areAllColumnsCovered()
{
  return static_cast<std::size_t>(std::count(covered_cols_.begin(), covered_cols_.end(), true))
    >= std::min(num_rows_, num_cols_);
}

void HungarianSolver::step3()
{
  bool zeros_found = true;
  while (zeros_found)
  {
    // Find zeros in uncovered cells (primed zeros):
    zeros_found = false;
    for (std::size_t col = 0; col < num_cols_; ++col)
    {
      if (!covered_cols_[col])
      {
        for (std::size_t row = 0; row < num_rows_; ++row)
        {
          if (!covered_rows_[row] && cost_matrix_[getIndex(row, col)] == 0)
          {
            // Found a primed zero:
            prime_matrix_[getIndex(row, col)] = true;

            // Find starred zero in current row:
            const std::size_t star_col = findInRow(star_matrix_, row);
            if (star_col < num_cols_)
            {
              // Found a starred zero in current row => cover row, uncover column, and proceed with next column:
              covered_rows_[row] = true;
              covered_cols_[star_col] = false;
              zeros_found = true;
              break;
            }

            // No starred zero found => move to step 4:
            step4(row, col);
            return;
          }
        }
      }
    }
  }
}

void HungarianSolver::step4(const std::size_t row, const std::size_t col)
{
  // Generate temporary copy of star matrix:
  std::vector<bool> new_star_matrix(star_matrix_);

  // Star current zero:
  new_star_matrix[getIndex(row, col)] = true;

  // Find starred zero in current column:
  std::size_t star_col = col;
  std::size_t star_row = findInCol(star_matrix_, star_col);
  while (star_row < num_rows_)
  {
    // Unstar the starred zero:
    new_star_matrix[getIndex(star_row, star_col)] = false;

    // Find primed zero in current row:
    star_col = findInRow(prime_matrix_, star_row);

    // Star the primed zero:
    // FIXME: can't star_col be >= numCols here??
    assert(star_col < num_cols_);
    new_star_matrix[getIndex(star_row, star_col)] = true;

    // Find starred zero in current column:
    star_row = findInCol(star_matrix_, star_col);
  }

  // Use temporary copy as new star matrix:
  star_matrix_ = new_star_matrix;

  // Delete all primes, uncover all rows:
  prime_matrix_.assign(cost_matrix_.size(), false);
  covered_rows_.assign(num_rows_, false);
  coverStarredColumns();
}

void HungarianSolver::subtractSmallestUncoveredElement()
{
  // Find smallest uncovered element:
  std::size_t h = std::numeric_limits<std::size_t>::max();
  for (std::size_t row = 0; row < num_rows_; ++row)
  {
    if (!covered_rows_[row])
    {
      for (std::size_t col = 0; col < num_cols_; ++col)
      {
        if (!covered_cols_[col])
        {
          h = std::min(h, cost_matrix_[getIndex(row, col)]);
        }
      }
    }
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
      }
      else if (!covered_rows_[row] && !covered_cols_[col])
      {
        cost_matrix_[i] -= h;
      }
    }
  }
}
}
