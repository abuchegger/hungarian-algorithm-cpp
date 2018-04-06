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
#include "hungarian_algorithm.h"
#include <algorithm>

namespace hungarian_algorithm
{
HungarianSolverBase::HungarianSolverBase(const std::size_t num_rows, const std::size_t num_cols)
  : num_rows_(static_cast<std::size_t>(num_rows)), num_cols_(static_cast<std::size_t>(num_cols)),
    covered_rows_(num_rows_, false), covered_cols_(num_cols_, false), zero_matrix_(num_rows_ * num_cols_, false),
    star_matrix_(num_rows_ * num_cols_, false), prime_matrix_(num_rows_ * num_cols_, false)
{
}

std::size_t HungarianSolverBase::findInCol(const std::vector<bool>& matrix, const std::size_t col) const
{
  std::size_t row = 0;
  for (std::size_t i = col; i < matrix.size() && !matrix[i]; i += num_cols_)
  {
    ++row;
  }
  return row;
}

std::size_t HungarianSolverBase::findInRow(const std::vector<bool>& matrix, const std::size_t row) const
{
  const typename std::vector<bool>::const_iterator row_begin = matrix.begin() + row * num_cols_;
  return static_cast<std::size_t>(std::distance(row_begin, std::find(row_begin, row_begin + num_cols_, true)));
}

void HungarianSolverBase::coverStarredColumns()
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

bool HungarianSolverBase::areAllColumnsCovered() const
{
  return static_cast<std::size_t>(std::count(covered_cols_.begin(), covered_cols_.end(), true))
    >= std::min(num_rows_, num_cols_);
}

bool HungarianSolverBase::findAssignment()
{
  while (true)
  {
    // Find zeros in uncovered cells (primed zeros):
    bool zeros_found = false;
    for (std::size_t col = 0; col < num_cols_; ++col)
    {
      if (!covered_cols_[col])
      {
        for (std::size_t row = 0; row < num_rows_; ++row)
        {
          if (!covered_rows_[row] && zero_matrix_[getIndex(row, col)])
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

            // No starred zero found => create a new permutation of the star matrix, starting from the current row and
            // column:
            permute(row, col);
            return true;
          }
        }
      }
    }

    if (!zeros_found)
    {
      return false;
    }
  }
}

void HungarianSolverBase::permute(const std::size_t row, const std::size_t col)
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
    if (star_col >= num_cols_)
    {
      throw std::logic_error("Hungarian algorithm: star_col >= num_cols_; this should never happen");
    }
    new_star_matrix[getIndex(star_row, star_col)] = true;

    // Find starred zero in current column:
    star_row = findInCol(star_matrix_, star_col);
  }

  // Use temporary copy as new star matrix:
  star_matrix_ = new_star_matrix;

  // Delete all primes, uncover all rows:
  prime_matrix_.assign(prime_matrix_.size(), false);
  covered_rows_.assign(num_rows_, false);
  coverStarredColumns();
}
}
