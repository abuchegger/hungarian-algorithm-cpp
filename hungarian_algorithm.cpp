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


void HungarianAlgorithm::setCosts(const std::vector<double>& costs, const std::size_t num_rows,
                                  const bool row_major_order)
{
  if (costs.empty())
  {
    throw std::invalid_argument("costs vector is empty");
  }

  if (num_rows == 0 || (costs.size() % num_rows) != 0)
  {
    throw std::invalid_argument("costs vector has invalid number of rows or columns");
  }

  const std::size_t num_cols = costs.size() / num_rows;

  if (row_major_order)
  {
    costs_ = Matrix<double>(num_rows, num_cols, costs);
  }
  else
  {
    costs_ = Matrix<double>(num_rows, num_cols, 0.0);
    for (std::size_t row = 0; row < num_rows; ++row)
    {
      for (std::size_t col = 0; col < num_cols; ++col)
      {
        costs_.at(row, col) = costs[col * num_rows + row];
      }
    }
  }
}

//********************************************************//
// Find optimal solution for assignment problem using Munkres algorithm, also known as Hungarian Algorithm.
//********************************************************//
double HungarianAlgorithm::findOptimalAssignment(std::vector<std::size_t>& assignment)
{
  assignment.assign(costs_.num_rows_, std::numeric_limits<std::size_t>::max());
  double total_cost = 0.0;

  // Make working copy of cost matrix:
  Matrix<double> costs_copy(costs_);

  // Create some auxillary vectors and matrices:
  std::vector<bool> covered_cols(costs_copy.num_cols_, false);
  std::vector<bool> covered_rows(costs_copy.num_rows_, false);
  Matrix<bool> star_matrix(costs_copy.num_rows_, costs_copy.num_cols_, false);
  Matrix<bool> prime_matrix(costs_copy.num_rows_, costs_copy.num_cols_, false);

  // Preliminary steps:
  std::size_t min_dim;
  if (costs_copy.num_rows_ <= costs_copy.num_cols_)
  {
    min_dim = costs_copy.num_rows_;

    for (std::size_t row = 0; row < costs_copy.num_rows_; row++)
    {
      // Find the smallest element in the row:
      double min_value = std::numeric_limits<double>::max();
      for (std::size_t col = 0; col < costs_copy.num_cols_; ++col)
      {
        const double value = costs_copy(row, col);
        if (value < min_value)
        {
          min_value = value;
        }
      }

      // Subtract the smallest element from each element of the row:
      for (std::size_t col = 0; col < costs_copy.num_cols_; ++col)
      {
        costs_copy(row, col) -= min_value;
      }
    }

    // Steps 1 and 2a:
    for (std::size_t row = 0; row < costs_copy.num_rows_; row++)
    {
      for (std::size_t col = 0; col < costs_copy.num_cols_; col++)
      {
        if (std::abs(costs_copy(row, col)) < std::numeric_limits<double>::epsilon() && !covered_cols[col])
        {
          star_matrix(row, col) = true;
          covered_cols[col] = true;
          break;
        }
      }
    }
  }
  else // num_rows_ > num_cols_
  {
    min_dim = costs_copy.num_cols_;

    for (std::size_t col = 0; col < costs_copy.num_cols_; col++)
    {
      // Find the smallest element in the column:
      double min_value = std::numeric_limits<double>::max();
      for (std::size_t row = 0; row < costs_copy.num_rows_; ++row)
      {
        const double value = costs_copy(row, col);
        if (value < min_value)
        {
          min_value = value;
        }
      }

      // Subtract the smallest element from each element of the column:
      for (std::size_t row = 0; row < costs_copy.num_rows_; ++row)
      {
        costs_copy(row, col) -= min_value;
      }
    }

    // Steps 1 and 2a:
    for (std::size_t col = 0; col < costs_copy.num_cols_; col++)
    {
      for (std::size_t row = 0; row < costs_copy.num_rows_; row++)
      {
        if (std::abs(costs_copy(row, col)) < std::numeric_limits<double>::epsilon() && !covered_rows[row])
        {
          star_matrix(row, col) = true;
          covered_cols[col] = true;
          covered_rows[row] = true;
          break;
        }
      }
    }
    covered_rows.assign(covered_rows.size(), false);
  }

/* move to step 2b */
  step2b(assignment,
         costs_copy,
         star_matrix,
         prime_matrix,
         covered_cols,
         covered_rows,
         min_dim);

/* compute cost and remove invalid assignments */
  computeassignmentcost(assignment, total_cost, costs_, num_rows_);

  return total_cost;
}

/********************************************************/
void HungarianAlgorithm::buildassignmentvector(int* assignment, bool* star_matrix)
{
  int row, col;

  for (row = 0; row < num_rows_; row++)
    for (col = 0; col < num_cols_; col++)
      if (star_matrix[row + num_rows_ * col])
      {
#ifdef ONE_INDEXING
        assignment[row] = col + 1; /* MATLAB-Indexing */
#else
        assignment[row] = col;
#endif
        break;
      }
}

/********************************************************/
void HungarianAlgorithm::computeassignmentcost(int* assignment, double* total_cost, double* costs_copy, int num_rows_)
{
  int row, col;

  for (row = 0; row < num_rows_; row++)
  {
    col = assignment[row];
    if (col >= 0)
      *total_cost += costs_copy[row + num_rows_ * col];
  }
}

/********************************************************/
void HungarianAlgorithm::step2a(int* assignment,
                                double* costs_copy,
                                bool* star_matrix,
                                bool* newStarMatrix,
                                bool* prime_matrix,
                                bool* covered_cols,
                                bool* covered_rows,
                                int num_rows_,
                                int num_cols_,
                                int minDim)
{
  bool* starMatrixTemp, * columnEnd;
  int col;

  /* cover every column containing a starred zero */
  for (col = 0; col < num_cols_; col++)
  {
    starMatrixTemp = star_matrix + num_rows_ * col;
    columnEnd = starMatrixTemp + num_rows_;
    while (starMatrixTemp < columnEnd)
    {
      if (*starMatrixTemp++)
      {
        covered_cols[col] = true;
        break;
      }
    }
  }

  /* move to step 3 */
  step2b(assignment,
         costs_copy,
         star_matrix,
         prime_matrix,
         covered_cols,
         covered_rows,
         minDim);
}

/********************************************************/
void HungarianAlgorithm::step2b(int* assignment,
                                double* costs_copy,
                                bool* star_matrix,
                                bool* prime_matrix,
                                bool* covered_cols,
                                bool* covered_rows,
                                int minDim)
{
  int col, nOfCoveredColumns;

  /* count covered columns */
  nOfCoveredColumns = 0;
  for (col = 0; col < num_cols_; col++)
    if (covered_cols[col])
      nOfCoveredColumns++;

  if (nOfCoveredColumns == minDim)
  {
    /* algorithm finished */
    buildassignmentvector(assignment, star_matrix);
  }
  else
  {
    /* move to step 3 */
    step3(assignment,
          costs_copy,
          star_matrix,
          prime_matrix,
          covered_cols,
          covered_rows,
          num_rows_,
          num_cols_,
          minDim);
  }

}

/********************************************************/
void HungarianAlgorithm::step3(int* assignment,
                               double* costs_copy,
                               bool* star_matrix,
                               bool* prime_matrix,
                               bool* covered_cols,
                               bool* covered_rows,
                               int num_rows_,
                               int num_cols_,
                               int minDim)
{
  bool zerosFound;
  int row, col, starCol;

  std::vector<bool> new_star_matrix(costs_.data_.size(), false);

  zerosFound = true;
  while (zerosFound)
  {
    zerosFound = false;
    for (col = 0; col < num_cols_; col++)
      if (!covered_cols[col])
        for (row = 0; row < num_rows_; row++)
          if ((!covered_rows[row]) && (fabs(costs_copy[row + num_rows_ * col]) < DBL_EPSILON))
          {
            /* prime zero */
            prime_matrix[row + num_rows_ * col] = true;

            /* find starred zero in current row */
            for (starCol = 0; starCol < num_cols_; starCol++)
              if (star_matrix[row + num_rows_ * starCol])
                break;

            if (starCol == num_cols_) /* no starred zero found */
            {
              /* move to step 4 */
              step4(assignment,
                    costs_copy,
                    star_matrix,
                    new_star_matrix,
                    prime_matrix,
                    covered_cols,
                    covered_rows,
                    num_rows_,
                    num_cols_,
                    minDim,
                    row,
                    col);
              return;
            }
            else
            {
              covered_rows[row] = true;
              covered_cols[starCol] = false;
              zerosFound = true;
              break;
            }
          }
  }

  /* move to step 5 */
  step5(assignment,
        costs_copy,
        star_matrix,
        new_star_matrix,
        prime_matrix,
        covered_cols,
        covered_rows,
        num_rows_,
        num_cols_,
        minDim);
}

/********************************************************/
void HungarianAlgorithm::step4(int* assignment,
                               double* costs_copy,
                               bool* star_matrix,
                               bool* newStarMatrix,
                               bool* prime_matrix,
                               bool* covered_cols,
                               bool* covered_rows,
                               int num_rows_,
                               int num_cols_,
                               int minDim,
                               int row,
                               int col)
{
  int n, starRow, starCol, primeRow, primeCol;
  int nOfElements = num_rows_ * num_cols_;

  /* generate temporary copy of star_matrix */
  for (n = 0; n < nOfElements; n++)
    newStarMatrix[n] = star_matrix[n];

  /* star current zero */
  newStarMatrix[row + num_rows_ * col] = true;

  /* find starred zero in current column */
  starCol = col;
  for (starRow = 0; starRow < num_rows_; starRow++)
    if (star_matrix[starRow + num_rows_ * starCol])
      break;

  while (starRow < num_rows_)
  {
    /* unstar the starred zero */
    newStarMatrix[starRow + num_rows_ * starCol] = false;

    /* find primed zero in current row */
    primeRow = starRow;
    for (primeCol = 0; primeCol < num_cols_; primeCol++)
      if (prime_matrix[primeRow + num_rows_ * primeCol])
        break;

    /* star the primed zero */
    newStarMatrix[primeRow + num_rows_ * primeCol] = true;

    /* find starred zero in current column */
    starCol = primeCol;
    for (starRow = 0; starRow < num_rows_; starRow++)
      if (star_matrix[starRow + num_rows_ * starCol])
        break;
  }

  /* use temporary copy as new star_matrix */
  /* delete all primes, uncover all rows */
  for (n = 0; n < nOfElements; n++)
  {
    prime_matrix[n] = false;
    star_matrix[n] = newStarMatrix[n];
  }
  for (n = 0; n < num_rows_; n++)
    covered_rows[n] = false;

  /* move to step 2a */
  step2a(assignment,
         costs_copy,
         star_matrix,
         newStarMatrix,
         prime_matrix,
         covered_cols,
         covered_rows,
         num_rows_,
         num_cols_,
         minDim);
}

/********************************************************/
void HungarianAlgorithm::step5(int* assignment,
                               double* costs_copy,
                               bool* star_matrix,
                               bool* newStarMatrix,
                               bool* prime_matrix,
                               bool* covered_cols,
                               bool* covered_rows,
                               int num_rows_,
                               int num_cols_,
                               int minDim)
{
  double h, value;
  int row, col;

  /* find smallest uncovered element h */
  h = DBL_MAX;
  for (row = 0; row < num_rows_; row++)
    if (!covered_rows[row])
      for (col = 0; col < num_cols_; col++)
        if (!covered_cols[col])
        {
          value = costs_copy[row + num_rows_ * col];
          if (value < h)
            h = value;
        }

  /* add h to each covered row */
  for (row = 0; row < num_rows_; row++)
    if (covered_rows[row])
      for (col = 0; col < num_cols_; col++)
        costs_copy[row + num_rows_ * col] += h;

  /* subtract h from each uncovered column */
  for (col = 0; col < num_cols_; col++)
    if (!covered_cols[col])
      for (row = 0; row < num_rows_; row++)
        costs_copy[row + num_rows_ * col] -= h;

  /* move to step 3 */
  step3(assignment,
        costs_copy,
        star_matrix,
        newStarMatrix,
        prime_matrix,
        covered_cols,
        covered_rows,
        num_rows_,
        num_cols_,
        minDim);
}
