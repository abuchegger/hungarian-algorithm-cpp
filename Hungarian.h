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

#include <cassert>
#include <vector>

class HungarianAlgorithm
{
public:
  HungarianAlgorithm() = default;
  virtual ~HungarianAlgorithm() = default;

  void setCosts(const std::vector<double>& costs, const std::size_t num_rows, const bool row_major_order = true);
  double findOptimalAssignment(std::vector<std::size_t>& assignment);

protected:
  // A simple helper class that saves a row-major ordered matrix in a vector.
  template<class T>
  class Matrix
  {
  public:
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

    std::vector<T> data_;
    std::size_t num_rows_;
    std::size_t num_cols_;
  };

  void buildassignmentvector(int* assignment, bool* starMatrix);
  void computeassignmentcost(int* assignment, double* cost, double* distMatrix, int nOfRows);
  void step2a(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
              bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
  void step2b(int* assignment, double* distMatrix, bool* starMatrix, bool* primeMatrix,
              bool* coveredColumns, bool* coveredRows, int minDim);
  void step3(int* assignment, double* distMatrix, bool* starMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
  void step4(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
  void step5(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);

  Matrix<double> costs_;
};


#endif // HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_H
