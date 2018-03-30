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
// TODO: make brute force algorithm a separate class again, a common base creates too much unnecessary template fuzz.
// TODO: fix new algorithm implementation, and remove old, fully templated version.
//
#ifndef HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_ALGORITHM_H
#define HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_ALGORITHM_H

#include <algorithm>
#include <cassert>
#include <limits>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

namespace hungarian_algorithm
{
// A simple helper class that saves a row-major ordered matrix in a vector.
template<class T>
class Matrix
{
public:
  // Declare a few types like the STL containers do:
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
    if (data_.size() != (num_rows_ * num_cols_))
    {
      data_.clear();
      num_rows_ = 0;
      num_cols_ = 0;
      throw std::invalid_argument("Size of data does not match num_rows * num_cols");
    }
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

  void print(std::ostream& out, const std::string& indent = std::string(), const std::string& colsep = ", ",
             const std::string& rowsep = ";\n") const
  {
    // Save formatting info for later:
    const std::streamsize width = out.width(0);
    const std::streamsize precision = out.precision();
    std::size_t i = 0;
    for (std::size_t row = 0; row < num_rows_; ++row)
    {
      out << indent;
      for (std::size_t col = 0; col < num_cols_; ++col)
      {
        if (col != 0)
        {
          out << colsep;
        }
        out.width(width);
        out.precision(precision);
        out << data_[i++];
      }
      out << rowsep;
    }
  }

  template<typename Size>
  void printAssignment(std::ostream& out, const std::vector<Size>& assignment,
                       const std::string& indent = std::string()) const
  {
    Size row(0);
    for (std::size_t i = 0; i < assignment.size(); ++i, ++row)
    {
      out << indent << i << " -> ";
      const Size col = assignment[i];
      if (0 <= col && col < Size(num_cols_))
      {
        out << col << " (cost: " << (*this)(row, col) << ")" << std::endl;
      }
      else
      {
        out << "invalid" << std::endl;
      }
    }
  }

protected:
  container_type data_;
  std::size_t num_rows_;
  std::size_t num_cols_;
};

// Tag classes for CostNormalizationStrategy:
class MinimizeCosts;
class MaximizeCosts;

template<typename Cost, typename CostNormalizationStrategy>
struct CostNormalizer
{
  // Must be implemented by specializations:
  static Cost getBetterCost(const Cost& a, const Cost& b);
  static Cost normalizeCost(const Cost& cost, const Cost& best_cost);
};

template<typename Cost>
struct CostNormalizer<Cost, MinimizeCosts>
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
struct CostNormalizer<Cost, MaximizeCosts>
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

// Tag classes for SolvingMethod:
/**
 * Computes optimal assignment by iteratively looking for minimal costs for the remaining assignments. Also known as
 * Munkres or Kuhn-Munkres algorithm.
 */
class HungarianMethod;

/**
 * Finds optimal assignment by iterating over all possible combinations.
 */
class BruteForceMethod;

template<typename Cost, typename CostNormalizationStrategy, typename SolvingMethod>
class SolverBase;

template<typename Cost, typename CostNormalizationStrategy = MinimizeCosts, typename SolvingMethod = HungarianMethod>
class Solver
{
protected:
  friend class SolverBase<Cost, CostNormalizationStrategy, SolvingMethod>;

  // Must be implemented by specializations:
  void doSolve();
  std::size_t getAssignment(const std::size_t row) const;
};

template<typename Cost, typename CostNormalizationStrategy, typename SolvingMethod>
class SolverBase
{
public:
  typedef Cost CostType;
  typedef Solver<Cost, CostNormalizationStrategy, SolvingMethod> SolverImplementation;
  typedef std::pair<std::size_t, Cost> CombinedCost;

  explicit SolverBase(const Cost invalid_cost)
    : invalid_cost_(invalid_cost)
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
   * @return the cost of the optimal assignment, as a pair consisting of the number of invalid assignments, and the sum
   *         of the costs of the valid assignments.
   */
  template<typename CostFunction, typename Size, typename AssignmentMap>
  CombinedCost solve(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                     AssignmentMap& assignment_map)
  {
    copyAndNormalizeCosts<CostFunction, Size>(cost_function, num_rows, num_cols);

    if (num_rows > Size(0) && num_cols > Size(0))
    {
      static_cast<SolverImplementation*>(this)->doSolve();
    }

    return fillAssignmentMap<CostFunction, Size, AssignmentMap>(
      cost_function, num_rows, num_cols, AssignmentMapAdapter<Size, AssignmentMap>(assignment_map, num_rows, num_cols));
  }

  template<typename AssignmentMap>
  CombinedCost solve(const Matrix<Cost>& cost_matrix, AssignmentMap& assignment_map)
  {
    return solve(cost_matrix, cost_matrix.numRows(), cost_matrix.numCols(), assignment_map);
  }

protected:
  template<typename CostFunction, typename Size>
  void copyAndNormalizeCosts(const CostFunction& cost_function, const Size num_rows, const Size num_cols)
  {
    typedef CostNormalizer<Cost, CostNormalizationStrategy> CostNormalizerImplementation;

    // Make cost matrix square, filling extraneous elements with invalid cost:
    const std::size_t max_size(static_cast<std::size_t>(std::max(num_rows, num_cols)));
    cost_matrix_.assign(max_size, max_size, invalid_cost_);

    // Copy costs, making sure optimal element is zero, and every other one is larger; note that this does not touch
    // extraneous and invalid elements:
    Cost best_cost = invalid_cost_;
    std::size_t s_row = 0;
    for (Size row(0); row < num_rows; ++row, ++s_row)
    {
      std::size_t s_col = 0;
      for (Size col(0); col < num_cols; ++col, ++s_col)
      {
        const Cost cost(cost_function(row, col));
        cost_matrix_(s_row, s_col) = cost;
        if (cost != invalid_cost_)
        {
          if (best_cost == invalid_cost_)
          {
            best_cost = cost;
          }
          else
          {
            best_cost = CostNormalizerImplementation::getBetterCost(best_cost, cost);
          }
        }
      }
    }

    s_row = 0;
    for (Size row(0); row < num_rows; ++row, ++s_row)
    {
      std::size_t s_col = 0;
      for (Size col(0); col < num_cols; ++col, ++s_col)
      {
        Cost& cost(cost_matrix_(s_row, s_col));
        if (cost != invalid_cost_)
        {
          cost = CostNormalizerImplementation::normalizeCost(cost, best_cost);
        }
      }
    }
  }

  template<typename CostFunction, typename Size, typename AssignmentMap>
  CombinedCost fillAssignmentMap(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                                 const AssignmentMapAdapter<Size, AssignmentMap>& assignment_map)
  {
    // Fill assignment vector and compute total cost of assignment:
    CombinedCost total_cost(0, Cost(0));
    std::size_t s_row = 0;
    for (Size row(0); row < num_rows; ++row, ++s_row)
    {
      const std::size_t s_col(num_cols > Size(0) ? static_cast<SolverImplementation*>(this)->getAssignment(s_row) : 0);
      const Size col(s_col);
      if (s_col < std::size_t(num_cols) && Size(0) <= col && col < num_cols) // Watch out for overflows and the like
      {
        assignment_map.insert(row, col);
        total_cost.second += cost_function(row, col);
      }
      else
      {
        ++total_cost.first;
      }
    }
    return total_cost;
  }

  const Cost invalid_cost_;
  Matrix<Cost> cost_matrix_;
};

template<typename Cost, typename CostNormalizationStrategy>
class Solver<Cost, CostNormalizationStrategy, HungarianMethod>
  : public SolverBase<Cost, CostNormalizationStrategy, HungarianMethod>
{
public:
  explicit Solver(const Cost invalid_cost, const Cost epsilon_cost = std::numeric_limits<Cost>::epsilon())
    : SolverBase<Cost, CostNormalizationStrategy, HungarianMethod>(invalid_cost), epsilon_cost_(epsilon_cost)
  {
  }

  explicit Solver(const Cost epsilon_cost = std::numeric_limits<Cost>::epsilon())
    : SolverBase<Cost, CostNormalizationStrategy, HungarianMethod>(std::numeric_limits<Cost>::max()),
      epsilon_cost_(epsilon_cost)
  {
  }

protected:
  friend class SolverBase<Cost, CostNormalizationStrategy, HungarianMethod>;
  using SolverBase<Cost, CostNormalizationStrategy, HungarianMethod>::cost_matrix_;
  using SolverBase<Cost, CostNormalizationStrategy, HungarianMethod>::invalid_cost_;

  void doSolve()
  {
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

    while (!areAllColumnsCovered())
    {
      step3();
    }
  }

  std::size_t getAssignment(const std::size_t row) const
  {
    return star_matrix_.findInRow(row, true);
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

  void step3()
  {
    bool zeros_found = true;
    while (zeros_found)
    {
      // Find zeros in uncovered cells (primed zeros):
      zeros_found = false;
      for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
      {
        if (!covered_cols_[col])
        {
          for (std::size_t row = 0; row < cost_matrix_.numRows(); ++row)
          {
            if (!covered_rows_[row] && cost_matrix_(row, col) <= epsilon_cost_)
            {
              // Found a primed zero:
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
              return;
            }
          }
        }
      }
    }

    // No more zeros found => generate more zeros by subtracting smallest remaining element:
    subtractSmallestUncoveredElement();
  }

  void step4(const std::size_t row, const std::size_t col)
  {
    // Generate temporary copy of star matrix:
    Matrix<bool> new_star_matrix(star_matrix_);

    // Star current zero:
    new_star_matrix(row, col) = true;

    // Find starred zero in current column:
    std::size_t star_col = col;
    std::size_t star_row = star_matrix_.findInCol(star_col, true);
    while (star_row < cost_matrix_.numRows())
    {
      // Unstar the starred zero:
      new_star_matrix(star_row, star_col) = false;

      // Find primed zero in current row:
      star_col = prime_matrix_.findInRow(star_row, true);

      // Star the primed zero:
      // FIXME: can't star_col be >= numCols here??
      assert(star_col < cost_matrix_.numCols());
      new_star_matrix(star_row, star_col) = true;

      // Find starred zero in current column:
      star_row = star_matrix_.findInCol(star_col, true);
    }

    // Use temporary copy as new star matrix:
    star_matrix_ = new_star_matrix;

    // Delete all primes, uncover all rows:
    prime_matrix_.assign(false);
    covered_rows_.assign(covered_rows_.size(), false);
    coverStarredColumns();
  }

  void subtractSmallestUncoveredElement()  // aka step5
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
            if (h == invalid_cost_)
            {
              h = cost_matrix_(row, col);
            }
            else
            {
              h = std::min(h, cost_matrix_(row, col));
            }
          }
        }
      }
    }

    // Subtract h from each uncovered element, and add h to each doubly covered element:
    for (std::size_t row = 0; row < cost_matrix_.numRows(); ++row)
    {
      for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
      {
        if (covered_rows_[row] && covered_cols_[col])
        {
          cost_matrix_(row, col) += h;
        }
        else if (!covered_rows_[row] && !covered_cols_[col])
        {
          cost_matrix_(row, col) -= h;
        }
      }
    }
  }

  const Cost epsilon_cost_;
  std::vector<bool> covered_rows_;
  std::vector<bool> covered_cols_;
  Matrix<bool> star_matrix_;
  Matrix<bool> prime_matrix_;
};

class HungarianSolver
{
public:
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
             const CostComparator cost_comparator = std::less<Cost>())
  {
    if (num_rows > Size(0) && num_cols > Size(0))
    {
      copyAndNormalizeCosts<Size, CostComparator, CostFunction>(cost_function, num_rows, num_cols, cost_comparator);
      doSolve();
    }

    return fillAssignmentMap<Cost, Size, CostFunction, AssignmentMap>(
      cost_function, num_rows, num_cols, AssignmentMapAdapter<Size, AssignmentMap>(assignment_map, num_rows, num_cols));
  }

  template<typename Cost, typename Size, typename CostFunction, typename AssignmentMap>
  Cost solve(const CostFunction& cost_function, const Size num_rows, const Size num_cols, AssignmentMap& assignment_map)
  {
    return solve<Cost>(cost_function, num_rows, num_cols, assignment_map, std::less<Cost>());
  }

protected:
  template<typename Size, typename CostFunction, typename CostComparator>
  struct CostFunctionIndexComparator
  {
    CostFunctionIndexComparator(const CostFunction& cost_function, const CostComparator& cost_comparator)
      : cost_function_(cost_function), cost_comparator_(cost_comparator)
    {
    }

    bool operator()(const std::pair<Size, Size>& index_a, const std::pair<Size, Size>& index_b)
    {
      return cost_comparator_(cost_function_(index_a.first, index_a.second),
                              cost_function_(index_b.first, index_b.second));
    }

    const CostFunction& cost_function_;
    const CostComparator cost_comparator_;
  };

  template<typename Size, typename CostComparator, typename CostFunction>
  void copyAndNormalizeCosts(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                             const CostComparator cost_comparator)
  {
    // TODO: exchange num_rows and num_cols if num_rows > num_cols

    typedef typename std::pair<Size, Size> CostFunctionIndex;

    num_rows_ = static_cast<std::size_t>(num_rows);
    num_cols_ = static_cast<std::size_t>(num_cols);
    std::vector<CostFunctionIndex> indices_((num_rows_ * num_cols_));
    std::size_t i = 0;
    for (Size row(0); row < num_rows; ++row)
    {
      for (Size col(0); col < num_cols; ++col, ++i)
      {
        indices_[i].first = row;
        indices_[i].second = col;
      }
    }

    std::sort(indices_.begin(), indices_.end(),
              CostFunctionIndexComparator<Size, CostFunction, CostComparator>(cost_function, cost_comparator));

    cost_matrix_.resize(indices_.size());
    std::size_t cost = 0;
    for (i = 0; i < indices_.size(); ++i)
    {
      if (i > 0 && cost_comparator(cost_function(indices_[i - 1].first, indices_[i - 1].second),
                                   cost_function(indices_[i].first, indices_[i].second)))
      {
        // Only increase cost if original cost also increased (= isn't equal); this should save some iterations later:
        ++cost;
      }
      cost_matrix_[static_cast<std::size_t>(indices_[i].first * num_cols + indices_[i].second)] = cost;
    }
  }

  template<typename Cost, typename Size, typename CostFunction, typename AssignmentMap>
  Cost fillAssignmentMap(const CostFunction& cost_function, const Size num_rows, const Size num_cols,
                         const AssignmentMapAdapter<Size, AssignmentMap>& assignment_map)
  {
    // Fill assignment vector and compute total cost of assignment:
    Cost total_cost(0);
    if (num_cols > Size(0))
    {
      std::size_t s_row = 0;
      for (Size row(0); row < num_rows; ++row, ++s_row)
      {
        const std::size_t s_col(findInRow(star_matrix_, s_row));
        if (s_col < num_cols_)
        {
          const Size col(s_col);
          assignment_map.insert(row, col);
          total_cost += cost_function(row, col);
        }
      }
    }
    return total_cost;
  }

  inline std::size_t getIndex(const std::size_t row, const std::size_t col) const
  {
    return row * num_cols_ + col;
  }

  std::size_t findInCol(const std::vector<bool>& matrix, const std::size_t col) const;
  std::size_t findInRow(const std::vector<bool>& matrix, const std::size_t row) const;

  void doSolve();
  void coverStarredColumns();
  bool areAllColumnsCovered();
  void step3();
  void step4(const std::size_t row, const std::size_t col);
  void subtractSmallestUncoveredElement();  // aka step5

  std::size_t num_rows_;
  std::size_t num_cols_;
  std::vector<std::size_t> cost_matrix_;
  std::vector<bool> covered_rows_;
  std::vector<bool> covered_cols_;
  std::vector<bool> star_matrix_;
  std::vector<bool> prime_matrix_;
};

template<typename Cost, typename CostNormalizationStrategy>
class Solver<Cost, CostNormalizationStrategy, BruteForceMethod>
  : public SolverBase<Cost, CostNormalizationStrategy, BruteForceMethod>
{
public:
  explicit Solver(const Cost invalid_cost = std::numeric_limits<Cost>::max())
    : SolverBase<Cost, CostNormalizationStrategy, BruteForceMethod>(invalid_cost)
  {
  }

protected:
  friend class SolverBase<Cost, CostNormalizationStrategy, BruteForceMethod>;
  using typename SolverBase<Cost, CostNormalizationStrategy, BruteForceMethod>::CombinedCost;
  using SolverBase<Cost, CostNormalizationStrategy, BruteForceMethod>::cost_matrix_;
  using SolverBase<Cost, CostNormalizationStrategy, BruteForceMethod>::invalid_cost_;

  void doSolve()
  {
    current_assignment_.assign(cost_matrix_.numRows(), std::numeric_limits<std::size_t>::max());
    optimal_assignment_.assign(cost_matrix_.numRows(), std::numeric_limits<std::size_t>::max());
    covered_cols_.assign(cost_matrix_.numCols(), false);
    min_cost_ = CombinedCost(std::numeric_limits<std::size_t>::max(), std::numeric_limits<Cost>::max());
    doSolve(0, CombinedCost(0, Cost(0)));
  }

  void doSolve(const std::size_t row, const CombinedCost accumulated_cost)
  {
    for (std::size_t col = 0; col < cost_matrix_.numCols(); ++col)
    {
      if (!covered_cols_[col])
      {
        const Cost cost = cost_matrix_(row, col);
        CombinedCost new_accumulated_cost = accumulated_cost;
        if (cost == invalid_cost_)
        {
          new_accumulated_cost.first += 1;
        }
        else
        {
          new_accumulated_cost.second += cost;
        }
        if (row < (cost_matrix_.numRows() - 1))
        {
          current_assignment_[row] = col;
          covered_cols_[col] = true;
          doSolve(row + 1, new_accumulated_cost);
          covered_cols_[col] = false;
        }
        else if (new_accumulated_cost < min_cost_)
        {
          min_cost_ = new_accumulated_cost;
          current_assignment_[row] = col;
          optimal_assignment_ = current_assignment_;
        }
      }
    }
  }

  std::size_t getAssignment(const std::size_t row) const
  {
    return optimal_assignment_[row];
  }

  std::vector<std::size_t> current_assignment_;
  std::vector<std::size_t> optimal_assignment_;
  std::vector<bool> covered_cols_;
  // We count rows containing invalid assignments separately to mitigate some overflow / loss of significance problems
  // found during testing, affecting both integral and floating-point types:
  CombinedCost min_cost_;
};
}

#endif // HUNGARIAN_ALGORITHM_CPP_HUNGARIAN_ALGORITHM_H
