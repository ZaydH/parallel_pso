//
// Created by Zayd Hammoudeh on 10/26/20.
//

#include <iostream>
#include <sysexits.h>

#include "cpu_config.h"
#include "types_cpu_only.h"
#include "types_general.h"

/** Simple convex method */
Param convex(const IntType &dim, const SerialMat &pt,
             const IntType &n_ele, const SerialMat &ext_data,
             const SerialMat &labels) { // NOLINT(readability-non-const-parameter)
  return pt.squaredNorm();
}

/** Simple convex method */
Param convex(const IntType &dim, const Param &pt,
             const IntType &n_ele, const Param &ext_data,
             const Param &labels) { // NOLINT(readability-non-const-parameter)
  Param tot = 0;
  const Param * ptr = &pt;
  for (IntType i = 0; i < dim; i++)
    tot += pow(ptr[i], 2);
  return tot;
}

/**
 * Rosenbrock function.  This definition is slightly different from Eberhart & Shi as their
 * definition is wrong since it assumes a dimension N+1 in an N-dimensional vector.
 */
Param rosenbrock(const IntType &dim, const SerialMat &pt,
                 const IntType &n_ele, const SerialMat &ext_data,
                 const SerialMat &labels) { // NOLINT(readability-non-const-parameter)
  SerialMat head = pt.leftCols(dim - 1);
  SerialMat tail = pt.rightCols(dim - 1);

  Param tot = (head - SerialMat::Ones(1, dim - 1)).squaredNorm();
  SerialMat head2 = head.cwiseAbs2();
  tot += 100 * (tail - head2).squaredNorm();
  return tot;
}

/**
 * Rosenbrock function.  This definition is slightly different from Eberhart & Shi as their
 * definition is wrong since it assumes a dimension N+1 in an N-dimensional vector.
 */
Param rosenbrock(const IntType &dim, const Param &pt,
                 const IntType &n_ele, const Param &ext_data,
                 const Param &labels) { // NOLINT(readability-non-const-parameter)
  Param tot = 0;
  const Param * ptr = &pt;
  for (IntType i = 0; i < dim - 1; i++)
    tot += 100. * pow(ptr[i+1] - pow(ptr[i], 2), 2) + pow(ptr[i] - 1, 2);
  return tot;
}

/** Generalized Rastrigrin function */
Param rastrigrin(const IntType &dim, const SerialMat &pt,
                 const IntType &n_ele, const SerialMat &ext_data,
                 const SerialMat &labels) { // NOLINT(readability-non-const-parameter)
  const static Param pi = 4 * atan(1);
  Param tot = pt.squaredNorm() + 10. * dim;

  SerialMat prod = (2 * pi) * pt;
  tot -= 10 * prod.array().cos().sum();
  return tot;
}

/** Generalized Rastrigrin function */
Param rastrigrin(const IntType &dim, const Param &pt,
                 const IntType &n_ele, const Param &ext_data,
                 const Param &labels) { // NOLINT(readability-non-const-parameter)
  const static Param pi = 4 * atan(1);
  Param tot = 10 * dim;

  const Param * ptr = &pt;
  for (IntType i = 0; i < dim; i++) {
    Param val = ptr[i];
    tot += val * val;
    #if Param == double
      tot -= 10 * cos(2 * pi * val);
    #elif Param == float
      tot -= 10 * cosf(2 * pi * val);
    #else
      assert(false);
    #endif
  }
  return tot;
}

/** Generalized Griewank function */
Param griewank(const IntType &dim, const SerialMat &pt,
               const IntType &n_ele, const SerialMat &ext_data,
               const SerialMat &labels) { // NOLINT(readability-non-const-parameter)
//  Eigen::ArrayXd pt_arr = (1 * pt).array();
//  std::cout << pt_arr << std::endl;
  Eigen::ArrayXd range = Eigen::ArrayXd::LinSpaced(dim, 1, dim).cwiseInverse().sqrt();
//  std::cout << range.transpose() << std::endl;
//  std::cout << range << std::endl;
//  SerialMat pt_cp = 1 * pt;
//  Eigen::ArrayXd pt_arr = pt_cp.array();
//  std::cout << pt_cp << std::endl;
  Param prod = pt.cwiseProduct(range.transpose().matrix()).array().cos().prod();
  return 1. + pt.squaredNorm() / 4000. - prod;
}

/** Generalized Griewank function */
Param griewank(const IntType &dim, const Param &pt,
               const IntType &n_ele, const Param &ext_data,
               const Param &labels) { // NOLINT(readability-non-const-parameter)
  Param l2_tot = 0, prod = 1., val;
  const Param *ptr = &pt;
  for (IntType i = 0; i < dim; i++) {
    val = ptr[i];
    l2_tot += val * val;
    prod *= cos(val / sqrt(i + 1.));
  }
  return 1. + l2_tot / 4000. - prod;
}

/** Generalized function used to calculate loss for external data */
Param ext_data(const IntType &dim, const SerialMat &pt,
               const IntType &n_ele, const SerialMat &ext_data,
               const SerialMat &labels) { // NOLINT(readability-non-const-parameter)
  SerialMat vec = ext_data * pt.transpose();
  vec = -vec.cwiseProduct(labels);
  vec = vec.array().exp();
  vec += SerialMat::Ones(n_ele, 1);
  vec = vec.array().inverse();
  return vec.sum();
}

/** Generalized function used to calculate loss for external data */
Param ext_data(const IntType &dim, const Param &pt,
               const IntType &n_ele, const Param &ext_data,
               const Param &labels) { // NOLINT(readability-non-const-parameter)
  Param tot_loss = 0;
  const Param *part_ptr = &pt;
  const Param *dat_ptr = &ext_data;
  const Param *lbl_ptr = &labels;

  for (int i_ele = 0; i_ele < n_ele; i_ele++) {
    CudaParam val = 0;
    IntType ele_offset = i_ele * dim;

    for (int d = 0; d < dim; d++)
      val += part_ptr[d] * dat_ptr[ele_offset + d];

    // sigmoid loss
    #if Param == float
      tot_loss += 1 / (1 + expf(-1 * val * lbl_ptr[i_ele]));
    #elif Param == double
      tot_loss += 1 / (1 + exp(-1 * val * labels[i_ele]));
    #else
      assert(False); // Unknown type
    #endif
  }

  return tot_loss;
}

template<class T>
LossFunc<T> getLossFunction(const std::string &task) {
  if (task == "convex") {
    return convex;
  } else  if (task == "rosenbrock") {
    return rosenbrock;
  } else  if (task == "rastrigrin") {
    return rastrigrin;
  } else  if (task == "griewank") {
    return griewank;
  } else  if (task == "breast-cancer" || task == "ionosphere") {
    return ext_data;
  } else {
    std::cerr << "Unknown task \"" << task << "\"\n";
    exit(EX_DATAERR);
  }
}
//template LossFunc<Param> getLossFunction<Param>(const std::string &);
//template LossFunc<SerialMat> getLossFunction<SerialMat>(const std::string &);
