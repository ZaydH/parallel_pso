//
// Created by Zayd Hammoudeh on 10/25/20.
//

#ifndef SERIAL_CPU_CONFIG_H
#define SERIAL_CPU_CONFIG_H

#include <cmath>
#include <iostream>
#include <sysexits.h>

#include "Eigen/Dense"
#include "base_config.h"
#include "types_cpu_only.h"
#include "types_general.h"

/** Simple convex method */
Param convex(const IntType &dim, const SerialMat &pt, const IntType &n_ele,
             const SerialMat &ext_data, const SerialMat &labels);

/** Simple convex method */
Param convex(const IntType &dim, const Param &pt, const IntType &n_ele, const Param &ext_data,
             const Param &labels);

/**
 * Rosenbrock function.  This definition is slightly different from Eberhart & Shi as their
 * definition is wrong since it assumes a dimension N+1 in an N-dimensional vector.
 */
Param rosenbrock(const IntType &dim, const SerialMat &pt, const IntType &n_ele,
                 const SerialMat &ext_data, const SerialMat &labels);

/**
 * Rosenbrock function.  This definition is slightly different from Eberhart & Shi as their
 * definition is wrong since it assumes a dimension N+1 in an N-dimensional vector.
 */
Param rosenbrock(const IntType &dim, const Param &pt, const IntType &n_ele, const Param &ext_data,
                 const Param &labels);

/** Generalized Rastrigrin function */
Param rastrigrin(const IntType &dim, const SerialMat &pt, const IntType &n_ele,
                 const SerialMat &ext_data, const SerialMat &labels);

/** Generalized Rastrigrin function */
Param rastrigrin(const IntType &dim, const Param &pt, const IntType &n_ele, const Param &ext_data,
                 const Param &labels);

/** Generalized Griewank function */
Param griewank(const IntType &dim, const SerialMat &pt, const IntType &n_ele,
               const SerialMat &ext_data, const SerialMat &labels);

/** Generalized Griewank function */
Param griewank(const IntType &dim, const Param &pt, const IntType &n_ele, const Param &ext_data,
               const Param &labels);

/** Generalized external data function */
Param ext_data(const IntType &dim, const SerialMat &pt, const IntType &n_ele,
               const SerialMat &ext_data, const SerialMat &labels);

/** Generalized external data function */
Param ext_data(const IntType &dim, const Param &pt, const IntType &n_ele, const Param &ext_data,
               const Param &labels);

template<class T>
LossFunc<T> getLossFunction(const std::string &task);
#include "cpu_config.cpp"
//extern template LossFunc<Param> getLossFunction<Param>(const std::string &);
//extern template LossFunc<SerialMat> getLossFunction<SerialMat>(const std::string &);

template<class S,class T>
class CpuConfig : public BaseConfig<S,LossFunc<T>> {
 protected:
  #if T == SerialMat
    SerialMat ser_data_;
    SerialMat ser_labels_;
  #endif

  /** Updates the loss function */
  void parseTask() {
    this->loss_ = getLossFunction<T>(this->task_name_);

    #if T == SerialMat
      if (this->is_ext_data()) {
        this->ser_data_ = SerialMat(this->n_ele(), this->dim());
        this->ser_labels_ = SerialMat(this->n_ele(), 1);

        for (IntType i = 0; i < this->n_ele(); i++) {
          this->ser_labels_(i, 0) = this->labels_[i];
          for (IntType j = 0; j < this->dim(); j++) {
            this->ser_data_(i, j) = this->data_[i * this->dim() + j];
          }
        }
      }
    #endif
  }

 public:
  explicit CpuConfig<S,T>(const char *config_path) : BaseConfig<S,LossFunc<T>>(config_path) {
    parseTask();
  };

  #if T == SerialMat
    SerialMat ext_labels_mat() const { return ser_labels_; }
    SerialMat ext_data_mat() const { return this->ser_data_; }
  #endif
};

#endif //SERIAL_CPU_CONFIG_H
