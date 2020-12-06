//
// Created by Zayd Hammoudeh on 10/25/20.
//

#ifndef SERIAL_BASE_PSO_H
#define SERIAL_BASE_PSO_H

#include <cassert>
#include <cfloat>
#include <limits>
#include <string>

#include "base_config.h"
#include "stopwatch.h"


template<class T, class L>
class BasePSO {
 private:
  StopWatch stopwatch_;
  double runtime_ = DBL_MAX;

 protected:
  BaseConfig<T,L> * config_;

  /** Store whether the model is fit */
  bool is_fit_ = false;

  /** Task specific fit method */
  virtual void fit_() = 0;

  /** Best training loss */
  T best_loss_ = std::numeric_limits<T>::max();

 public:
  explicit BasePSO(BaseConfig<T,L> * config) {
    this->config_ = config;
  }

  /** Wrap fit method in an outer function to prevent need to duplicate boilerplate code */
  void fit() {
    std::cout << "Starting fit " << this->name() << "..." << std::endl;
    assert(!this->is_fit_);
    this->stopwatch_.start();

    this->is_fit_ = true;

    this->fit_();

    this->stopwatch_.stop();
    this->runtime_ = this->stopwatch_.getElapsedSeconds();
  }

  /** Returns the PSO's best loss. */
  T best_loss() const {
    assert(this->is_fit_);
    return this->best_loss_;
  }

  /** Simple method for printing the best loss */
  virtual void printBest(IntType iter) const = 0;

  /** Prints the final training results */
  void printFinalResults() const {
    this->printBest(-1);
    std::cout << "Runtime: " << this->runtime_ << "s\n";
    std::cout << "Best Loss: " << this->best_loss() << "\n";
  }

  /** Accessor for the learner's configuration */
  BaseConfig<T,L>* config() const { return this->config_; }

  /** Execution time */
  double exec_time() const { return this->runtime_; }

  virtual std::string name() const = 0;
};

#endif //SERIAL_BASE_PSO_H