//
// Created by Zayd Hammoudeh on 10/24/20.
//

#ifndef SERIAL_OPENMP_TEAMS_PSO_H
#define SERIAL_OPENMP_TEAMS_PSO_H

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cinttypes>
#include <cstring>
#include <ctime>
#include <iostream>
#include <memory>  // shared_ptr
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <sstream>

#include "base_pso.h"
#include "cpu_config.h"
#include "types_cpu_only.h"
#include "types_general.h"


template<class S>
class OpenmpTeamsPSO : public BasePSO<S, LossFunc<S>> {
 private:
  unsigned int * prngs_;
  /** Number of threads */
  const IntType n_thread_;
  /** Number of particles */
  const IntType n_part_;
  /** Dimension of the particle */
  const IntType dim_;
  /** Total length of the particle arrays */
  const IntType tot_len_;

  /** Particle vectors */
  S * parts_;
  /** Velocity vectors */
  S * velos_;

  /** Contains best parameters */
  S* best_;

  /** Simple method to construct the thread-specific seeds.  Iterate to improve seed diversity */
  void seedPRNGs() {
    // Initialize thread specific RNGs
    this->prngs_ = new unsigned int[this->tot_len_];

    PRNG prng(time(nullptr));
    std::uniform_int_distribution<uint64_t> dist(0, RAND_MAX - 1);
    for (IntType i = 0; i < tot_len_; i++)
        this->prngs_[i] = dist(prng);
  }

  void fit_() {
    // Stores the best result vector.  Exclude from param_prs as stores final result and will be
    // freed in the class destructor.
    S* best = new S[this->dim_];
    this->best_ = best;
    S* parts = this->parts_, *velos = this->velos_;
    LossFunc<S> loss = this->config_->loss_func();

    // Random variables used exclusively when fitting
    S *pos_best = nullptr, *pos_best_loss = nullptr;

    IntType n_part = this->n_part_, dim = this->dim_, tot_len = this->tot_len_, best_pos = 0;

    unsigned int *prngs = this->prngs_;
    // Initialize the loss values as negative
    pos_best = new S[tot_len];
    memcpy(pos_best, parts, tot_len * sizeof(S));
    pos_best_loss = new S[this->n_part_];
    #pragma omp parallel for
    for (IntType i = 0; i < n_part; i++)
      pos_best_loss[i] = loss(dim, *(parts + i * dim), this->config()->n_ele(),
                              *this->config()->ext_data(), *this->config()->ext_labels());

    // Need to define variables in the method or it yields a runtime seg fault
    S best_loss = std::numeric_limits<S>::max();
    for (IntType i = 0; i < n_part; i++) {
      if (this->best_loss_ > pos_best_loss[i]) {
        this->best_loss_ = pos_best_loss[i];
        best_pos = i;
      }
    }
    memcpy(best, pos_best + best_pos * dim, dim * sizeof(S));
    best_loss = this->best_loss_;

    if (this->config_->d())
      this->printBest(0);

    // Use consts to simplify copy in
    const S b_lo = this->config_->bound_lo(), b_hi = this->config_->bound_hi();
    const S vd = this->config_->bound_hi() - this->config_->bound_lo();
    const S rate_global = this->config_->rate_global(), rate_point = this->config_->rate_point();
    const S momentum = this->config_->momentum();
    const S lr = this->config_->lr();
    std::vector<S*> param_ptrs = {pos_best, pos_best_loss};

    #pragma omp target data \
          map(to: pos_best[:tot_len], best[:dim], velos[:tot_len], prngs[:tot_len]), \
          map(tofrom: parts[:tot_len])
    {
      for (IntType itr = 1; itr <= this->config_->n_iter(); itr++) {
        #pragma omp target update to(pos_best[:tot_len])
        #pragma omp target teams distribute parallel for
        for (IntType idx = 0; idx < tot_len; idx++) {
          // Simple fast RNG with no libraries to prevent issues compiling to the GPU
          prngs[idx] = 1103515245 * prngs[idx] +	12345;
          S r_p = 1. * (prngs[idx] & 0x3FFFFFFF) / 0x3FFFFFFF;
          prngs[idx] = 1103515245 * prngs[idx] +	12345;
          S r_g = 1. * (prngs[idx] & 0x3FFFFFFF) / 0x3FFFFFFF;
          // Update with momentum
          velos[idx] *= momentum;
          velos[idx] += rate_point * r_p * (pos_best[idx] - parts[idx]);
          velos[idx] += rate_global * r_g * (best[idx % dim] - parts[idx]);

          parts[idx] += lr * velos[idx];
          // Clip the results
          velos[idx] = (velos[idx] > vd) ? vd : ((velos[idx] < -vd) ? -vd : velos[idx]);
          parts[idx] = (parts[idx] > b_hi) ? b_hi : ((parts[idx] < b_lo) ? b_lo : parts[idx]);
        }
        // Download the part location information
        #pragma omp target update from(parts[:tot_len])

        // Update the losses
        #pragma omp parallel for
        for (IntType i = 0; i < n_part; i++) {
          IntType offset = i * dim;
          S p_loss = loss(dim, *(parts + offset), this->config_->n_ele(),
                          *this->config_->ext_data(), *this->config_->ext_labels());
          if (pos_best_loss[i] > p_loss) {
            pos_best_loss[i] = p_loss;
            memcpy(pos_best + offset, parts + offset, dim * sizeof(S));
          }
        }
        // Update the best overall (if applicable)
        for (IntType i = 0; i < n_part; i++) {
          if (best_loss > pos_best_loss[i]) {
            best_loss = pos_best_loss[i];
            best_pos = i;
          }
        }
        memcpy(best, pos_best + best_pos * dim, dim * sizeof(S));
        #pragma omp target update to(best[:dim])
        this->best_loss_ = best_loss;

        // Update and print the best particle information
        if (this->config_->d())
          this->printBest(itr);
      }
    }

    // =========== Cleanup the memory ===========
    for (auto ptr : param_ptrs)
      delete ptr;
  }


public:
  explicit OpenmpTeamsPSO(CpuConfig<S,S> *config, IntType n_threads)
      : BasePSO<S, LossFunc<S>>(config), n_thread_(n_threads),
        n_part_(config->n_particle()), dim_(config->dim()),
        tot_len_(config->n_particle() * config->dim()) {
    assert(this->n_thread_ > 0);
    omp_set_num_threads(this->n_thread_);

    this->seedPRNGs();

    this->parts_ = new S[this->tot_len_];
    this->velos_ = new S[this->tot_len_];

    // Define the random number generators used to create the variables
    S bound_lo = this->config_->bound_lo(), bound_hi = this->config_->bound_hi();
    S v_diff = bound_hi - bound_lo;
    // Initialize the initial particles and velocities
    std::uniform_real_distribution<S> rand01(0, 1);
    #pragma omp parallel for collapse(2)
    for (IntType i = 0; i < this->n_part_; i++) {
      for (IntType j = 0; j < this->dim_; j++) {
        IntType idx = i * this->dim_ + j;
        this->parts_[idx] = v_diff * rand_r(prngs_ + idx) / RAND_MAX + bound_lo;
        this->velos_[idx] = (2 * v_diff) * rand_r(prngs_ + idx) / RAND_MAX - v_diff;
      }
    }
  }

  ~OpenmpTeamsPSO() {
    delete this->prngs_;
    std::vector<S*> full_ptrs = {this->parts_, this->velos_, this->best_};
    for (auto ptr : full_ptrs)
      delete ptr;
  }

  /** Returns copy of the best vector */
  S* getBest() {
    S* best = new S[this->dim_];
    memcpy(best, this->best_, this->dim_ * sizeof(S));
    return best;
  }

  void printBest(const IntType iter = -1) const {
    assert(this->is_fit_);
    std::cout << this->name();
    if (iter >= 0)
      std::cout << " Iter " << iter << ":";

    for (IntType i = 0; i < this->config_->dim(); i++)
      std::cout << " " << this->best_[i];

    std::cout << "   -- Loss: " << this->best_loss_ << std::endl;
  }

  /** Name of the serial class */
  std::string name() const {
    std::stringstream ss;
    ss << "OpenMP-Teams(" << this->n_thread_ << ")";
    return ss.str();
  }
};

#endif //SERIAL_OPENMP_TEAMS_PSO_H
