//
// Created by Zayd Hammoudeh on 10/24/20.
//

#ifndef SERIAL_OPENMP_PSO_H
#define SERIAL_OPENMP_PSO_H

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cinttypes>
#include <cstring>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>

#include "base_pso.h"
#include "cpu_config.h"
#include "types_cpu_only.h"
#include "types_general.h"


template<class T>
class OmpParts {
 private:

  const IntType n_threads_;

  const PartLens * part_lens_;

  std::vector<T*> parts_;

 public:
  explicit OmpParts<T>(const PartLens * part_lens)
      : n_threads_(part_lens->size()), part_lens_(part_lens) {
    assert(this->n_threads_ > 0);

    // Create the particle arrays
    this->parts_ = std::vector<T*>(this->n_threads_);
    for (IntType i = 0; i < this->n_threads_; i++) {
      IntType len = (*this->part_lens_)[i];
      assert(len > 0);

      this->parts_[i] = new T[len];
    }
  }

  /** Delete the particle container */
  ~OmpParts<T>() {
    for (T* part : this->parts_)
      delete part;
  }

  OmpParts<T>(const OmpParts<T> &other)
      : n_threads_(other.n_threads_), part_lens_(other.part_lens_) {

    // Create the particle arrays
    this->parts_ = std::vector<T*>(this->n_threads_);
    #pragma omp parallel for default(shared)
    for (IntType i = 0; i < this->n_threads_; i++) {
      IntType len = (*this->part_lens_)[i];
      assert(len > 0);
      T* arr = new T[len];

      IntType n_bytes = len * sizeof(Param);
      memcpy(arr, other.parts_[i], n_bytes);

      this->parts_[i] = arr;
    }
  }

  /** Get the array for the specified thread */
  T* get(IntType thread_id) const {
    assert(thread_id < this->n_threads_);
    return this->parts_[thread_id];
  }

  /** Randomize the memory */
  void randomize(const IntType thread_id, PRNG * prng, const Param lo = 0,
                 const Param hi = 1) const {
    assert(thread_id < this->n_threads_);

    T* mat = this->parts_[thread_id];
    std::uniform_real_distribution<Param> dist(lo, hi);

    IntType len = (*this->part_lens_)[thread_id];
    for (IntType i = 0; i < len; i++)
      mat[i] = dist(*prng);
  }

  /** Bound the particle locations within the configuration limits */
  void clip(IntType tid, Param lo, Param hi) {
    assert((long unsigned int)tid < this->parts_.size());

    T* mat = this->parts_[tid];
    IntType len = (*this->part_lens_)[tid];
    for (IntType i = 0; i < len; i++)
      mat[i] = std::min(hi, std::max(lo, mat[i]));
  }

  /** Scale the parameter value by a float */
  void scale(IntType tid, Param scalar) {
    T* mat = this->parts_[tid];
    IntType len = (*this->part_lens_)[tid];

    for (IntType i = 0; i < len; i++)
      mat[i] *= scalar;
  }

  /** Sanity compatibility check */
  bool isCompatible(const OmpParts<T> &other) const {
    bool compat = this->n_threads_ == other.n_threads_;
    return compat && this->part_lens_ == other.part_lens_; // part_lens_ checks pointers
  }

  /** Add the one particle value to the other */
  void add(IntType tid, const Param scale, const OmpParts<T> * other) const {
    // Sanity check the parameters
    assert(tid < this->n_threads_);
    assert(this->isCompatible(*other));

    this->add(tid, scale, other->parts_[tid]);
  }

  void add(IntType tid, const Param scale, const T* arr) const {
    IntType len = (*this->part_lens_)[tid];

    T* mat = this->parts_[tid];
    for (IntType i = 0; i < len; i++)
      mat[i] += scale * arr[i];
  }
};


template<class T>
class OpenmpPSO : public BasePSO<T, LossFunc<T>> {
 private:
  PRNG ** prngs_;
  const IntType n_threads_;

  /** Particle vectors */
  OmpParts<T> * parts_;
  /** Velocity vectors */
  OmpParts<T> * velos_;
  PartLens part_lens_;

  /** Contains best parameters */
  T* best_;

  /** Simple method to construct the thread-specific seeds.  Iterate to improve seed diversity */
  void seedPRNGs() {
    // Initialize thread specific RNGs
    this->prngs_ = new PRNG*[this->n_threads_];

    // seed_seq generates nearly u.a.r. seeds
    std::seed_seq seq{time(nullptr)};
    std::vector<std::uint32_t> seeds(this->n_threads_);
    seq.generate(seeds.begin(), seeds.end());
    // Did not parallelize here intentionally since operations so fast not worth the overhead
    for (IntType i = 0; i < this->n_threads_; i++) {
      this->prngs_[i] = new PRNG(seeds[i]);
      if (this->config_->d())
        std::cout << "Seed Thread #" << i << ": " << seeds[i] << "\n";
    }
  }

  /** Copy the best vector into the memory */
  void copyParticle(T* src, T* dest = nullptr) {
    memcpy(dest, src, this->config_->dim() * sizeof(Param));
  }

  void fit_() {
    // Random variables setting part directions
    auto r_p = OmpParts<T>(&this->part_lens_);
    auto r_g = OmpParts<T>(&this->part_lens_);

    // Construct the best positions
    OmpParts<T> pos_best = (*this->parts_); // Relies on copy constructor
    this->updateBest(pos_best);
    if (this->config_->d())
      this->printBest(0);

    for (IntType itr = 1; itr <= this->config_->n_iter(); itr++) {
      #pragma omp parallel default(shared) // NOLINT(openmp-use-default-none)
      {
        IntType tid = omp_get_thread_num();
        PRNG * prng = this->prngs_[tid];
        r_p.randomize(tid, prng);
        r_g.randomize(tid, prng);

        this->velos_->scale(tid, this->config_->momentum()); // Attenuate the velocity
        this->updateVeloWithDiff(tid, this->config_->rate_point(), r_p, pos_best);
        this->updateVeloWithBest(tid, this->config_->rate_global(), r_g);

        // Update the particle positions
        this->parts_->add(tid, this->config_->lr(), this->velos_);

        //Ensure that the points and velocities are within the valid grid
        Param v_bound = this->config_->bound_hi() - this->config_->bound_lo();
        this->velos_->clip(tid, -v_bound, v_bound);
        this->parts_->clip(tid, this->config_->bound_lo(), this->config_->bound_hi());
      }
      this->updateBest(pos_best);
      if (this->config_->d())
        this->printBest(itr);
    }
  }

  void updateBest(OmpParts<T> &pos_best) {
    Param t_bl = this->best_loss_;
    IntType dim = this->config_->dim();

    #pragma omp parallel default(shared) firstprivate(t_bl) // NOLINT(openmp-use-default-none)
    {
      IntType tid = omp_get_thread_num();
      IntType tot_len = this->part_lens_[tid];

      T* t_parts = this->parts_->get(tid);
      T* t_olds = pos_best.get(tid);
      T* t_best = new T[this->config_->dim()];

      for (IntType i = 0; i < tot_len; i += dim) {
        T* p_cur = t_parts + i;
        T* p_old = t_olds + i;
        Param l_pos = this->config_->loss_func()(this->config_->dim(), *p_cur,
                                                 this->config_->n_ele(), *this->config_->ext_data(),
                                                 *this->config_->ext_labels());
        Param l_old = this->config_->loss_func()(this->config_->dim(), *p_old,
                                                 this->config_->n_ele(), *this->config_->ext_data(),
                                                 *this->config_->ext_labels());

        // Update the best position for this point
        if (l_pos < l_old)
          this->copyParticle(p_cur, p_old);
        // Update the global best candidate
        if (l_pos < t_bl) {
          t_bl = l_pos;
          this->copyParticle(p_cur, t_best);
        }
      }

      // Update the overall best result
      #pragma omp critical
      {
        if (t_bl < this->best_loss_) {
          this->best_loss_ = t_bl;
          this->copyParticle(t_best, this->best_);
        }
      }
    }
  }

  /** Use the difference between teh point's best and its current position to update the velocity */
  void updateVeloWithDiff(IntType tid, Param scalar, OmpParts<T> &r, OmpParts<T> &ref) {
    // Ensure the shapes are compatible
    assert(r.isCompatible(ref));
    assert(this->velos_->isCompatible(r));

    T* r_T = r.get(tid);
    T* ref_T = ref.get(tid);
    T* pos = this->parts_->get(tid);

    IntType len = this->part_lens_[tid];
    for (IntType i = 0; i < len; i++)
      r_T[i] *= ref_T[i] - pos[i];

    this->velos_->add(tid, scalar, r_T);
  }

  void updateVeloWithBest(IntType tid, Param scalar, OmpParts<T> &r) {
    // Ensure the shapes are compatible
    assert(this->velos_->isCompatible(r));

    T* r_T = r.get(tid);
    T* pos = this->parts_->get(tid);
    T* best = this->getBest();

    IntType dim = this->config_->dim();
    IntType tot_len = this->part_lens_[tid];
    for (IntType i = 0; i < tot_len; i++)
      r_T[i] *= best[i % dim] - pos[i];

    this->velos_->add(tid, scalar, r_T);
  }

public:
  explicit OpenmpPSO(CpuConfig<Param,T> *config, IntType n_threads)
      : BasePSO<Param, LossFunc<T>>(config), n_threads_(n_threads) {
    assert(this->n_threads_ > 0);
    omp_set_num_threads(n_threads);

    this->prngs_ = nullptr;
    this->seedPRNGs();

    // Define the length of each thread's parallel block
    this->part_lens_ = PartLens(this->n_threads_);
    IntType div = (this->config_->n_particle()) / (n_threads);
    for (IntType i = 0; i < n_threads - 1; i++)
      this->part_lens_[i] = this->config_->dim() * div;
    // Handle last thread subset to ensure coverage of whole range
    this->part_lens_[n_threads_ - 1] = this->config_->n_particle() - (n_threads_ - 1) * div;
    this->part_lens_[n_threads_ - 1] *= this->config_->dim();

    this->parts_ = new OmpParts<T>(&this->part_lens_);
    this->velos_ = new OmpParts<T>(&this->part_lens_);

    // Each threads set of particles can be randomized in parallel
    #pragma omp parallel default(shared) // NOLINT(openmp-use-default-none)
    {
      IntType tid = omp_get_thread_num();
      PRNG * prng = this->prngs_[tid];
      Param v_max = this->config_->bound_hi() - this->config_->bound_lo();

      this->parts_->randomize(tid, prng, this->config_->bound_lo(), this->config_->bound_hi());
      this->velos_->randomize(tid, prng, -v_max, v_max);
    };

    this->best_ = new T[this->config_->dim()];
  }

  ~OpenmpPSO() {
    for (IntType i = 0; i < this->n_threads_; i++)
      delete this->prngs_[i];
    delete this->prngs_;

    delete this->parts_;
    delete this->velos_;
    delete this->best_;
  }

  /** Returns copy of the best vector */
  T* getBest() {
    T* best = new T[this->config_->dim()];
    this->copyParticle(this->best_, best);
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
    ss << "OpenMP-CPU(" << this->n_threads_ << ")";
    return ss.str();
  }
};

#endif //SERIAL_OPENMP_PSO_H
