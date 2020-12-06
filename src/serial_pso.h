//
// Created by Zayd Hammoudeh on 10/24/20.
//

#ifndef SERIAL_SERIAL_PSO_H
#define SERIAL_SERIAL_PSO_H

#include <cassert>
#include <cfloat>
#include <string>

#include "Eigen/Dense"
#include "base_pso.h"
#include "types_general.h"


template <class T>
class SerialPSO : public BasePSO<Param, LossFunc<T>> {
 private:
  /** Set of particle locations */
  T* parts_;
  /** Set of particle velocities */
  T* velos_;
  /** Best particle */
  T best_;

  CpuConfig<Param, T> * cpu_config_;

  void updateBest(T &best_point) {
    assert(best_point.rows() == this->parts_->rows());

    for (IntType i = 0; i < this->config_->n_particle(); i++) {
      Param pt_loss = this->config_->loss_func()(this->config_->dim(), this->parts_->row(i),
                                                 this->config_->n_ele(),
                                                 this->cpu_config_->ext_data_mat(),
                                                 this->cpu_config_->ext_labels_mat());
      Param old_loss = this->config_->loss_func()(this->config_->dim(), best_point.row(i),
                                                  this->config_->n_ele(),
                                                  this->cpu_config_->ext_data_mat(),
                                                  this->cpu_config_->ext_labels_mat());

      // Update the best point in the swarm
      if (old_loss > pt_loss)
        best_point.row(i) = this->parts_->row(i);
      // Need to place outside above due to first time through the data old_loss == pt_loss
      if (this->best_loss_ > pt_loss) {
        this->best_loss_ = pt_loss;
        this->best_ = this->parts_->row(i);
      }
    }
  }

  /** Bound the particle locations within the configuration limits */
  static void clip(T * mat, const Param lo, const Param hi) {
    *mat = mat->cwiseMin(hi).cwiseMax(lo);
  }

  /** Gets random matrix u.a.r. from set [0,1]^{n_particles x dim} */
  void randomize(T* mat) {
    // setRandom places in range [-1,1] so abs normalizes to range [0,1]
    *mat = mat->setRandom().cwiseAbs();
//    (*mat) = mat->cwiseAbs();
//    (*mat) *= 0.5;
  }

  /** Initialize the particles */
  T* initParticles(bool random = false, IntType n_particles = 0) const {
    // ToDo define the initialize function
    IntType rows = this->getSubparticleCount(n_particles);
    T* mat = new T(rows, this->config_->dim());
    if (random) {
      mat->setRandom();  // Randomly initializes in range [-1,1]
      Param scalar = (this->config_->bound_hi() - this->config_->bound_lo()) / 2;
      *mat *= scalar;
      // Need to add scalar back to normalized to range[0,bound_hi-bound_lo] then add bound_lo to
      // move to range [bound_lo,bound_hi]
      *mat += (scalar + this->config_->bound_lo()) * T::Ones(mat->rows(), mat->cols());
    }
    return mat;
  }

  /** Initialize the velocity information */
  T* initVelos(IntType n_particles = 0) const {
    T *mat = this->initParticles(n_particles);
    mat->setRandom();  // Sets all elements to range (-1,1)

    Param bound_diff = this->config_->bound_hi() - this->config_->bound_lo();
    *mat *= bound_diff;
    return mat;
  }

  /** Helper method for defining subparticle information */
  IntType getSubparticleCount(IntType n) const {
    if (n == 0)
      n = this->config_->n_particle();

    assert(n > 0);
    return n;
  }

  /** Serial fit method */
  void fit_() {
//    auto pos_best = Eigen::Map<T>(*this->parts_, this->parts_->rows(), this->parts_->cols());
    T pos_best = this->parts_->replicate(1, 1);
    Param v_bound = this->config_->bound_hi() - this->config_->bound_lo();

    this->updateBest(pos_best);

    T *r_p = this->initParticles();
    T *r_g = this->initParticles();
    if (this->config_->d()) // Initial state for verification
      this->printBest(0);

    for (IntType iter = 1; iter <= this->config_->n_iter(); iter++) {
      // Set the random offsets
      this->randomize(r_p);
      this->randomize(r_g);

      // New velocity
      (*this->velos_) *= this->config_->momentum();

      T point_mv = r_p->cwiseProduct(pos_best - *this->parts_);
      (*this->velos_) += this->config_->rate_point() * point_mv;

      T broadcast_best = this->best_.replicate(this->config_->n_particle(), 1);
      T global_mv = r_g->cwiseProduct(broadcast_best - *this->parts_);

      (*this->velos_) += this->config_->rate_global() * global_mv;

      // Update the particle locations
      *this->parts_ += this->config_->lr() * (*this->velos_);
      this->clip(this->velos_, -v_bound, v_bound);
      this->clip(this->parts_, this->config_->bound_lo(), this->config_->bound_hi());
      this->updateBest(pos_best);

      if (this->config_->d())
        this->printBest(iter);
    }
    // Clean-up memory to prevent memory leaks
    delete r_p;
    delete r_g;
  }

public:

  explicit SerialPSO(CpuConfig<Param,T> * config) : BasePSO<Param,LossFunc<T>>(config) {
    this->cpu_config_ = config;

    // Eigen uses the default library
    IntType seed = this->config_->d() ? 42 : time(nullptr);
    srand(seed);

    // Initialize the fit parameters including the velocity and particle locations
    this->parts_ = this->initParticles(true);

    this->velos_ = this->initVelos();
  }

  ~SerialPSO() {
    delete this->parts_;
    delete this->velos_;
  }

  /** Returns the best particle */
  T getBest() const {
    assert(this->is_fit_);
    return this->best_;
  }

  void printBest(const IntType iter = -1) const {
    assert(this->is_fit_);
    std::cout << this->name() << " ";
    if (iter >= 0)
      std::cout << "Iter " << iter << ":";
    std::cout << this->getBest() << "   -- Loss: " << this->best_loss_ << std::endl;
  }

  /** Name of the serial class */
  std::string name() const { return "Serial"; }
};

#endif //SERIAL_SERIAL_PSO_H
