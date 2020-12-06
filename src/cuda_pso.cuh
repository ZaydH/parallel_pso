
#ifndef SERIAL_CUDA_PSO_H
#define SERIAL_CUDA_PSO_H

#include <cassert>
#include <cinttypes>
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "base_pso.h"
#include "cuda_config.cuh"
#include "types_general.h"


class CudaPSO : public BasePSO<CudaParam, CudaLoss> {
private:
  /** Number of CUDA blocks */
  IntType DEFAULT_N_;
  /** Number of CUDA threads */
  IntType M_;
  /** Separate kernels */
  const bool sep_kernel_;
  /** Use a fast, weakly-guaranteed randomization method */
  const bool fast_rand_;

  /** Particle values */
  CudaMat parts_;
  /** Particle velocities */
  CudaMat velos_;
  /** Best result */
  CudaMat best_ = nullptr;
  /** Size of the Cuda Matrices */
  const IntType tot_len_;
  /** Stronger, slower random number generator seeds */
  curandState* rng_;
  /** Faster, weaker random number generator seeds */
  IntType* fast_rng_;

  void fit_();

  /** Calculates the number of blocks for a standard vector operation of the full size */
  IntType make_n_blocks(IntType len) {
    return (len + this->M_ - 1) / this->M_;
  }

  /** Initializes the seeds of the random number generators */
  void seedRngs();
  /** Randomizes the specified vector */
  void randomize(CudaMat vec, IntType n, CudaParam bound_lo, CudaParam bound_hi,
                 cudaStream_t * stream);

 public:
  explicit CudaPSO(CudaConfig * config, bool sep_kernel, bool fast_rand);

  ~CudaPSO() {
    std::vector<CudaMat> ptrs = {this->parts_, this->velos_};
    for (auto ptr : ptrs)
      cudaFree(ptr);
    cudaFree(this->rng_);
    delete best_;
  }

  /**
   * Run at the end of an iteration to update the losses, best particle position, best overall
   * position etc.
   * @param itr Iteration number
   * @param tmp_scratch Scratch data structure used to store the point loss information
   * @param pos_best Best position for each particle vector
   * @param pos_best_losses
   * @param best_gpu Best overall position, stored on the GPU
   * @param best_loss_gpu Best overall loss, stored on the GPU
   */
  void calcLossAndUpdate(IntType itr, CudaMat tmp_scratch, CudaMat pos_best,
                         CudaMat pos_best_losses, CudaMat best_gpu, CudaMat best_loss_gpu);

  void printBest(const IntType iter = -1) const {
    assert(this->is_fit_);
    std::cout << this->name() << " ";
    if (iter >= 0)
      std::cout << "Iter " << iter << ":";

    for (IntType i = 0; i < this->config_->dim(); i++)
      std::cout << " " << this->best_[i];

    std::cout << "   -- Loss: " << this->best_loss_ << std::endl;
  }

  /** Name of the CUDA PSO learner */
  std::string name() const {
    std::stringstream ss;
    ss << "CUDA(" << this->M_ << ")-" << ((this->sep_kernel_) ? "baseline" : "stream");
    if (this->fast_rand_)
      ss << "-fastrand";
    return ss.str();
  }
};

#endif //SERIAL_CUDA_PSO_H
