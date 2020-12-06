//
// Created by Zayd Hammoudeh on 10/26/20.
//

#include <cassert>
#include <cfloat>
#include <curand_kernel.h>
#include <list>
#include <omp.h>

#include "base_pso.h"
#include "cuda_config.cuh"
#include "cuda_pso.cuh"
#include "types_general.h"

#define PARALLEL_MAX_REDUCE


__global__
void initRand(curandState * state) {
  IntType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(clock64(), idx, 0, &state[idx]);
}

__global__
void vecAdd(CudaMat v, CudaParam scalar) {
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  v[id] += scalar;
}

__global__
void vecScale(CudaMat v, CudaParam scalar) {
  // Get our global thread ID
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  // Make sure we do not go out of bounds
  v[id] *= scalar;
}

__global__
void vecCwiseProd(CudaMat dest, const CudaMat other) { // NOLINT(readability-non-const-parameter,misc-misplaced-const)
  // Get our global thread ID
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  // Make sure we do not go out of bounds
  dest[id] *= other[id];
}

__global__
void vecDiff(CudaMat dest, const CudaMat left, const CudaMat right) {
  // Get our global thread ID
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  // Make sure we do not go out of bounds
  dest[id] = left[id] - right[id];
}

__global__
void vecClip(CudaMat v, CudaParam bound_lo, CudaParam bound_hi) {
  // Get our global thread ID
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  v[id] = fmax(bound_lo, fmin(v[id], bound_hi));
}

/** y = aX + y */
__global__
void vecSaxpy(CudaMat y, CudaParam a, const CudaMat x) {
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  y[id] += a * x[id];
}

__global__
void vecRand(curandState * state, CudaMat v) {
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  #if CudaParam == double
    v[id] = curand_uniform_double(state + id);
  #elif CudaParam == float
    v[id] = curand_uniform(&state[id]);
  #else
    assert(false);
  #endif
}

__global__
void vecRandFast(IntType * prngs, CudaMat v, CudaParam lo, CudaParam diff) {
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  prngs[id] = 1103515245 * prngs[id] +	12345;
  v[id] = lo + diff * (prngs[id] & 0x3FFFFFFF) / 0x3FFFFFFF;
}

/**
 * Uses the best vector position and calculates the relative position for use in calculating
 * the velocity.
 * @param dest Location to store the position
 * @param best Best overall position so far
 * @param parts_ Vector of all particle locations
 * @param n_part Number of particles
 */
__global__
void vecBestDiff(CudaMat dest, CudaMat best, CudaMat parts_) {
  IntType idx = blockIdx.x * blockDim.x + threadIdx.x;
  dest[idx] = best[threadIdx.x] - parts_[idx];
}

__global__
void vecCombine(CudaMat parts, CudaMat velos, CudaMat const tmp_p, CudaMat const tmp_g,
                const CudaParam momentum, const CudaParam bound_lo, const CudaParam bound_hi,
                const CudaParam lr) {
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  CudaParam diff = bound_hi - bound_lo;
  velos[id] = momentum * velos[id] + tmp_p[id] + tmp_g[id];
  parts[id] += lr * velos[id];

  velos[id] = fmax(-diff, fmin(velos[id], diff));
  parts[id] = fmax(bound_lo, fmin(parts[id], bound_hi));
}

__global__
void vecPoint(CudaMat tmp, CudaMat const pos_best, CudaMat const parts, CudaMat const r_p,
              const CudaParam rate_point) {
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  tmp[id] = rate_point * r_p[id] * (pos_best[id] - parts[id]);
}

__global__
void vecGlobal(CudaMat tmp, CudaMat const best_gpu, CudaMat const parts, CudaMat const r_g,
               const CudaParam rate_global) {
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  tmp[id] = rate_global * r_g[id] * (best_gpu[threadIdx.x] - parts[id]);
}

__global__
void updatePosBest(CudaMat parts, CudaMat parts_loss, CudaMat pos_best, // NOLINT(misc-misplaced-const,readability-non-const-parameter)
                   CudaMat pos_best_loss) {
  IntType idx = blockIdx.x;
  // Update the best position for the part
  if (pos_best_loss[idx] > parts_loss[idx]) {
    IntType tid = threadIdx.x;
    IntType offset = idx * blockDim.x + tid;
    pos_best[offset] = parts[offset];
    __syncthreads();
    if (tid == 0)
      pos_best_loss[idx] = parts_loss[idx];
  }
}

#ifdef PARALLEL_MAX_REDUCE
  __global__
  void updateBest(CudaMat pos_best, CudaMat pos_best_loss, CudaMat best,
                  CudaMat best_loss, IntType n_part, IntType dim) {

    unsigned int n_threads = blockDim.x;
    unsigned int tid = threadIdx.x;

    extern __shared__ float shared[];
    CudaParam * all_loss = shared;
    int * best_idx = (int *)&shared[n_threads];

    CudaParam t_loss = *best_loss;
    IntType b_idx = -1;
    for (IntType idx = tid; idx < n_part; idx += n_threads) {
      if (t_loss > pos_best_loss[idx]) {
        t_loss = pos_best_loss[idx];
        b_idx = idx;
      }
    }
    all_loss[tid] = t_loss;
    best_idx[tid] = b_idx;
    for (unsigned int s = n_threads / 2; s > 0; s>>=1) {
      __syncthreads();
      if (tid < s) {
        if (all_loss[tid] > all_loss[tid + s]) {
          all_loss[tid] = all_loss[tid + s];
          best_idx[tid] = best_idx[tid + s];
        }
      }
    }
    // Copy the best overall result only once
    if (tid == 0 && best_idx[0] >= 0) {
      IntType offset = best_idx[0] * dim;
      *best_loss = all_loss[0];
      for (IntType j = 0; j < dim; j++)
        best[j] = pos_best[j+offset];
    }
  }
#else
  __global__
  void updateBest(CudaMat pos_best, CudaMat pos_best_loss, CudaMat best,
                  CudaMat best_loss, IntType n_part, IntType dim) {
    IntType best_idx;
    bool best_found = false; // Used to queue the best result
    for (IntType idx = 0; idx < n_part; idx++) {
      // Update the best overall position
      if (best_loss[0] > pos_best_loss[idx]) {
        best_loss[0] = pos_best_loss[idx];
        best_idx = idx;
        best_found  = true;
      }
    }
    // Copy the best overall result only once
    if (best_found) {
      IntType offset = best_idx * dim;
      for (IntType j = 0; j < dim; j++)
        best[j] = pos_best[j+offset];
    }
  }
#endif

void CudaPSO::seedRngs() {
  if (this->fast_rand_) {
    std::seed_seq seq{time(nullptr)};
    std::vector<std::uint32_t> seeds(this->tot_len_);
    seq.generate(seeds.begin(), seeds.end());

    // Place the random seeds on the device
    IntType n_bytes = this->tot_len_ * sizeof(IntType);
    cudaMalloc(&this->fast_rng_, n_bytes);
    cudaMemcpy(this->fast_rng_, seeds.data(), n_bytes, cudaMemcpyHostToDevice);
  } else {
    cudaMalloc(&this->rng_, this->tot_len_ * sizeof(curandState_t));
    initRand<<<DEFAULT_N_,M_>>>(this->rng_);
  }
}


void CudaPSO::randomize(CudaMat vec, IntType n, CudaParam bound_lo, CudaParam bound_hi,
                        cudaStream_t * stream = nullptr) {
  CudaParam diff = bound_hi - bound_lo;
  if (this->fast_rand_) {
    if (!stream) {
      vecRandFast<<<DEFAULT_N_, M_>>>(this->fast_rng_, vec, bound_lo, diff);
    } else {
      vecRandFast<<<DEFAULT_N_,M_,0,*stream>>>(this->fast_rng_, vec, bound_lo, diff);
    }
  } else {
    if (!stream) {
      vecRand<<<DEFAULT_N_, M_>>>(this->rng_, vec);

      if (diff != 1.)
        vecScale<<<DEFAULT_N_, M_>>>(vec, diff);
      if (bound_lo != 0.)
        vecAdd<<<DEFAULT_N_, M_>>>(vec, bound_lo);
    } else {
      vecRand<<<DEFAULT_N_,M_,0,*stream>>>(this->rng_, vec);

      if (diff != 1.)
        vecScale<<<DEFAULT_N_,M_,0,*stream>>>(vec, diff);
      if (bound_lo != 0.)
        vecAdd<<<DEFAULT_N_,M_,0,*stream>>>(vec, bound_lo);
    }
  }
}


CudaPSO::CudaPSO(CudaConfig *config, const bool sep_kernel, const bool fast_rand)
  : BasePSO<CudaParam, CudaLoss>(config), tot_len_(config->n_particle() * config->dim()),
    M_(config->dim()), DEFAULT_N_(config->n_particle()),
    sep_kernel_(sep_kernel), fast_rand_(fast_rand) {
  assert(M_ > 0);
  assert(DEFAULT_N_ > 0);

  // Allocate the random number generator memory
  this->seedRngs();

  this->best_ = new CudaParam[this->config_->dim()];

  IntType tot_len_bytes = this->tot_len_ * (sizeof(int) + sizeof(CudaParam));
  // Particle and velocity information
  cudaMalloc(&this->parts_, tot_len_bytes);
  this->randomize(this->parts_, this->tot_len_, this->config_->bound_lo(), this->config_->bound_hi());

  cudaMalloc(&this->velos_, tot_len_bytes);
  CudaParam v_max = this->config_->bound_hi() - this->config_->bound_lo();
  this->randomize(this->velos_, this->tot_len_, -v_max, v_max);
}


void CudaPSO::calcLossAndUpdate(IntType itr, CudaMat tmp_scratch, CudaMat pos_best,
                                CudaMat pos_best_losses, CudaMat best_gpu,
                                CudaMat best_loss_gpu) {
  CudaLoss loss = this->config_->loss_func();
  const IntType bl_tr = 32;
  const IntType n_part_blocks = (this->config_->n_particle() + bl_tr + 1) / bl_tr;
  loss<<<n_part_blocks, bl_tr>>>(tmp_scratch, this->parts_,
                                 this->config_->n_particle(), this->config_->dim(),
                                 this->config_->n_ele(), this->config_->ext_data(),
                                 this->config_->ext_labels());

  updatePosBest<<<DEFAULT_N_, M_>>>(this->parts_, tmp_scratch, pos_best, pos_best_losses);

  #ifdef PARALLEL_MAX_REDUCE
    int n_threads = std::min(64, (int)log2((float)this->config_->n_particle()));
    unsigned int shared = n_threads * (sizeof(int) + sizeof(CudaParam));
    updateBest<<<1, n_threads, shared>>>(pos_best, pos_best_losses, best_gpu, best_loss_gpu,
                                         this->config_->n_particle(), this->config_->dim());
  #else
    updateBest<<<1,1>>>(pos_best, pos_best_losses, best_gpu, best_loss_gpu,
                        this->config_->n_particle(), this->config_->dim());
  #endif

//  printf("Iter: %d", itr);
  IntType best_len = this->config_->dim() * sizeof(CudaParam);
  cudaMemcpy(this->best_, best_gpu, best_len, cudaMemcpyDeviceToHost);
  cudaMemcpy(&this->best_loss_, best_loss_gpu, sizeof(CudaParam), cudaMemcpyDeviceToHost);
//  printf(" After\n");

  if (this->config_->d())
    this->printBest(itr);
}


void CudaPSO::fit_() {
  CudaMat r_p, r_g, pos_best, pos_best_losses, tmp_p, tmp_g, best_gpu, best_loss_gpu;

  IntType best_len = this->config_->dim() * sizeof(CudaParam);
  cudaMalloc(&best_gpu, best_len);
  cudaMalloc(&best_loss_gpu, sizeof(CudaParam));
  // Set an high loss then overwrite
  this->best_loss_ = std::numeric_limits<CudaParam>::max();
  cudaMemcpy(best_loss_gpu, &this->best_loss_, sizeof(CudaParam), cudaMemcpyHostToDevice);

  // List is used to both allocate and later free the CUDA memory
  IntType tot_len_bytes = this->tot_len_ * sizeof(CudaParam);
  std::list<CudaMat*> tot_len_ptrs = {&r_p, &r_g, &pos_best, &tmp_p, &tmp_g};
  #pragma omp parallel for
  for (auto ptr : tot_len_ptrs)
    cudaMalloc(ptr, tot_len_bytes);
  // Copy the best position information
  cudaMemcpy(pos_best, this->parts_, tot_len_bytes, cudaMemcpyDeviceToDevice);

  // Number of blocks when operating at the particle level -- not the parameter level
  cudaMalloc(&pos_best_losses, this->config_->n_particle() * sizeof(CudaParam));
  this->calcLossAndUpdate(0, pos_best_losses, pos_best, pos_best_losses,
                          best_gpu, best_loss_gpu);

  IntType n_part_blocks = (this->config_->n_particle() + M_ + 1) / M_;

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Iteration loop
  for (IntType itr = 1; itr <= this->config_->n_iter(); itr++) {

    if (this->sep_kernel_) {
      this->randomize(r_p, this->tot_len_, 0., 1.);
      this->randomize(r_g, this->tot_len_, 0., 1.);

      vecScale<<<DEFAULT_N_,M_>>>(this->velos_, this->config_->momentum());

      // Use particle's previous best information
      vecDiff<<<DEFAULT_N_,M_>>>(tmp_p, pos_best, this->parts_);
      vecCwiseProd<<<DEFAULT_N_, M_>>>(tmp_p, r_p);
      vecSaxpy<<<DEFAULT_N_,M_>>>(this->velos_, this->config_->rate_point(), tmp_p);

      // Use global best particle information
      vecBestDiff<<<DEFAULT_N_,M_>>>(tmp_g, best_gpu, this->parts_);
      vecCwiseProd<<<DEFAULT_N_, M_>>>(tmp_g, r_g);
      vecSaxpy<<<DEFAULT_N_,M_>>>(this->velos_, this->config_->rate_global(), tmp_g);

      vecSaxpy<<<DEFAULT_N_,M_>>>(this->parts_, this->config_->lr(), this->velos_);

      // Update the position and velocities including clipping
      CudaParam v_max = this->config_->bound_hi() - this->config_->bound_lo();
      vecClip<<<DEFAULT_N_,M_>>>(this->velos_, -v_max, v_max);
      vecClip<<<DEFAULT_N_,M_>>>(this->parts_, this->config_->bound_lo(),
                                 this->config_->bound_hi());
    } else {
      this->randomize(r_p, this->tot_len_, 0., 1., &stream1);
      vecPoint<<<DEFAULT_N_,M_,0,stream1>>>(tmp_p, pos_best, this->parts_, r_p,
                                              this->config_->rate_point());

      this->randomize(r_g, this->tot_len_, 0., 1., &stream2);
      vecGlobal<<<DEFAULT_N_, M_,0,stream2>>>(tmp_g, best_gpu, this->parts_, r_g,
                                              this->config_->rate_global());
      cudaDeviceSynchronize();
      vecCombine<<<DEFAULT_N_, M_>>>(this->parts_, this->velos_, tmp_p, tmp_g,
                                     this->config_->momentum(), this->config_->bound_lo(),
                                     this->config_->bound_hi(), this->config_->lr());
    }

    // == Update the best position information
    this->calcLossAndUpdate(itr, tmp_p, pos_best, pos_best_losses, best_gpu, best_loss_gpu);
  }

  // ===== Cleanup
  for (auto ptr : tot_len_ptrs)
    cudaFree(*ptr);
  cudaFree(best_gpu);
  cudaFree(best_loss_gpu);
  cudaFree(pos_best_losses);
}

