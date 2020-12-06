//
// Created by Zayd Hammoudeh on 10/26/20.
//

#include <iostream>
#include <sysexits.h>

#include "cuda_config.cuh"
#include "math_constants.h"
#include "types_general.h"

/** Simple convex method */
__global__
void convex(CudaMat dest, CudaMat parts, const IntType n_part, const IntType dim,
            const IntType n_ele, CudaMat ext_data,
            CudaMat labels) { // NOLINT(readability-non-const-parameter)
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n_part) {
    IntType offset = dim * id;
    CudaParam tot = 0;
    for (IntType i = offset; i < offset + dim; i++)
      tot += parts[i] * parts[i];
    dest[id] = tot;
  }
}

/**
 * Rosenbrock function.  This definition is slightly different from Eberhart & Shi as their
 * definition is wrong since it assumes a dimension N+1 in an N-dimensional vector.
 */
__global__
void rosenbrock(CudaMat dest, CudaMat parts, const IntType n_part, const IntType dim,
                const IntType n_ele, CudaMat ext_data,
                CudaMat labels) { // NOLINT(readability-non-const-parameter)
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n_part) {
    IntType offset = dim * id;
    CudaParam tot = 0;
    for (IntType i = offset; i < offset + dim - 1; i++)  // Stop one before end of particle's vector
      tot += 100. * pow(parts[i + 1] - pow(parts[i], 2), 2) + pow(parts[i] - 1, 2);
    dest[id] = tot;
  }
}

/** Generalized Rastrigrin function */
__global__
void rastrigrin(CudaMat dest, CudaMat parts, const IntType n_part, const IntType dim,
                const IntType n_ele, CudaMat ext_data,
                CudaMat labels) { // NOLINT(readability-non-const-parameter)
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n_part) {
    IntType offset = dim * id;
    CudaParam tot = 10 * dim;
    for (IntType i = offset; i < offset + dim; i++) {
      CudaParam val = parts[i];
      tot += pow(val, 2) - 10 * cos(2 * CUDART_PI_F * val);
    }
    dest[id] = tot;
  }
}

/** Generalized Griewank function */
__global__
void griewank(CudaMat dest, CudaMat parts, const IntType n_part, const IntType dim,
              const IntType n_ele, CudaMat ext_data,
              CudaMat labels) { // NOLINT(readability-non-const-parameter)
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n_part) {
    IntType offset = dim * id;
    CudaParam l2_tot = 0, prod = 1., val;
    for (IntType i = 0; i < dim; i++) {
      val = parts[i + offset];
      l2_tot += pow(val, 2);
      prod *= cos(val / sqrt(i + 1.));
    }
    dest[id] = 1. + l2_tot / 4000. - prod;
  }
}

/** Generalized linear-in-parameter loss framework */
__global__
void ext_data(CudaMat dest, CudaMat parts, const IntType n_part, const IntType dim,
              const IntType n_ele, CudaMat ext_data,
              CudaMat labels) { // NOLINT(readability-non-const-parameter)
  IntType id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n_part) {
    IntType part_offset = dim * id;
    CudaParam tot_loss = 0;
    for (int i_ele = 0; i_ele < n_ele; i_ele++) {
      CudaParam val = 0;
      IntType ele_offset = i_ele * dim;
      for (int d = 0; d < dim; d++)
        val += parts[part_offset + d] * ext_data[ele_offset + d];
      // Sigmoid Loss
      CudaParam lbl = labels[i_ele];
      tot_loss += 1.0 / (1.0 + exp(-1 * lbl * val));
    }
    dest[id] = tot_loss;
  }
}


CudaLoss parseCudaTask(const std::string &task) {
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


void CudaConfig::allocateExtDataMemory() {
  assert(sizeof(Param) == sizeof(CudaParam));

  // Allocate the EXTERNAL DATA on the GPU
  IntType n_bytes = this->dim_ * this->n_ele_ * sizeof(Param);
  cudaMalloc(&this->cu_data_, n_bytes);
  cudaMemcpy(this->cu_data_, this->data_, n_bytes, cudaMemcpyHostToDevice);

  // Allocate the EXTERNAL LABELS on the GPU
  n_bytes = this->n_ele_ * sizeof(Param);
  cudaMalloc(&this->cu_labels_, n_bytes);
  cudaMemcpy(this->cu_labels_, this->labels_, n_bytes, cudaMemcpyHostToDevice);
}


CudaConfig::~CudaConfig() {
  if (cu_data_)
    cudaFree(cu_data_);
  if (cu_labels_)
    cudaFree(cu_labels_);
}
