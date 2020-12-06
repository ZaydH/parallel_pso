//
// Created by Zayd Hammoudeh on 11/9/20.
//
// void initVec(int* prngs, float* vals, float lo, float diff, int n) {

// Generates a simple random number
__kernel void initVec(__global int * prngs,
                      __global float * vals,
                      float lo,
                      float diff
                     )
{
  int id = get_global_id(0);
  prngs[id] = 1103515245 * prngs[id] +	12345;
  vals[id] = lo + diff * (prngs[id] & 0x3FFFFFFF) / 0x3FFFFFFF;
}

// Generates a simple random number
__kernel void updatePosAndVelo(__global float * parts,
                               __global float * velos,
                               __global float * r_p,
                               __global float * r_g,
                               __global float * pos_best,
                               __global float * best,
                               float momentum,
                               float rate_point,
                               float rate_global,
                               float lr,
                               int dim
                              )
{
  int id = get_global_id(0);
  // Use a register for velo to prevent dirtying the memory unnecessarily
  float velo = velos[id];
  velo *= momentum;
  velo += rate_point * r_p[id] * (pos_best[id] - parts[id]);
  velo += rate_global * r_g[id] * (best[id % dim] - parts[id]);

  velos[id] = velo;
  parts[id] += lr * velo;
}

__kernel
void clip(__global float* data,
          float bound_lo,
          float bound_hi)
{
  int id = get_global_id(0);
  data[id] = clamp(data[id], bound_lo, bound_hi);
}


__kernel
void updatePosBest(__global float * parts,
                   __global float * parts_loss,
                   __global float * pos_best,
                   __global float * pos_best_loss,
                   int dim)
{
  int id = get_global_id(0);
  if (parts_loss[id] < pos_best_loss[id]) {
    pos_best_loss[id] = parts_loss[id];
    int offset = id * dim;
    for (int i = offset; i < offset + dim; i++)
      pos_best[i] = parts[i];
  }
}

__kernel
void updateTotBest(__global float * parts,
                   __global float * parts_loss,
                   __global float * best,
                   __global float * best_loss,
                   int n,
                   int dim)
{
  int id = get_global_id(0);
  if (id == 0) {
    int best_idx = -1;
    // Check for best overall loss
    for (int i = 0; i < n; i++) {
      if (best_loss[0] > parts_loss[i]) {
        best_idx = i;
        best_loss[0] = parts_loss[i];
      }
    }
    // Copy best part if applicable
    if (best_idx >= 0) {
      int offset = best_idx * dim;
      for (int i = 0; i < dim; i++)
        best[i] = parts[i + offset];
    }
  }
}

__kernel
void convex(__global float * dest,
            __global float * parts,
            int dim) {
  int id = get_global_id(0);
  int offset = dim * id;
  float tot = 0;

  for (int i = offset; i < offset + dim; i++)
    tot += parts[i] * parts[i];

  dest[id] = tot;
}

/**
 * Rosenbrock function.  This definition is slightly different from Eberhart & Shi as their
 * definition is wrong since it assumes a dimension N+1 in an N-dimensional vector.
 */
__kernel
void rosenbrock(__global float *dest,
                __global float *parts,
                int dim)
{
  int id = get_global_id(0);
  int offset = dim * id;
  float tot = 0;

  float cur = parts[offset];
  for (int i = offset + 1; i < offset + dim; i++) { // Stop one before end of particle's vector
    float next = parts[i];
    tot += 100. * pow(next - pow(cur, 2), 2) + pow(cur - 1, 2);
    cur = next;
  }
  
  dest[id] = tot;
}

/** Generalized Rastrigrin function */
__kernel
void rastrigrin(__global float *dest,
                __global float *parts,
                int dim)
{
  int id = get_global_id(0);
  int offset = dim * id;
  float tot = 10 * dim;

  for (int i = offset; i < offset + dim; i++) {
    float val = parts[i];
    tot += pow(val, 2) - 10 * cospi(2 * val);
  }

  dest[id] = tot;
}

/** Generalized Griewank function */
__kernel
void griewank(__global float *dest,
              __global float *parts,
              int dim)
{
  int id = get_global_id(0);
  int offset = dim * id;

  float l2_tot = 0, prod = 1.;

  for (int i = 0; i < dim; i++) {
    float val = parts[i + offset];
    l2_tot += pow(val, 2);
    prod *= cos(val / sqrt(i + 1.));
  }

  dest[id] = 1. + l2_tot / 4000. - prod;
}

/** Generalized loss framework */
__kernel
void ext_data(__global float *dest,
              __global float *parts,
              __global float *ext_data,
              __global float *labels,
              int n_ele,
              int dim) {
  int id = get_global_id(0);

  int part_offset = dim * id;
  float tot_loss = 0;
  for (int i_ele = 0; i_ele < n_ele; i_ele++) {
    float val = 0;
    int ele_offset = i_ele * dim;
    for (int d = 0; d < dim; d++)
      val += parts[part_offset + d] * ext_data[ele_offset + d];
    // Sigmoid Loss
    float lbl = labels[i_ele];
    tot_loss += 1.0 / (1.0 + exp(-1 * lbl * val));
  }
  dest[id] = tot_loss;
}
