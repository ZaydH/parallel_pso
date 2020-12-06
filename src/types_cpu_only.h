//
// Created by zayd on 10/26/20.
//

#ifndef SERIAL_TYPES_CPU_ONLY_H
#define SERIAL_TYPES_CPU_ONLY_H

#include <random>
#include <vector>

#include "Eigen/Dense"
#include "types_general.h"

/** Matrix used in serial implementation */
using SerialMat = Eigen::MatrixXd;

/** OpenMP variables */
using PRNG = std::mt19937;
using PartLens = std::vector<IntType>;

#endif //SERIAL_TYPES_CPU_ONLY_H
