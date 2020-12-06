//
// Created by Zayd Hammoudeh on 10/24/20.
//

#ifndef SERIAL_TYPES_GENERAL_H
#define SERIAL_TYPES_GENERAL_H

#include <cinttypes>
#include <random>
#include <string>


/**
 * @brief Compile Warning Prevention Macro
 * 
 * Some variables return unused warnings if they only appear in assert statements.
 * This macro prevents that warning.  Inspired by:
 * 
 * https://stackoverflow.com/questions/777261/avoiding-unused-variables-warnings-when-using-assert-in-a-release-build
 */
#define _unused(x) ((void)(x))

/** Type of parameters in the vectors/array */
using Param = float;

/** Standardized integer type */
using IntType = int;

/** OpenMP CPU parameters */
using OmpMat = Param*;

/** Type of parameters in the vectors/array */
using CudaParam = Param;
using CudaMat = CudaParam*;

//using CudaLoss = CudaParam (*)(const IntType&, const CudaMat); // idx, dim, data NOLINT(misc-misplaced-const)
using CudaLoss = void (*)(CudaMat, CudaMat, const IntType, const IntType, const IntType,
                          CudaMat, CudaMat);

using ClInt = int;
using OpenClParam = float;
using OpenClMat = OpenClParam*;
using OpenClLoss = std::string;

template<class T>
using LossFunc = Param (*)(const IntType &, const T&, const IntType &, const T&, const T&);

#endif //SERIAL_TYPES_GENERAL_H
