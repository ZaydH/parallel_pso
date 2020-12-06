#define USE_SERIAL

#include <cstdlib>
#include <iostream>
#include <sysexits.h>
#include <vector>

#include "cpu_config.h"
#include "logger.h"
#ifdef USE_SERIAL
  #include "Eigen/Dense"
  #include "serial_pso.h"
#endif
#ifdef USE_CUDA
  #include "cuda_config.cuh"
  #include "cuda_pso.cuh"
#endif
#ifdef USE_OPENCL
  #include "opencl_z_pso.h"
#endif
#ifdef USE_OPENMP
  #include "openmp_pso.h"
#endif
#ifdef USE_OPENMP_TEAMS
  #include "openmp_teams_pso.h"
#endif

/**
 * Reads, parses and returns the job configuration.
 *
 * @param argc Number of input arguments
 * @param argv Input argument list
 * @return Parsed configuration
 */
template<class T>
T parse_config(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Insufficient input arguments.  No configuration file.\n";
    exit(EX_USAGE);
  }
  return T(argv[1]);
}

int main(int argc, char *argv[]) {
  Logger logger;

  #ifdef USE_SERIAL
//  #if 0
    auto config_serial = parse_config<CpuConfig<Param,SerialMat>>(argc, argv);
    SerialPSO<SerialMat> serial(&config_serial);
    serial.fit();
    serial.printFinalResults();
    logger.writeResult(serial);
  #endif

  #ifdef USE_OPENMP
//  #if 0
    auto config_cpu_omp = parse_config<CpuConfig<Param,Param>>(argc, argv);
    std::vector<IntType> thread_counts = {1, 2, 4, 6, 8};
    for(auto n_t : thread_counts) {
      OpenmpPSO<Param> cpu_omp(&config_cpu_omp, n_t);
      cpu_omp.fit();
      cpu_omp.printFinalResults();
      logger.writeResult(cpu_omp);
      #ifdef USE_OPENMP_TEAMS
//      #if 0
        OpenmpTeamsPSO<Param> omp_teams(&config_cpu_omp, n_t);
        omp_teams.fit();
        omp_teams.printFinalResults();
        logger.writeResult(omp_teams);
      #endif
    }
  #endif

  #ifdef USE_OPENCL
//  #if 0
    auto config_opencl = parse_config<OpenClConfig<OpenClParam>>(argc, argv);
    OpenClPSO opencl_pso(&config_opencl);
    opencl_pso.fit();
    opencl_pso.printFinalResults();
    logger.writeResult(opencl_pso);
  #endif

  #ifdef USE_CUDA
//  #if 0
    auto config_cuda = parse_config<CudaConfig>(argc, argv);
    std::vector<bool> tf_vec = { false, true };
    for (auto sep_kernel : tf_vec ) {
      for (auto fast_rand : tf_vec ) {
        CudaPSO cuda_pso(&config_cuda, sep_kernel, fast_rand);
        cuda_pso.fit();
        cuda_pso.printFinalResults();
        logger.writeResult(cuda_pso);
      }
    }
  #endif
}
