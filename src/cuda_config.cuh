//
// Created by zayd on 10/26/20.
//

#ifndef SERIAL_CUDA_CONFIG_CUH
#define SERIAL_CUDA_CONFIG_CUH

#include <cuda.h>
#include <string>

#include "base_config.h"
#include "types_general.h"

CudaLoss parseCudaTask(const std::string &task);

class CudaConfig : public BaseConfig<CudaParam,CudaLoss> {
 protected:
  CudaMat cu_data_ = nullptr;
  CudaMat cu_labels_ = nullptr;

  /** Updates the loss function */
  void parseTask() {
    this->loss_ = parseCudaTask(this->task_name_);
  }

  void allocateExtDataMemory();

 public:
  explicit CudaConfig(const char *config_path) : BaseConfig<CudaParam,CudaLoss>(config_path) {
    parseTask();

    if (this->is_ext_data())
      this->allocateExtDataMemory();
  };

  ~CudaConfig();

  CudaMat ext_labels() const { return cu_labels_; }

  CudaMat ext_data() const { return cu_data_; }
};

#endif //SERIAL_CUDA_CONFIG_CUH
