//
// Created by zayd on 11/9/20.
//

#ifndef CL_PSO_CL_CONFIG_H
#define CL_PSO_CL_CONFIG_H

#include "base_config.h"
#include "types_general.h"

template<class T>
class OpenClConfig : public BaseConfig<T,OpenClLoss> {
 protected:
  /** Updates the loss function */
  void parseTask() {
    this->loss_ = this->task_name_;
  }
 public:
  explicit OpenClConfig<T>(const char *config_path) : BaseConfig<T,OpenClLoss>(config_path) {
    parseTask();
  };
};


#endif //CL_PSO_CL_CONFIG_H
