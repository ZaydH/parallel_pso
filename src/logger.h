//
// Created by zayd on 10/28/20.
//

#ifndef SERIAL_LOGGER_H
#define SERIAL_LOGGER_H

#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sysexits.h>

#include "base_pso.h"

inline char separator() {
  #ifdef _WIN32
    return '\\';
  #else
    return '/';
  #endif
}

class Logger {

 private:
  time_t start_time_;
  /** Store the time buffer */
  std::string time_str_;
  /** Output file where the results are written */
  std::ofstream f_out_;
  /** Results directory */
  const std::string RES_DIR_ = "res";

  /** Stores the start time of the experiment */
  void recordStartTime() {
    time (&this->start_time_);
    struct tm * timeinfo = localtime(&this->start_time_);

    char time_buf_[100];
    strftime(time_buf_, sizeof(time_buf_),"%Y-%m-%d-%H-%M-%S", timeinfo);
    this->time_str_ = std::string(time_buf_);
  }

  /** Creates the results file including the header line */
  void createOutputFile() {
    struct stat info;
    stat(this->RES_DIR_.c_str(), &info);
    if ((info.st_mode & S_IFDIR) == 0) {
      std::cerr << "Error: Results directory \"" << this->RES_DIR_ << "\" does not exist. "
                << "Exiting." << std::endl;
      exit(EX_IOERR);
    }

    // Create an output file
    std::stringstream ss;
    ss << this->RES_DIR_ << separator() << "res_" << this->time_str_ << ".csv";
    this->f_out_.open(ss.str().c_str());

    this->createHeaderLine();
  }

  /** Creates the file header line */
  void createHeaderLine() {
    f_out_ << "time" << ",name" << ",task-name" 
           << ",dim" << ",debug-mode"
           << ",num-iterations" << ",num-particles" 
           << ",bound-low" << ",bound-high"
           << ",learning-rate" << ",momentum"
           << ",rate-global" << ",rate-point" 
           << ",exec-time" << ",best-loss" << "\n";
  }

 public:
  explicit Logger() {
    this->recordStartTime();
    std::cout << "==================  " << "Job Started at: " << this->time_str_
              << "  ==================" << std::endl;
    this->createOutputFile();
  }

  ~Logger() {
    this->f_out_.close();
  }

  template<class T, class L>
  void writeResult(BasePSO<T,L> &pso) {
    BaseConfig<T,L>* config = pso.config();

    f_out_ << this->time_str_ << "," << pso.name() << "," << config->task_name() << "," 
           << config->dim() << "," << config->d() << ","
           << config->n_iter() << "," << config->n_particle() << ","
           << config->bound_lo() << "," << config->bound_hi() << ","
           << config->lr() << "," << config->momentum() << ","
           << config->rate_global() << "," << config->rate_point() << ","
           << pso.exec_time() << "," << pso.best_loss() << "\n";
  }
};

#endif //SERIAL_LOGGER_H
