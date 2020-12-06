//
// Created by zayd on 11/9/20.
//

#ifndef SERIAL_OPENCL_PSO_H
#define SERIAL_OPENCL_PSO_H

#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
  #include "OpenCL/cl.hpp"
#else
  #include <CL/cl.hpp>
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "base_pso.h"
#include "cl_config.h"
#include "types_general.h"

#define CL_PSO_KERNELS_FILE "opencl_z_pso.cl"


class OpenClPSO : public BasePSO<OpenClParam, OpenClLoss> {
 private:
  ClInt tot_len_;

  cl::Platform platform_;

  cl::Device device_;
  cl::Context context_;
  /* Device particle information */
  cl::Buffer parts_;
  /* Device velocity information */
  cl::Buffer velos_;

  cl::Buffer rng_;

  cl::Buffer ext_data_;
  cl::Buffer ext_labels_;

  cl::Program prog_;
  cl::Program::Sources sources_;

  cl::CommandQueue queue_;

  OpenClParam *best_;

  /** Generates the random number seeds */
  void generateRng() {
    // Device buffer of random number seeds defined
    this->rng_ = cl::Buffer(this->context_, CL_MEM_READ_WRITE, tot_len_ * sizeof(ClInt));

    std::seed_seq seq{time(nullptr)};
    std::vector<ClInt> seeds(this->tot_len_);
    seq.generate(seeds.begin(), seeds.end());

    this->uploadHostData<ClInt>(tot_len_, seeds.data(), this->rng_);
  }

  /** Helper method to download data from the host */
  void duplicateBuffer(ClInt &n_bytes, cl::Buffer &dest, cl::Buffer &src) {
    if (queue_.enqueueCopyBuffer(src, dest, 0, 0, n_bytes) != CL_SUCCESS)
      throw std::runtime_error("*** Error: Failed to copy data");
    queue_.finish();
  }

  /** Helper method to download data from the host */
  template<typename T>
  void uploadHostData(ClInt n, T * data, cl::Buffer &buf) {
    if (queue_.enqueueWriteBuffer(buf, CL_TRUE, 0, n * sizeof(T), data) != CL_SUCCESS)
      throw std::runtime_error("*** Error: Failed to write to device");
    queue_.finish();
  }

  /** Helper method to download data from the host */
  void downloadDeviceData(ClInt n_bytes, OpenClParam * data, cl::Buffer &buf) {
    if(queue_.enqueueReadBuffer(buf, CL_TRUE, 0, n_bytes, data) != CL_SUCCESS)
      throw std::runtime_error("*** Error: Failed to enqueue read buffer");
    queue_.finish();
  }

  /** Initialize the particle location and velocity */
  void initializeParticleInfo() {
    this->parts_ = cl::Buffer(this->context_, CL_MEM_READ_WRITE, tot_len_ * sizeof(OpenClParam));
    initRand(this->parts_, this->config_->bound_lo(), this->config_->bound_hi());

    this->velos_ = cl::Buffer(this->context_, CL_MEM_READ_WRITE, tot_len_ * sizeof(OpenClParam));
    OpenClParam v_diff = this->config_->bound_hi() - this->config_->bound_lo();
    initRand(this->velos_, -v_diff, v_diff);
  }

  void initRand(cl::Buffer &buf, OpenClParam bound_lo, OpenClParam bound_hi) {
    cl::Event event;

    OpenClParam diff = bound_hi - bound_lo;
    cl::Kernel kern(this->prog_, "initVec");
    ClInt argc = 0;
    kern.setArg(argc++, this->rng_);
    kern.setArg(argc++, buf);
    kern.setArg(argc++, bound_lo);
    kern.setArg(argc++, diff);
    this->queue_.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(this->tot_len_),
                                      cl::NullRange, NULL, &event);
    this->queue_.finish();
  }

  void downloadBestInfo(cl::Buffer &best, cl::Buffer &best_loss) {
    downloadDeviceData(this->config_->dim() * sizeof(OpenClParam), this->best_, best);
    downloadDeviceData(sizeof(OpenClParam), &this->best_loss_, best_loss);
  }

  /** Updates the velocity and position using the PSO algorithm. Also performs clipping */
  void updatePosAndVelos(cl::Buffer &pos_best, cl::Buffer &best, cl::Buffer &r_p, cl::Buffer &r_g) {
    cl::Event events[3];

    cl::Kernel upd_kern(this->prog_, "updatePosAndVelo");
    ClInt argc = 0;
    upd_kern.setArg(argc++, this->parts_);
    upd_kern.setArg(argc++, this->velos_);
    upd_kern.setArg(argc++, r_p);
    upd_kern.setArg(argc++, r_g);
    upd_kern.setArg(argc++, pos_best);
    upd_kern.setArg(argc++, best); // best
    upd_kern.setArg(argc++, this->config_->momentum());
    upd_kern.setArg(argc++, this->config_->rate_point());
    upd_kern.setArg(argc++, this->config_->rate_global());
    upd_kern.setArg(argc++, this->config_->lr());
    upd_kern.setArg(argc++, this->config_->dim());
    this->queue_.enqueueNDRangeKernel(upd_kern, cl::NullRange, cl::NDRange(this->tot_len_),
                                      cl::NullRange, NULL, events + 0);
    this->queue_.finish();

    // Clip the positions and velocity values
    OpenClParam lo, hi;
    for (int i = 0; i < 2; i++) {
      if (i == 0) {
        lo = this->config_->bound_lo();
        hi = this->config_->bound_hi();
      } else {
        hi = (hi - lo);
        lo = -hi;
      }
      cl::Kernel clip_kern(this->prog_, "clip");
      argc = 0;
      clip_kern.setArg(argc++, (i == 0) ? this->parts_ : this->velos_);
      clip_kern.setArg(argc++, lo);
      clip_kern.setArg(argc++, hi);
      this->queue_.enqueueNDRangeKernel(clip_kern, cl::NullRange, cl::NDRange(this->tot_len_),
                                        cl::NullRange, NULL, events + (i + 1));
    }
    this->queue_.finish();
  }

  void calcLoss(cl::Buffer &loss_buf) {
    cl::Event events;

    std::string loss = this->config_->is_ext_data() ? "ext_data" : this->config_->loss_func();
    cl::Kernel loss_kern(this->prog_, loss.c_str());
    ClInt argc = 0;
    loss_kern.setArg(argc++, loss_buf);
    loss_kern.setArg(argc++, this->parts_);
    if (this->config_->is_ext_data()) {
      loss_kern.setArg(argc++, this->ext_data_);
      loss_kern.setArg(argc++, this->ext_labels_);
      loss_kern.setArg(argc++, this->config_->n_ele());
    }
    loss_kern.setArg(argc++, this->config_->dim());
    this->queue_.enqueueNDRangeKernel(loss_kern, cl::NullRange,
                                      cl::NDRange(this->config_->n_particle()),
                                      cl::NullRange, NULL, &events);
    this->queue_.finish();
  }

  void updateBest(cl::Buffer &part_loss, cl::Buffer &pos_best, cl::Buffer &pos_best_loss,
                  cl::Buffer &best, cl::Buffer &best_loss) {
    cl::Event events[2];

    cl::Kernel pos_best_kern(this->prog_, "updatePosBest");
    ClInt argc = 0;
    pos_best_kern.setArg(argc++, this->parts_);
    pos_best_kern.setArg(argc++, part_loss);
    pos_best_kern.setArg(argc++, pos_best);
    pos_best_kern.setArg(argc++, pos_best_loss);
    pos_best_kern.setArg(argc++, this->config_->dim());
    this->queue_.enqueueNDRangeKernel(pos_best_kern, cl::NullRange,
                                      cl::NDRange(this->config_->n_particle()),
                                      cl::NullRange, NULL, events + 0);
    this->queue_.finish();

    cl::Kernel best_kern(this->prog_, "updateTotBest");
    argc = 0;
    best_kern.setArg(argc++, this->parts_);
    best_kern.setArg(argc++, pos_best_loss);
    best_kern.setArg(argc++, best);
    best_kern.setArg(argc++, best_loss);
    best_kern.setArg(argc++, this->config_->n_particle());
    best_kern.setArg(argc++, this->config_->dim());
    this->queue_.enqueueNDRangeKernel(best_kern, cl::NullRange, cl::NDRange(1),
                                      cl::NullRange, NULL, events + 1);
    this->queue_.finish();
  }

  /** Reads an OpenCL file and loads it into a string object */
  static std::string loadFile(const std::string &FileName) {
    std::ifstream File;
    File.open(FileName);
    if (!File.is_open())
      throw std::runtime_error("*** Error: File " + FileName + " doesn't exist");

    std::stringstream StringStream;
    StringStream << File.rdbuf();
    return StringStream.str();
  }

 public:
  explicit OpenClPSO(OpenClConfig<OpenClParam> *config)
    : BasePSO<OpenClParam, OpenClLoss>(config), tot_len_(config->n_particle() * config->dim())
  {
    this->best_ = new OpenClParam[this->config_->dim()];

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(platforms.size() == 1); // One platform available
    this->platform_ = platforms[0];
    // Print information about the platforms
    if (config_->d()) {
      std::cout << "Platform number is: " << platforms.size() << std::endl;
      std::string platformVendor;
      for (auto & platform : platforms) {
        platform.getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
        std::cout << "Platform is by: " << platformVendor << std::endl;
      }
    }

    cl_context_properties properties[] =
      {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[0])(),
        0
      };
    this->context_ = cl::Context(CL_DEVICE_TYPE_ALL, properties);

    std::vector<cl::Device> devices = this->context_.getInfo<CL_CONTEXT_DEVICES>();
    #if defined(__APPLE__) || defined(__MACOSX)
      bool found = false;
      for (unsigned int i = 0; i < devices.size(); ++i) {
        std::string str(devices[i].getInfo<CL_DEVICE_NAME>());
        if (strstr(str.c_str(), "AMD") != nullptr) {
          this->device_ = devices[i];
          std::cout << "Apple Device: " << str << std::endl;
          found = true;
          break;
        }
      }
      assert(found);
      _unused(found);
    #else
      assert(devices.size() == 1);  //
      this->device_ = devices[0];
    #endif
    this->queue_ = cl::CommandQueue(this->context_, this->device_);
    if (config_->d()) {
      for (unsigned int i = 0; i < devices.size(); ++i)
        std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    std::string file_text = loadFile(CL_PSO_KERNELS_FILE);
    this->sources_.push_back({file_text.c_str(), file_text.length()});
    this->prog_ = cl::Program(this->context_, this->sources_);
    prog_.build(devices);

    this->generateRng();
    this->initializeParticleInfo();
  }

  ~OpenClPSO() {
    delete this->best_;
  }

  void fit_() {
    // Best positions same as starting positions
    ClInt tot_n_bytes = tot_len_ * sizeof(OpenClParam);
    cl::Buffer pos_best(this->context_, CL_MEM_READ_WRITE, tot_n_bytes);
    this->duplicateBuffer(tot_n_bytes, pos_best, this->parts_);

    // Upload the external linear-in-parameter data to the GPU
    if (this->config_->is_ext_data()) {
      IntType data_len = this->config_->n_ele() * this->config_->dim();
      this->ext_data_ = cl::Buffer(this->context_, CL_MEM_READ_WRITE, data_len * sizeof(OpenClParam));
      this->uploadHostData<OpenClParam>(data_len, this->config_->ext_data(), this->ext_data_);

      this->ext_labels_ = cl::Buffer(this->context_, CL_MEM_READ_WRITE,
                                     data_len * sizeof(OpenClParam));
      this->uploadHostData<OpenClParam>(data_len, this->config_->ext_labels(), this->ext_labels_);
    }

    // Calculate initial losses
    cl::Buffer pos_best_loss(this->context_, CL_MEM_READ_WRITE,
                             this->config_->n_particle() * sizeof(OpenClParam));
    this->calcLoss(pos_best_loss);

    cl::Buffer tmp_res(this->context_, CL_MEM_READ_WRITE,
                       this->config_->n_particle() * sizeof(OpenClParam));

    // Construct
    cl::Buffer best(this->context_, CL_MEM_READ_WRITE, this->config_->dim() * sizeof(OpenClParam));
    cl::Buffer best_loss(this->context_, CL_MEM_READ_WRITE, sizeof(OpenClParam));
    this->uploadHostData(1, &this->best_loss_, best_loss);

    // Randomize point and global movement
    cl::Buffer r_p(this->context_, CL_MEM_READ_WRITE, tot_len_ * sizeof(OpenClParam));
    cl::Buffer r_g(this->context_, CL_MEM_READ_WRITE, tot_len_ * sizeof(OpenClParam));

    this->updateBest(pos_best_loss, pos_best, pos_best_loss, best, best_loss);
    if (this->config_->d()) {
      this->downloadBestInfo(best, best_loss);
      this->printBest(0);
    }

    for (ClInt iter = 0; iter < config_->n_iter(); iter++) {
      // Initial vector of random positions
      this->initRand(r_p, 0, 1);
      this->initRand(r_g, 0, 1);

      this->updatePosAndVelos(pos_best, best, r_p, r_g);

      this->calcLoss(tmp_res);
      this->updateBest(tmp_res, pos_best, pos_best_loss, best, best_loss);

      // Print the current best part
      if (this->config_->d()) {
        this->downloadBestInfo(best, best_loss);
        this->printBest(iter);
      }
    }
    this->downloadBestInfo(best, best_loss);
  }

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
  std::string name() const override {
    std::stringstream ss;
    ss << "OpenCL";
    return ss.str();
  }
};

#endif //SERIAL_OPENCL_PSO_H
