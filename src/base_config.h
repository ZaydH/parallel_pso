//
// Created by Zayd Hammoudeh on 10/24/20.
//

#ifndef SERIAL_BASE_CONFIG_H
#define SERIAL_BASE_CONFIG_H

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sysexits.h>

#include "csv/csv.h"
#include "types_general.h"

#define POS_LABEL 1
#define NEG_LABEL -1

template <class T, class L>
class BaseConfig {
 private:
  std::string config_path_;

  std::string data_path_;
  IntType n_iter_ = 10000;
  IntType n_part_ = 100;
  /** Learning rate */
  T lr_ = 1E-3;

  bool debug_ = false; // ToDo Disable debug mode

  T momentum_ = 0.9;
  T rate_point_ = 0.1;
  T rate_global_ = 0.1;

  T bound_lo_ = -5;
  T bound_hi_ = 5;

  void validateConfig() const {
    assert(this->bound_lo() < this->bound_hi());
    assert(this->momentum_ > 0 && this->momentum_ < 1);

    // Fields that must be strictly positive
    std::vector<T> pos_params = {this->lr(), this->rate_global(), this->rate_point()};
    for (auto field : pos_params) {
      assert(field > 0);
      _unused(field);
    }

    // Positive integers
    std::vector<IntType> pos_ints = {this->dim(), this->n_particle(), this->n_iter()};
    for (auto field : pos_params) {
      assert(field > 0);
      _unused(field);
    }
  }

  void parseConfigFile() {
    std::ifstream input_file(this->config_path_);
    if (!input_file) {
      std::cout << "Cannot open CNF file: " << this->config_path_ << std::endl;
      exit(EX_IOERR);
    }

    std::string line;
    while(std::getline(input_file, line)) {
      std::size_t loc = line.find(',');
      // Report error if comma not found
      if (loc == std::string::npos) {
        std::cerr << "File line \"" << line << "\" missing comma.\n";
        exit(EX_IOERR);
      }
      std::string flag = line.substr(0, loc);
      std::string val = line.substr(loc+1);
      if (flag == "bound_lo") {
        this->bound_lo_ = std::stod(val);
      } else if (flag == "bound_hi") {
        this->bound_hi_ = std::stod(val);
      } else if (flag == "pop") {
        this->n_part_ = std::stoull(val);
      } else if (flag == "n_iterations") {
        this->n_iter_ = std::stoull(val);
      } else if (flag == "dim") {
        this->dim_ = std::stoull(val);
      } else if (flag == "lr") {
        this->lr_ = std::stod(val);
      } else if (flag == "task") {
        this->task_name_ = val;
      } else if (flag == "data-path") {
        this->data_path_ = "../data/" + val;
      } else {
        std::cerr << "File line \"" << line << "\" could not be parsed\n";
        exit(EX_IOERR);
      }
    }

    // Parse and compare the task name
    if (this->is_ext_data()) {
      if (this->task_name_ == "breast-cancer") {
        this->parse_breast_cancer();
      } else if (this->task_name_ == "ionosphere") {
        this->parse_ionosphere();
      } else {
        throw std::runtime_error("Unknown data file to parse");
      }
    }

    input_file.close();
    if (this->task_name_.empty()) {
      std::cerr << "No task name specified in configuration file\n";
      exit(EX_IOERR);
    }
  }

  /** Parse the ionosphere dataset */
  void parse_ionosphere() {
    #define NEGATIVE_FIELD "b" 
    #define POSITIVE_FIELD "g" 

    std::ifstream fin; 
    fin.open(this->data_path_);
    if (!fin.is_open())
      throw std::runtime_error ("Failed to open ionosphere dataset");

    // Read line by line
    std::vector<T*> data;
    std::vector<IntType> labels;
    std::string line;
		while (getline(fin, line)) {
			std::stringstream ss ( line );

      T * features = new T[this->dim_ + 1];
      std::string field;
      for (int i = 0; i < this->dim_; i++) {
        getline(ss, field, ',');
        features[i] = atof(field.c_str());
      }
      features[this->dim_] = 1.; // Offset w0
      data.push_back(features);

      // Get the label
      getline(ss, field, ',');
      labels.push_back((field == NEGATIVE_FIELD) ? NEG_LABEL : POS_LABEL);
		}
    this->dim_++;
    this->convertDataAndLabels(data, labels);
  }

  void convertDataAndLabels(std::vector<T*> data, std::vector<IntType> labels) {
    // Make sure not a mismatch in data
    this->n_ele_ = data.size();
    assert(this->n_ele_ == (IntType)labels.size());
    // Allocate the memory
    this->data_ = new T[this->n_ele_ * this->dim_];
    this->labels_ = new T[this->n_ele_];
    // Transfer the data to a contiguous array
    for (IntType row = 0; row < this->n_ele_; row++) {
      IntType offset = row * this->dim_;
      T * row_data = data[row];

      for (IntType col = 0; col < this->dim_; col++)
        this->data_[offset + col] = row_data[col];

      delete row_data;
      this->labels_[row] = labels[row];
    }
  }

  /** Parse the breast cancer dataset */
  void parse_breast_cancer() {
    #define BENIGN_LABEL 2
    #define MALIGNANT_LABEL 4

    // Parameter is the number of columns
//    const unsigned ext_dim = this->dim_ + 2;
//    io::CSVReader<ext_dim> in(this->data_path_);
    io::CSVReader<11> in(this->data_path_);
//  in.read_header(io::ignore_extra_column, "vendor", "size", "speed");
    this->dim_++; // Add a dummy column for offset

    // Parameter values
    IntType raw_label;
    T radius, texture, perim, area, smoothness, compact, concavity, concav_pts, symmetry, frac_dim;
    // Store the values
    std::vector<T*> data;
    std::vector<IntType> labels;
    while(in.read_row(radius, texture, perim, area, smoothness, compact, concavity, concav_pts,
                      symmetry, frac_dim, raw_label)){
      // do stuff with the data
      T * features = new T[this->dim_];
      IntType loc = 0;
      features[loc++] = radius / 1.4E6;
      features[loc++] = texture / 10;
      features[loc++] = perim / 10;
      features[loc++] = area / 10;
      features[loc++] = smoothness / 10;
      features[loc++] = compact / 10;
      features[loc++] = concavity / 10;
      features[loc++] = concav_pts / 10;
      features[loc++] = symmetry / 10;
      features[loc++] = frac_dim / 5;
      features[loc++] = 1;  // Offset term
      data.push_back(features);

      // Standardize the label and then correct
      assert(raw_label == BENIGN_LABEL || raw_label == MALIGNANT_LABEL);
      IntType lbl = (raw_label == MALIGNANT_LABEL) ? POS_LABEL : NEG_LABEL;
      labels.push_back(lbl);
    }
    this->convertDataAndLabels(data, labels);
  }

 protected:
  /** Name of the tasks used in the parse function */
  std::string task_name_;
  /** Loss function */
  L loss_;
  /** Updates the loss function */
  virtual void parseTask() = 0;

  IntType dim_ = 500;
  IntType n_ele_ = 0;
  T * data_ = nullptr;
  T * labels_ = nullptr;

 public:
  explicit BaseConfig(const char *config_path) {
    this->config_path_ = config_path;
    this->parseConfigFile();
    this->validateConfig();
  }

  ~BaseConfig() {
    if (this->data_)
      delete this->data_;
    if (this->labels_)
      delete this->labels_;
  }

  L loss_func() const { return this->loss_; }

  /** Checks whether external data is used */
  bool is_ext_data() {
    return this->data_path_.size() > 0;
  }

  /** Minimum bound for the dimension of the points */
  T bound_lo() const { return this->bound_lo_; }

  /** Maximum bound for the dimension of the points */
  T bound_hi() const { return this->bound_hi_; }

  /** Velocity momentum */
  T momentum() const { return this->momentum_; }

  /** Rate of point's best */
  T rate_point() const { return this->rate_point_; }

  /** Rate of global best */
  T rate_global() const { return this->rate_global_; }

  /** Returns \p true if in debug mode */
  bool d() const { return this->debug_; }

  /** Accessor for the learning rate */
  T lr() const { return this->lr_; }

  /** Accessor for the number of iterations */
  IntType n_iter() const { return this->n_iter_; }

  /** Accessor for the number of particles */
  IntType n_particle() const { return this->n_part_; }

  /** Accessor for the dimension of the system */
  IntType dim() const { return this->dim_; }

  /** Accessor for the external data (if any) */
  virtual T* ext_data() const { return this->data_; }

  /** Accessor for the external labels (if any) */
  virtual T* ext_labels() const { return this->labels_; }

  /** Accessor for the number of elements in external data (if any) */
  IntType n_ele() const { return this->n_ele_; }

  /** Name of the task being performed */
  std::string task_name() const { return this->task_name_; }

  /** Path to an external data file (if applicable) */
  std::string data_file() const { return this->data_path_; }
};

#endif //SERIAL_BASE_CONFIG_H
