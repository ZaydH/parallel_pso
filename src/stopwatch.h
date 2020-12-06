//
// Created by Zayd Hammoudeh on 10/25/20.
//

#ifndef SERIAL_STOPWATCH_H
#define SERIAL_STOPWATCH_H

#include <cinttypes>
#include <sys/time.h> // NOLINT(modernize-deprecated-headers)

#include "types_general.h"

class StopWatch {
  /**
   * Used to track time events in the execution of the solver.
   *
   * Built using the internal C++ timer (i.e., timeval()).  Resolution down to microseconds
   * via the "tv_usec" property.
   *
   * @see timeval
   */
 public:
  explicit StopWatch() {
    interval_length_.tv_sec = 60;
    gettimeofday(&last_interval_start_, nullptr);
    start_time_ = stop_time_ = last_interval_start_;
  }

  bool start() {
    auto ret = static_cast<bool>(gettimeofday(&last_interval_start_, nullptr));
    start_time_ = stop_time_ = last_interval_start_;
    return !ret;
  }
  /**
   * Records and saves the stop time independent of the time zone.
   *
   * @return 0 for success, or -1 for failure (in which case errno is set appropriately).
   */
  bool stop() {
    return gettimeofday(&stop_time_, nullptr) == 0;
  }

  double getElapsedSeconds() {
    timeval r = getElapsedTime();
    return r.tv_sec + static_cast<double>(r.tv_usec) / 1000000;
  }

 private:
  timeval start_time_;
  timeval stop_time_;

  IntType time_bound_ = INT32_MAX;

  /**
   * Interval used to separate events.  It is set by default to 60s.
   *
   * For instance, this is used to set how often
   */
  timeval interval_length_;
  timeval last_interval_start_;

  /**
   * if we have started and then stopped the watch, this returns
   * the elapsed time.  Otherwise, time elapsed from start_time_
   * till now is returned
   */
  timeval getElapsedTime() {
    timeval other_time = stop_time_;
    if (stop_time_.tv_sec == start_time_.tv_sec
        && stop_time_.tv_usec == start_time_.tv_usec)
      gettimeofday(&other_time, nullptr);
    long int ad = 0;
    unsigned int bd = 0;

    if (other_time.tv_usec < start_time_.tv_usec) {
      ad = 1;
      bd = 1000000;
    }
    timeval r = (struct timeval) {0};
    r.tv_sec = other_time.tv_sec - ad - start_time_.tv_sec;
    r.tv_usec = other_time.tv_usec + bd - start_time_.tv_usec;
    return r;
  }
};


#endif //SERIAL_STOPWATCH_H
