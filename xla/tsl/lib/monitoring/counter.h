/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_TSL_LIB_MONITORING_COUNTER_H_
#define XLA_TSL_LIB_MONITORING_COUNTER_H_

#include <atomic>
#include <cstdint>

#include "absl/status/status.h"

// We replace this implementation with a null implementation for mobile
// platforms.
// TODO(chokobole): Uncomment this. Dependency: Gauge
// #ifdef IS_MOBILE_PLATFORM

namespace tsl::monitoring {

// CounterCell which has a null implementation.
class CounterCell {
 public:
  CounterCell() {}
  ~CounterCell() {}

  void IncrementBy(int64_t step) {}
  int64_t value() const { return 0; }

 private:
  CounterCell(const CounterCell&) = delete;
  void operator=(const CounterCell&) = delete;
};

// Counter which has a null implementation.
template <int NumLabels>
class Counter {
 public:
  ~Counter() {}

  template <typename... MetricDefArgs>
  static Counter* New(MetricDefArgs&&... metric_def_args) {
    return new Counter<NumLabels>();
  }

  template <typename... Labels>
  CounterCell* GetCell(const Labels&... labels) {
    return &default_counter_cell_;
  }

  absl::Status GetStatus() { return absl::OkStatus(); }

 private:
  Counter() {}

  CounterCell default_counter_cell_;

  Counter(const Counter&) = delete;
  void operator=(const Counter&) = delete;
};

}  // namespace tsl::monitoring

#endif  // XLA_TSL_LIB_MONITORING_COUNTER_H_
