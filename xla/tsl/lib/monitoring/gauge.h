/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

// Stub implementation of tsl::monitoring::Gauge.
// TODO(monitoring): Port the full TSL monitoring infrastructure if metrics
// collection is needed.
// Reference: xla/tsl/lib/monitoring/gauge.h

#ifndef XLA_TSL_LIB_MONITORING_GAUGE_H_
#define XLA_TSL_LIB_MONITORING_GAUGE_H_

#include <atomic>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"

namespace tsl {
namespace monitoring {

// Stub GaugeCell that stores values but doesn't report to any monitoring
// backend.
template <typename T>
class GaugeCell {
 public:
  GaugeCell() : value_(T{}) {}
  ~GaugeCell() = default;

  void Set(const T& value) { value_ = value; }

  T value() const { return value_; }

 private:
  T value_;

  GaugeCell(const GaugeCell&) = delete;
  void operator=(const GaugeCell&) = delete;
};

// Explicit specialization for int64_t using atomic.
template <>
class GaugeCell<int64_t> {
 public:
  GaugeCell() : value_(0) {}
  ~GaugeCell() = default;

  void Set(int64_t value) { value_.store(value, std::memory_order_relaxed); }

  int64_t value() const { return value_.load(std::memory_order_relaxed); }

 private:
  std::atomic<int64_t> value_;

  GaugeCell(const GaugeCell&) = delete;
  void operator=(const GaugeCell&) = delete;
};

// Explicit specialization for bool using atomic.
template <>
class GaugeCell<bool> {
 public:
  GaugeCell() : value_(false) {}
  ~GaugeCell() = default;

  void Set(bool value) { value_.store(value, std::memory_order_relaxed); }

  bool value() const { return value_.load(std::memory_order_relaxed); }

 private:
  std::atomic<bool> value_;

  GaugeCell(const GaugeCell&) = delete;
  void operator=(const GaugeCell&) = delete;
};

// Stub Gauge that provides the expected API but does not register with any
// monitoring system.
template <typename ValueType, int NumLabels>
class Gauge {
 public:
  ~Gauge() = default;

  // Creates a new Gauge. The metric_def_args are accepted but ignored since
  // this is a stub implementation.
  template <typename... MetricDefArgs>
  static Gauge* New(MetricDefArgs&&... /*metric_def_args*/) {
    return new Gauge();
  }

  // Returns a cell for the given labels. Labels are accepted but ignored
  // in this stub - all calls return the same default cell.
  template <typename... Labels>
  GaugeCell<ValueType>* GetCell(const Labels&... /*labels*/) {
    return &default_cell_;
  }

 private:
  Gauge() = default;

  GaugeCell<ValueType> default_cell_;

  Gauge(const Gauge&) = delete;
  void operator=(const Gauge&) = delete;
};

}  // namespace monitoring
}  // namespace tsl

#endif  // XLA_TSL_LIB_MONITORING_GAUGE_H_
