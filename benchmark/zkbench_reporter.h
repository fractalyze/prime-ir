/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_BENCHMARK_ZKBENCH_REPORTER_H_
#define PRIME_IR_BENCHMARK_ZKBENCH_REPORTER_H_

#include <string>

#include "benchmark/benchmark.h"
#include "zkbench/schema.h"

namespace mlir::prime_ir::benchmark {

/// Custom Google Benchmark reporter that outputs zkbench-compatible JSON.
class ZkBenchReporter : public ::benchmark::BenchmarkReporter {
 public:
  ZkBenchReporter(std::string_view implementation, std::string_view version);

  bool ReportContext(const Context& context) override;
  void ReportRuns(const std::vector<Run>& report) override;
  void Finalize() override;

  const zkbench::BenchmarkReport& GetReport() const { return report_; }
  std::string ToJson() const;

 private:
  zkbench::BenchmarkReport report_;
};

}  // namespace mlir::prime_ir::benchmark

#endif  // PRIME_IR_BENCHMARK_ZKBENCH_REPORTER_H_
