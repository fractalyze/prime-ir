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

#include "benchmark/zkbench_reporter.h"

namespace mlir::prime_ir::benchmark {

ZkBenchReporter::ZkBenchReporter(std::string_view implementation,
                                 std::string_view version) {
  report_.metadata = zkbench::Metadata::Create(implementation, version);
}

bool ZkBenchReporter::ReportContext(
    const ::benchmark::BenchmarkReporter::Context& context) {
  // Context is already captured in metadata
  return true;
}

void ZkBenchReporter::ReportRuns(
    const std::vector<::benchmark::BenchmarkReporter::Run>& runs) {
  for (const auto& run : runs) {
    if (run.skipped != ::benchmark::internal::NotSkipped) {
      continue;  // Skip skipped benchmarks
    }

    zkbench::BenchmarkResult result;

    // Convert real_accumulated_time to latency
    // Google Benchmark reports in nanoseconds
    double time_ns = run.GetAdjustedRealTime();
    std::string time_unit;

    switch (run.time_unit) {
      case ::benchmark::kNanosecond:
        time_unit = "ns";
        break;
      case ::benchmark::kMicrosecond:
        time_unit = "us";
        break;
      case ::benchmark::kMillisecond:
        time_unit = "ms";
        break;
      case ::benchmark::kSecond:
        time_unit = "s";
        break;
      default:
        time_unit = "ns";
        time_ns = run.real_accumulated_time * 1e9 / run.iterations;
    }

    result.latency = zkbench::MetricValue::Create(time_ns, time_unit);
    result.iterations = static_cast<size_t>(run.iterations);

    // Use run_name as the benchmark name
    std::string bench_name = run.run_name.str();

    // Remove any suffix like "/iterations:X" for cleaner names
    auto slash_pos = bench_name.find('/');
    if (slash_pos != std::string::npos) {
      bench_name = bench_name.substr(0, slash_pos);
    }

    report_.benchmarks[bench_name] = result;
  }
}

void ZkBenchReporter::Finalize() {
  // Nothing to finalize
}

std::string ZkBenchReporter::ToJson() const { return report_.ToJson(); }

}  // namespace mlir::prime_ir::benchmark
