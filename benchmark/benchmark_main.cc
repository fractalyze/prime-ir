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

#include <cstring>
#include <fstream>
#include <iostream>

#include "benchmark/benchmark.h"
#include "benchmark/zkbench_reporter.h"

namespace {

// Command line flag for zkbench JSON output
const char* g_zkbench_output_file = nullptr;

bool ParseZkBenchFlag(int* argc, char** argv) {
  for (int i = 1; i < *argc; ++i) {
    if (strncmp(argv[i], "--zkbench_out=", 14) == 0) {
      g_zkbench_output_file = argv[i] + 14;
      // Remove this argument from argv
      for (int j = i; j < *argc - 1; ++j) {
        argv[j] = argv[j + 1];
      }
      --(*argc);
      --i;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  // Parse zkbench-specific flags before Google Benchmark
  ParseZkBenchFlag(&argc, argv);

  // Initialize Google Benchmark
  ::benchmark::Initialize(&argc, argv);

  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  // Run with console reporter (default)
  ::benchmark::RunSpecifiedBenchmarks();

  // If zkbench output is requested, run again with our reporter
  if (g_zkbench_output_file != nullptr) {
    mlir::prime_ir::benchmark::ZkBenchReporter reporter("prime-ir", "1.0.0");
    ::benchmark::RunSpecifiedBenchmarks(&reporter);

    std::ofstream ofs(g_zkbench_output_file);
    if (ofs.is_open()) {
      ofs << reporter.ToJson() << std::endl;
      std::cerr << "zkbench JSON written to: " << g_zkbench_output_file
                << std::endl;
    } else {
      std::cerr << "Error: Could not open zkbench output file: "
                << g_zkbench_output_file << std::endl;
      return 1;
    }
  }

  ::benchmark::Shutdown();
  return 0;
}
