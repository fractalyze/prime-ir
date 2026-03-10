/* Copyright 2026 The ZKX Authors.

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

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "icicle/curves/curve_config.h"

#include "zkx/backends/gpu/runtime/msm/msm_runner.h"

using affine_t = curve_config::affine_t;
using scalar_t = curve_config::scalar_t;
using projective_t = curve_config::projective_t;

namespace {

bool AffineEqual(const affine_t& a, const affine_t& b) {
  return std::memcmp(&a, &b, sizeof(affine_t)) == 0;
}

void PrintAffine(const std::string& label, const affine_t& p) {
  uint64_t x_lo, y_lo;
  std::memcpy(&x_lo, &p.x, sizeof(x_lo));
  std::memcpy(&y_lo, &p.y, sizeof(y_lo));
  std::cout << label << ": x_lo=0x" << std::hex << x_lo << " y_lo=0x" << y_lo
            << std::dec << std::endl;
}

int RunMsm(int n, const scalar_t* h_scalars, const affine_t* h_bases,
           affine_t* h_result, cudaStream_t stream) {
  void *d_scalars, *d_bases, *d_result;
  cudaMalloc(&d_scalars, n * sizeof(scalar_t));
  cudaMalloc(&d_bases, n * sizeof(affine_t));
  cudaMalloc(&d_result, sizeof(affine_t));

  cudaMemcpyAsync(d_scalars, h_scalars, n * sizeof(scalar_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_bases, h_bases, n * sizeof(affine_t),
                  cudaMemcpyHostToDevice, stream);

  const char* error_msg = nullptr;
  int ret = zkx::gpu::RunBn254G1Msm(d_scalars, d_bases, n, d_result, stream,
                                    /*window_bits=*/0,
                                    /*scalars_montgomery=*/false,
                                    /*points_montgomery=*/false,
                                    /*result_montgomery=*/false, &error_msg);
  if (ret != 0) {
    std::cerr << "MSM failed: " << (error_msg ? error_msg : "unknown")
              << std::endl;
    cudaFree(d_scalars);
    cudaFree(d_bases);
    cudaFree(d_result);
    return ret;
  }

  cudaMemcpyAsync(h_result, d_result, sizeof(affine_t), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_scalars);
  cudaFree(d_bases);
  cudaFree(d_result);
  return 0;
}

// Test: Run MSM of size n, then split into 2 halves and add results.
// MSM(s,b) should equal MSM(s[:n/2],b[:n/2]) + MSM(s[n/2:],b[n/2:]).
bool TestAdditiveSplit(int n, cudaStream_t stream) {
  std::cout << "TestAdditiveSplit(n=" << n << ")..." << std::endl;

  std::vector<scalar_t> scalars(n);
  std::vector<affine_t> bases(n);
  scalar_t::rand_host_many(scalars.data(), n);
  projective_t::rand_host_many(bases.data(), n);

  // Full MSM.
  affine_t full_result;
  if (RunMsm(n, scalars.data(), bases.data(), &full_result, stream) != 0) {
    return false;
  }

  // Split MSM.
  int half = n / 2;
  affine_t half1_result, half2_result;
  if (RunMsm(half, scalars.data(), bases.data(), &half1_result, stream) != 0) {
    return false;
  }
  if (RunMsm(n - half, scalars.data() + half, bases.data() + half,
             &half2_result, stream) != 0) {
    return false;
  }

  // Add the two halves on host via ICICLE projective arithmetic.
  auto proj1 = projective_t::from_affine(half1_result);
  auto proj2 = projective_t::from_affine(half2_result);
  auto sum_proj = proj1 + proj2;
  auto sum_affine = projective_t::to_affine(sum_proj);

  if (!AffineEqual(full_result, sum_affine)) {
    PrintAffine("full", full_result);
    PrintAffine("split_sum", sum_affine);
    std::cerr << "FAIL: AdditiveSplit mismatch for n=" << n << std::endl;
    return false;
  }

  std::cout << "PASS" << std::endl;
  return true;
}

// Test: Run n_calls consecutive MSMs of the given size.
// Verifies no crashes (ICICLE race/OOM) on repeated calls.
bool TestConsecutiveMsms(int msm_size, int n_calls, cudaStream_t stream) {
  std::cout << "TestConsecutiveMsms(size=" << msm_size << ", calls=" << n_calls
            << ")..." << std::endl;

  std::vector<scalar_t> scalars(msm_size);
  std::vector<affine_t> bases(msm_size);
  scalar_t::rand_host_many(scalars.data(), msm_size);
  projective_t::rand_host_many(bases.data(), msm_size);

  affine_t first_result;
  for (int i = 0; i < n_calls; ++i) {
    affine_t result;
    if (RunMsm(msm_size, scalars.data(), bases.data(), &result, stream) != 0) {
      std::cerr << "FAIL: call " << i << " failed" << std::endl;
      return false;
    }
    if (i == 0) {
      first_result = result;
    } else if (!AffineEqual(result, first_result)) {
      PrintAffine("first", first_result);
      PrintAffine("call_" + std::to_string(i), result);
      std::cerr << "FAIL: non-deterministic at call " << i << std::endl;
      return false;
    }
  }

  std::cout << "PASS" << std::endl;
  return true;
}

}  // namespace

int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  bool all_pass = true;

  // Additive split correctness at various sizes.
  all_pass &= TestAdditiveSplit(1'000'000, stream);
  all_pass &= TestAdditiveSplit(4'000'000, stream);

  // Consecutive calls — the key regression test for ICICLE cleanup race.
  all_pass &= TestConsecutiveMsms(2'000'000, 5, stream);
  all_pass &= TestConsecutiveMsms(4'000'000, 5, stream);
  all_pass &= TestConsecutiveMsms(5'000'000, 3, stream);

  cudaStreamDestroy(stream);

  if (all_pass) {
    std::cout << "\nAll tests PASSED." << std::endl;
    return 0;
  } else {
    std::cerr << "\nSome tests FAILED." << std::endl;
    return 1;
  }
}
