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

#ifndef ZKX_SERVICE_POSEIDON2_REFERENCE_H_
#define ZKX_SERVICE_POSEIDON2_REFERENCE_H_

#include <vector>

#include "zkx/literal.h"
#include "zkx/literal_util.h"

namespace zkx::gpu {

// m4 = circ(2, 3, 1, 1) — the base circulant matrix for Poseidon2 MDS.
constexpr int kM4[4][4] = {
    {2, 3, 1, 1},
    {1, 2, 3, 1},
    {1, 1, 2, 3},
    {3, 1, 1, 2},
};

// Full MDS entry: M[i][j] = m4[i % 4][j % 4] * (1 + δ(i / 4, j / 4)).
constexpr int MdsEntry(int i, int j) {
  int base = kM4[i % 4][j % 4];
  int same_group = (i / 4 == j / 4) ? 1 : 0;
  return base * (1 + same_group);
}

// Computes reference Poseidon2 permutation using native field arithmetic.
// Templated on NativeT (e.g., BabyBearMont, KoalaBearMont, GoldilocksMont).
//
// This is useful for verifying GPU emitter output against a known-correct
// scalar implementation.
template <typename NativeT>
Literal ComputeReferencePoseidon2(
    const Literal& input, const Literal& ext_init_rc, const Literal& int_rc,
    const Literal& ext_term_rc, const Literal& diag, int width, int ext_rounds,
    int int_rounds, int sbox_degree) {
  // Load state.
  std::vector<NativeT> state(width);
  for (int i = 0; i < width; ++i) {
    state[i] = input.Get<NativeT>({i});
  }

  // MDS diffusion.
  auto mds = [&](std::vector<NativeT>& s) {
    std::vector<NativeT> out(width, NativeT(0));
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < width; ++j) {
        out[i] = out[i] + s[j] * NativeT(MdsEntry(i, j));
      }
    }
    s = out;
  };

  // S-box.
  auto sbox = [&](NativeT x) { return x.Pow(sbox_degree); };

  // Initial MDS.
  mds(state);

  // Initial external rounds.
  for (int r = 0; r < ext_rounds; ++r) {
    for (int i = 0; i < width; ++i) {
      state[i] = sbox(state[i] + ext_init_rc.Get<NativeT>({r, i}));
    }
    mds(state);
  }

  // Internal rounds.
  for (int r = 0; r < int_rounds; ++r) {
    state[0] = state[0] + int_rc.Get<NativeT>({r});
    state[0] = sbox(state[0]);
    // Internal diffusion: state = diag * state + sum(state).
    NativeT total_sum(0);
    for (int i = 0; i < width; ++i) {
      total_sum = total_sum + state[i];
    }
    for (int i = 0; i < width; ++i) {
      state[i] = diag.Get<NativeT>({i}) * state[i] + total_sum;
    }
  }

  // Terminal external rounds.
  for (int r = 0; r < ext_rounds; ++r) {
    for (int i = 0; i < width; ++i) {
      state[i] = sbox(state[i] + ext_term_rc.Get<NativeT>({r, i}));
    }
    mds(state);
  }

  // Build output literal.
  return LiteralUtil::CreateR1<NativeT>(state);
}

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_POSEIDON2_REFERENCE_H_
