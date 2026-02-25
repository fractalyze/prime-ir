/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/util.h"

#include <functional>
#include <numeric>
#include <tuple>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

#include "xla/tsl/platform/numbers.h"

namespace zkx {

absl::Status AddStatus(absl::Status prior, std::string_view context) {
  CHECK(!prior.ok());
  return absl::Status{prior.code(),
                      absl::StrCat(context, ": ", prior.message())};
}

absl::Status AppendStatus(absl::Status prior, std::string_view context) {
  CHECK(!prior.ok());
  return absl::Status{prior.code(),
                      absl::StrCat(prior.message(), ": ", context)};
}

PaddingConfig MakeNoPaddingConfig(int64_t rank) {
  PaddingConfig padding_config;
  for (int64_t dnum = 0; dnum < rank; ++dnum) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(0);
    // TODO(chokobole): Do we need this? Dependency: interior_padding
    // dimension->set_interior_padding(0);
  }
  return padding_config;
}

namespace {
std::string HumanReadableNumOps(double flops, double nanoseconds,
                                std::string_view op_prefix) {
  if (nanoseconds == 0) {
    return absl::StrCat("NaN ", op_prefix, "OP/s");
  }
  double nano_flops = flops / nanoseconds;
  std::string throughput =
      tsl::strings::HumanReadableNum(static_cast<int64_t>(nano_flops * 1e9));
  std::string_view sp(throughput);
  // Use the more common "G(FLOPS)", rather than "B(FLOPS)"
  if (absl::EndsWith(sp, "B") ||  // Ends in 'B', ignoring case
      absl::EndsWith(sp, "b")) {
    *throughput.rbegin() = 'G';
  }
  throughput += absl::StrCat(op_prefix, "OP/s");
  return throughput;
}
}  // namespace

std::string HumanReadableNumFlops(double flops, double nanoseconds) {
  return HumanReadableNumOps(flops, nanoseconds, "FL");
}

int64_t Product(absl::Span<const int64_t> xs) {
  return std::accumulate(xs.begin(), xs.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

absl::InlinedVector<std::pair<int64_t, int64_t>, 8> CommonFactors(
    absl::Span<const int64_t> a, absl::Span<const int64_t> b) {
  CHECK_EQ(Product(a), Product(b));
  absl::InlinedVector<std::pair<int64_t, int64_t>, 8> bounds;
  if (absl::c_equal(a, b)) {
    bounds.reserve(a.size() + 1);
    for (int64_t i = 0; i <= a.size(); ++i) {
      bounds.emplace_back(i, i);
    }
    return bounds;
  }
  int64_t i = 0, j = 0, prior_i = -1, prior_j = -1;
  while (i < a.size() && j < b.size() && a[i] == b[j]) {
    std::tie(prior_i, prior_j) = std::make_pair(i, j);
    bounds.emplace_back(i, j);
    ++i;
    ++j;
  }
  // If the product is different after filtering out zeros, return full group.
  // E.g.,:
  // a={0, 10 ,3}
  //       ^
  //      i=1
  //
  // b={0, 3}
  //       ^
  //      j=1
  if (Product(a.subspan(i)) != Product(b.subspan(j))) {
    return {std::make_pair(0, 0), std::make_pair(a.size(), b.size())};
  }
  if (0 == Product(a.subspan(i))) {
    bounds.push_back(std::make_pair(i, j));
    bounds.push_back(std::make_pair(a.size(), b.size()));
    return bounds;
  }

  for (int64_t partial_size_a = 1, partial_size_b = 1;;) {
    if (partial_size_a == partial_size_b && (i > prior_i || j > prior_j)) {
      std::tie(prior_i, prior_j) = std::make_pair(i, j);
      bounds.emplace_back(i, j);
      continue;
    }
    bool in_bounds_i = i < a.size();
    bool in_bounds_j = j < b.size();
    if (!(in_bounds_i || in_bounds_j)) {
      break;
    }
    bool next_a =
        partial_size_a < partial_size_b ||
        (in_bounds_i &&
         (!in_bounds_j || (partial_size_a == partial_size_b && a[i] <= b[j])));
    bool next_b =
        partial_size_b < partial_size_a ||
        (in_bounds_j &&
         (!in_bounds_i || (partial_size_b == partial_size_a && b[j] <= a[i])));
    if (next_a) {
      partial_size_a *= a[i];
      ++i;
    }
    if (next_b) {
      partial_size_b *= b[j];
      ++j;
    }
  }
  return bounds;
}

DimensionVector GetNonContractingDims(
    int64_t rank, absl::Span<const int64_t> contracting_dim_numbers,
    absl::Span<const int64_t> batch_dim_numbers) {
  DimensionVector non_contracting_dim_numbers;
  for (int64_t i = 0; i < rank; ++i) {
    if (!absl::c_linear_search(contracting_dim_numbers, i) &&
        !absl::c_linear_search(batch_dim_numbers, i)) {
      non_contracting_dim_numbers.push_back(i);
    }
  }
  return non_contracting_dim_numbers;
}

std::string SanitizeFileName(std::string_view old_file_name) {
  std::string file_name = std::string(old_file_name);
  for (char& c : file_name) {
    if (c == '/' || c == '\\' || c == '[' || c == ']' || c == ' ') {
      c = '_';
    }
  }
  return file_name;
}

}  // namespace zkx
