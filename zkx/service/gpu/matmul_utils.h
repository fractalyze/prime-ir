/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_GPU_MATMUL_UTILS_H_
#define ZKX_SERVICE_GPU_MATMUL_UTILS_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/shape.h"

namespace zkx::gpu {

// Returns a (batch, rows, columns) shape derived from the input shape by
// combining batch, row and column dimensions into single dimensions.
// Returns an error if the dimensions are not physically sequential in the
// layout.
absl::StatusOr<Shape> GetBatchRowColumnShape(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> row_dims, absl::Span<const int64_t> col_dims);

// Describes the memory layout of a matrix for GEMM operations.
struct MatrixLayout {
  enum class Order {
    kRowMajor,     // Elements in the same row are contiguous in memory.
    kColumnMajor,  // Elements in the same column are contiguous in memory.
  };

  PrimitiveType dtype;
  int64_t num_rows;
  int64_t num_cols;
  Order order;
  int64_t batch_size = 1;
  int64_t leading_dim_stride;
  int64_t batch_stride = 0;

  // Returns the matrix layout for a logical shape (batch, rows, columns).
  static absl::StatusOr<MatrixLayout> For(const Shape& shape);

  // Returns the matrix layout with the given batch, row, col dimensions.
  static absl::StatusOr<MatrixLayout> For(const Shape& shape,
                                          absl::Span<const int64_t> batch_dims,
                                          absl::Span<const int64_t> row_dims,
                                          absl::Span<const int64_t> col_dims);

  // Returns the matrix layout for the output.
  static absl::StatusOr<MatrixLayout> For(const Shape& shape,
                                          size_t lhs_num_batch_dims,
                                          size_t lhs_num_row_dims,
                                          size_t rhs_num_batch_dims,
                                          size_t rhs_num_col_dims);
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_MATMUL_UTILS_H_
