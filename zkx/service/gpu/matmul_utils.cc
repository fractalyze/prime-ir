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

#include "zkx/service/gpu/matmul_utils.h"

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/shape_util.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"

namespace zkx::gpu {

absl::StatusOr<Shape> GetBatchRowColumnShape(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> row_dims, absl::Span<const int64_t> col_dims) {
  TF_RET_CHECK(shape.has_layout());

  std::vector<int64_t> minor_to_major;
  for (size_t i = 0; i < shape.rank();) {
    // The GeMM output always has its layout set such that the batch, row, and
    // col dim groups are each laid out physically sequentially. GeMM operands
    // must, therefore, be laid out similarly.
    auto check_physically_sequential =
        [&](absl::Span<const int64_t> dims) -> absl::Status {
      for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        // NOTE: `i` is incremented as we check the dimensions.
        if (*it != shape.layout().minor_to_major()[i++])
          return absl::InvalidArgumentError("dims not physically_sequential");
      }
      return absl::OkStatus();
    };

    int64_t dim = shape.layout().minor_to_major()[i];
    if (!row_dims.empty() && dim == row_dims.back()) {
      minor_to_major.push_back(1);
      TF_RETURN_IF_ERROR(check_physically_sequential(row_dims));
    } else if (!col_dims.empty() && dim == col_dims.back()) {
      minor_to_major.push_back(2);
      TF_RETURN_IF_ERROR(check_physically_sequential(col_dims));
    } else if (!batch_dims.empty() && (dim == batch_dims.back())) {
      minor_to_major.push_back(0);
      TF_RETURN_IF_ERROR(check_physically_sequential(batch_dims));
    } else {
      return absl::InvalidArgumentError("dims not physically sequential");
    }
  }

  if (col_dims.empty()) minor_to_major.push_back(2);
  if (row_dims.empty()) minor_to_major.push_back(1);
  if (batch_dims.empty()) minor_to_major.push_back(0);

  auto dim_size = [&](absl::Span<const int64_t> dims) {
    return absl::c_accumulate(dims, 1, [&](int64_t size, int64_t dim) {
      return size * shape.dimensions(dim);
    });
  };

  return ShapeUtil::MakeShapeWithDenseLayout(
      shape.element_type(),
      {dim_size(batch_dims), dim_size(row_dims), dim_size(col_dims)},
      minor_to_major);
}

// Returns the matrix layout for a logical shape (batch, rows, columns).
/*static*/ absl::StatusOr<MatrixLayout> MatrixLayout::For(const Shape& shape) {
  TF_RET_CHECK(shape.rank() == 3);
  TF_RET_CHECK(shape.has_layout());

  int64_t batch_size = shape.dimensions(0);
  int64_t num_rows = shape.dimensions(1);
  int64_t num_cols = shape.dimensions(2);

  Order order{Order::kRowMajor};
  int64_t leading_dim_stride = num_cols;
  int64_t batch_stride = num_rows * num_cols;

  // `MatrixLayout`, like BLAS, uses only two strides, so either the row or
  // column must be contiguous in memory (i.e. most minor physical dimension).
  absl::Span<const int64_t> minor_to_major = shape.layout().minor_to_major();
  switch (64 * minor_to_major[2] + 8 * minor_to_major[1] + minor_to_major[0]) {
    case 012:  // (B,R,C) (major-to-minor)
      break;
    case 021:  // (B,C,R)
      order = Order::kColumnMajor;
      leading_dim_stride = num_rows;
      break;
    case 0102:  // (R,B,C)
      leading_dim_stride = batch_size * num_cols;
      batch_stride = num_cols;
      break;
    case 0201:  // (C,B,R)
      order = Order::kColumnMajor;
      leading_dim_stride = batch_size * num_rows;
      batch_stride = num_rows;
      break;
    default:
      return absl::UnimplementedError("batch in most minor dimension");
  }

  if (batch_size == 1) {
    batch_stride = 0;
  }
  return MatrixLayout{
      shape.element_type(), num_rows,           num_cols,    order,
      batch_size,           leading_dim_stride, batch_stride};
}

/*static*/ absl::StatusOr<MatrixLayout> MatrixLayout::For(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> row_dims, absl::Span<const int64_t> col_dims) {
  TF_ASSIGN_OR_RETURN(
      Shape batch_row_col_shape,
      GetBatchRowColumnShape(shape, batch_dims, row_dims, col_dims));
  return MatrixLayout::For(batch_row_col_shape);
}

/*static*/ absl::StatusOr<MatrixLayout> MatrixLayout::For(
    const Shape& shape, size_t lhs_num_batch_dims, size_t lhs_num_row_dims,
    size_t rhs_num_batch_dims, size_t rhs_num_col_dims) {
  size_t num_batch_dims = std::max(lhs_num_batch_dims, rhs_num_batch_dims);

  TF_RET_CHECK(shape.rank() ==
               num_batch_dims + lhs_num_row_dims + rhs_num_col_dims);

  std::vector<int64_t> dims(shape.rank());
  absl::c_iota(dims, 0);

  auto batch_dims = absl::Span<const int64_t>(dims).first(num_batch_dims);
  auto row_dims =
      absl::Span<const int64_t>(dims).subspan(num_batch_dims, lhs_num_row_dims);
  auto col_dims = absl::Span<const int64_t>(dims).last(rhs_num_col_dims);

  return MatrixLayout::For(shape, batch_dims, row_dims, col_dims);
}

}  // namespace zkx::gpu
