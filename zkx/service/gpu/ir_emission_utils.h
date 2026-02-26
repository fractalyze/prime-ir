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

#ifndef ZKX_SERVICE_GPU_IR_EMISSION_UTILS_H_
#define ZKX_SERVICE_GPU_IR_EMISSION_UTILS_H_

#include <stdint.h>

#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/utils/hlo_traversal.h"
#include "zkx/literal.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
inline constexpr int64_t kMinDimensionToTransposeTiled = 16;
// But if both swap dimensions are larger than 'kMinDimensionToTransposeTiled2',
// and the product of the dimensions to be swapped is larger than
// 'kMinTotalDimensionsToTransposeTiled', tiled transposition may be more
// efficient.
inline constexpr int64_t kMinDimensionToTransposeTiled2 = 8;
inline constexpr int64_t kMinTotalDimensionsToTransposeTiled = 64 * 128;
// As the amount of shared memory is limited, we need to make sure that we don't
// detect 102 transposes that would require too much bytes for the most minor
// dimension.
inline constexpr int64_t kMaxBytesInMostMinorDimension = 8;

inline constexpr int64_t WarpSize(
    const se::DeviceDescription& gpu_device_info) {
  return gpu_device_info.threads_per_warp();
}

// Fusions that implemented with pre-compiled device kernels have
// FusionBackendConfig.kind equal to this string.
inline constexpr std::string_view kCustomFusionKind = "__custom_fusion";

// Generic fusions that use Triton have FusionBackendConfig.kind equal to this
// string. This fusion kind will eventually subsume all usages of
// kTritonGemmFusionKind and kTritonSoftmaxFusionKind.
inline constexpr std::string_view kTritonFusionKind = "__triton";

// Fusions that use Triton have FusionBackendConfig.kind equal to this string.
inline constexpr std::string_view kTritonGemmFusionKind = "__triton_gemm";

inline constexpr std::string_view kUncompilableFusion = "__uncompilable_fusion";

// CUB library calls.
// Reference: https://nvlabs.github.io/cub/
inline constexpr std::string_view kCubDeviceRadixSortTarget =
    "__cub$DeviceRadixSort";

// Returns true if the instruction is a CUB DeviceRadixSort custom call.
bool IsCubDeviceRadixSort(const HloInstruction& hlo);

// Returns true if `instr` is a non-strided slice.
bool IsSliceWithUnitStrides(const HloInstruction* instr);

// Returns true if `instr` is a slice (or dynamic slice) instruction and
// operates on a contiguous slice of the input buffer.
bool IsContiguousSlice(const HloInstruction& instr);

absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    const BufferAssignment& buffer_assignment, const HloInstruction* instr,
    const ShapeIndex& index);

// Returns whether the fusion represented by 'fusion_adaptor' can be emitted
// with the dynamic update slice in-place emitter. If 'fusion_adaptor'
// represents a single fusion computation, 'fusion' should provide the fusion
// instruction corresponding to that fusion computation. 'get_allocation_slice'
// is a callback for getting the allocated buffer slice, given an instruction
// and a shape index. This is ignored in case 'fusion' is a nullptr.
absl::StatusOr<bool> CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
    const HloFusionAdaptor& fusion_adaptor,
    std::function<absl::StatusOr<BufferAllocation::Slice>(
        const HloInstruction* instr, const ShapeIndex& index)>
        get_allocation_slice,
    const HloInstruction* fusion = nullptr);

// Returns the first hero instruction reachable from `instr` as root. Hero
// instruction can be in a different computation if the parent HloFusionAdaptor
// is a producer-consumer fusion.
HloInstructionAdaptor FindNonTrivialHero(const HloInstructionAdaptor& instr);

// Same as above, but fusion is the parent computation of the hlo instruction.
const HloInstruction& FindNonTrivialHero(const HloInstruction& instr);

// Description of how to emit a given transposition.
struct TransposeDescription {
  // Transpose instruction.
  const HloInstruction* instr;

  // Normalized transpose dimensions.
  absl::InlinedVector<int64_t, 3> dimensions;

  // Permutations of normalized transpose dimensions.
  absl::InlinedVector<int64_t, 3> permutation;

  TransposeDescription(absl::InlinedVector<int64_t, 3> dimensions,
                       absl::InlinedVector<int64_t, 3> permutation)
      : TransposeDescription(/*instr=*/nullptr, dimensions, permutation) {}

  TransposeDescription(const HloInstruction* instr,
                       absl::InlinedVector<int64_t, 3> dimensions,
                       absl::InlinedVector<int64_t, 3> permutation)
      : instr(instr), dimensions(dimensions), permutation(permutation) {}

  // Transpose instruction input shape.
  const Shape& input_shape() const { return instr->operand(0)->shape(); }

  // Returns true, if both descriptions have the same dimensions and
  // permutation, even if they're produced by different instructions.
  bool IsEquivalent(const TransposeDescription& other) const {
    return dimensions == other.dimensions && permutation == other.permutation;
  }
};

std::optional<TransposeDescription> GetDescriptionForTiledTransposeEmitter(
    const HloInstruction& hero);

// Returns the dynamic-update-slice instructions defining the results of a
// fusion node. A dynamic slice update is said to be "defining" of a result if
// that result is the output of a dynamic slice update, or if that result is the
// output of a bitcast of a dynamic slice update---since such bitcast may be
// handled as a no-op.
std::vector<HloInstructionAdaptor> GetOutputDefiningDynamicUpdateSlices(
    absl::Span<HloInstructionAdaptor const> roots);

// Checks if the instruction is elementwise.
bool IsIntermediate(const HloInstruction* instr, int allowed_operand_count = 1);

// This class stores either a non-owning reference or owns data that represents
// a dense array in ZKX format. It is used for intermediate storage during IR
// constant emission.
class DenseDataIntermediate {
 public:
  // Creates an instance of DenseDataIntermediate that owns the provided vector.
  static DenseDataIntermediate Own(std::vector<uint8_t> owned) {
    DenseDataIntermediate di;
    di.data_ = std::move(owned);
    return di;
  }

  // Creates an instance of DenseDataIntermediate that aliases the input.
  static DenseDataIntermediate Alias(absl::Span<const uint8_t> aliased) {
    DenseDataIntermediate di;
    di.data_ = aliased;
    return di;
  }

  // Returns a reference to the data this object represents.
  absl::Span<const uint8_t> span() const {
    return data_.index() == 0 ? absl::Span<const uint8_t>(std::get<0>(data_))
                              : std::get<1>(data_);
  }

 private:
  std::variant<std::vector<uint8_t>, absl::Span<const uint8_t>> data_;
};

absl::StatusOr<DenseDataIntermediate> LiteralToZkxFormat(
    const Literal& literal);

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_IR_EMISSION_UTILS_H_
