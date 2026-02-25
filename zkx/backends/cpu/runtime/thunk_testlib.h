/* Copyright 2024 The OpenXLA Authors.
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

#ifndef ZKX_BACKENDS_CPU_RUNTIME_THUNK_TESTLIB_H_
#define ZKX_BACKENDS_CPU_RUNTIME_THUNK_TESTLIB_H_

#include <stddef.h>

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"

#include "xla/tsl/concurrency/async_value_ref.h"
#include "zkx/backends/cpu/runtime/buffer_allocations.h"
#include "zkx/backends/cpu/runtime/resource_use.h"
#include "zkx/backends/cpu/runtime/thunk.h"
#include "zkx/literal.h"
#include "zkx/runtime/buffer_use.h"
#include "zkx/service/buffer_assignment.h"

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"

namespace zkx::cpu {

//===----------------------------------------------------------------------===//
// A set of helper functions to create buffer allocations from Literals.
//===----------------------------------------------------------------------===//

// Creates a BufferAllocation with given index from a literal.
BufferAllocation CreateBufferAllocation(size_t index, const Literal& literal);

// Creates an array of BufferAllocations from a variadic pack of literals.
template <
    typename... Literals,
    std::enable_if_t<std::conjunction_v<std::is_same<Literals, Literal>...>>* =
        nullptr>
std::array<BufferAllocation, sizeof...(Literals)> CreateBufferAllocation(
    Literals&... literals) {
  size_t index = 0;
  return {CreateBufferAllocation(index++, literals)...};
}

// Creates a BufferAllocation::Slice that covers the entire allocation.
BufferAllocation::Slice CreateBufferAllocationSlice(
    const BufferAllocation& allocation);

// Creates a BufferAllocation::Slice that covers a subrange of the allocation.
BufferAllocation::Slice CreateBufferAllocationSlice(
    const BufferAllocation& allocation, int64_t offset, int64_t size);

// Creates an array of BufferAllocation::Slice from a pack of allocations. Each
// slice covers the entire corresponding allocation.
template <typename... BufferAllocations,
          std::enable_if_t<std::conjunction_v<
              std::is_same<BufferAllocations, BufferAllocation>...>>* = nullptr>
std::array<BufferAllocation::Slice, sizeof...(BufferAllocations)>
CreateBufferAllocationSlice(const BufferAllocations&... allocations) {
  return {CreateBufferAllocationSlice(allocations)...};
}

// Creates a BufferAllocations from a span of literals.
BufferAllocations CreateBufferAllocations(absl::Span<Literal*> literals);

// Creates a BufferAllocations from a variadic pack of literals.
template <
    typename... Literals,
    std::enable_if_t<std::conjunction_v<std::is_same<Literals, Literal>...>>* =
        nullptr>
BufferAllocations CreateBufferAllocations(Literals&... literals) {
  std::vector<Literal*> literals_ptrs = {&literals...};
  return CreateBufferAllocations(absl::MakeSpan(literals_ptrs));
}

//===----------------------------------------------------------------------===//
// A library of test-only thunks.
//===----------------------------------------------------------------------===//

// A test-only thunk to create a Thunk with a specific buffer use.
class BufferUseThunk : public Thunk {
 public:
  explicit BufferUseThunk(BufferUse buffer_use)
      : Thunk(Kind::kKernel, {"buffer-use"}), buffer_use_(buffer_use) {}

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final {
    return absl::UnimplementedError("Unimplemented");
  }

  BufferUses buffer_uses() const final { return {buffer_use_}; }

 private:
  BufferUse buffer_use_;
};

// A test-only thunk to create a Thunk with a specific resource use.
class ResourceUseThunk : public Thunk {
 public:
  explicit ResourceUseThunk(ResourceUse resource_use)
      : Thunk(Kind::kKernel, {"resource-use"}), resource_use_(resource_use) {}

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final {
    return absl::UnimplementedError("Unimplemented");
  }

  BufferUses buffer_uses() const final { return {}; }
  ResourceUses resource_uses() const final { return {resource_use_}; }

 private:
  ResourceUse resource_use_;
};

void InitSharedResource();
int64_t GetSharedResource();

// A test-only thunk for verifying thunk executor implementation:
//
//   dst += src (for all srcs and dsts slices)
//
// We generate random thunk sequences reading and writing different slices of
// the same buffer, and check that at run time it does not lead to any data
// races and produces expected result.
class AddI32Thunk final : public Thunk {
 public:
  AddI32Thunk(std::string name, std::vector<BufferAllocation::Slice> srcs,
              std::vector<BufferAllocation::Slice> dsts,
              std::vector<std::string>* trace, bool use_shared_resource,
              bool inject_error)
      : Thunk(Kind::kKernel, Info{name}),
        srcs_(std::move(srcs)),
        dsts_(std::move(dsts)),
        trace_(trace),
        use_shared_resource_(use_shared_resource),
        inject_error_(inject_error) {}

  static std::unique_ptr<Thunk> Create(
      std::string name, std::vector<BufferAllocation::Slice> srcs,
      std::vector<BufferAllocation::Slice> dsts,
      std::vector<std::string>* trace = nullptr,
      bool use_shared_resource = false, bool inject_error = false) {
    return std::make_unique<AddI32Thunk>(std::move(name), std::move(srcs),
                                         std::move(dsts), trace,
                                         use_shared_resource, inject_error);
  }

  // Executes `dst += src` for a single src/dst pair.
  static absl::Status Execute(const BufferAllocations* allocations,
                              BufferAllocation::Slice src_slice,
                              BufferAllocation::Slice dst_slice);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final;

  BufferUses buffer_uses() const final;
  ResourceUses resource_uses() const final;

 private:
  std::vector<BufferAllocation::Slice> srcs_;
  std::vector<BufferAllocation::Slice> dsts_;
  std::vector<std::string>* trace_;  // not owned
  bool use_shared_resource_;
  bool inject_error_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_RUNTIME_THUNK_TESTLIB_H_
