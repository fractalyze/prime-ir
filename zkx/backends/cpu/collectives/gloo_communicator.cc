/* Copyright 2023 The OpenXLA Authors.
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

#include "zkx/backends/cpu/collectives/gloo_communicator.h"

#include <cstring>
#include <vector>

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "gloo/algorithm.h"
#include "gloo/allgather.h"
#include "gloo/allreduce.h"
#include "gloo/math.h"
#include "gloo/reduce_scatter.h"
#include "gloo/transport/unbound_buffer.h"
#include "gloo/types.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/all_types.h"
#include "zkx/backends/cpu/collectives/cpu_collectives.h"
#include "zkx/core/collectives/reduction_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/collective_ops_utils.h"
#include "zkx/status_macros.h"

namespace gloo {

template <typename T>
void EcPointSum(void* c_, const void* a_, const void* b_, size_t n) {
  absl::Span<T> c = absl::MakeSpan(static_cast<T*>(c_), n);
  absl::Span<const T> a = absl::MakeConstSpan(static_cast<const T*>(a_), n);
  absl::Span<const T> b = absl::MakeConstSpan(static_cast<const T*>(b_), n);
  if constexpr (zk_dtypes::IsAffinePoint<T>) {
    using JacobianPoint = typename T::JacobianPoint;
    std::vector<JacobianPoint> jacobian_points;
    jacobian_points.reserve(n);
    for (auto i = 0; i < n; i++) {
      jacobian_points.push_back(a[i] + b[i]);
    }
    CHECK(JacobianPoint::BatchToAffine(jacobian_points, &c).ok());
  } else {
    for (auto i = 0; i < n; i++) {
      c[i] = a[i] + b[i];
    }
  }
}

template <typename T>
void EcPointSum(T* a, const T* b, size_t n) {
  EcPointSum<T>(a, a, b, n);
}

#define ADD_EC_REDUCTION_FUNCTION(cpp_type, ...)                        \
  template <>                                                           \
  class ReductionFunction<cpp_type> {                                   \
   public:                                                              \
    using Function = void(cpp_type*, const cpp_type*, size_t n);        \
                                                                        \
    static const ReductionFunction<cpp_type>* sum;                      \
                                                                        \
    ReductionFunction(ReductionType type, Function* fn)                 \
        : type_(type), fn_(fn) {}                                       \
                                                                        \
    ReductionType type() const { return type_; }                        \
                                                                        \
    void call(cpp_type* x, const cpp_type* y, size_t n) const {         \
      fn_(x, y, n);                                                     \
    }                                                                   \
                                                                        \
   protected:                                                           \
    ReductionType type_;                                                \
    Function* fn_;                                                      \
  };                                                                    \
                                                                        \
  const ReductionFunction<cpp_type>* ReductionFunction<cpp_type>::sum = \
      new ReductionFunction<cpp_type>(SUM, &EcPointSum<cpp_type>);

ZK_DTYPES_ALL_AFFINE_POINT_TYPE_LIST(ADD_EC_REDUCTION_FUNCTION)

#undef ADD_EC_REDUCTION_FUNCTION

}  // namespace gloo

namespace zkx::cpu {
namespace {

static constexpr uint8_t kCollectivePermuteSlotPrefix = 0x40;

template <typename T>
absl::Status SetAllReduceOptions(ReductionKind reduction_kind,
                                 se::DeviceMemoryBase input_buffer,
                                 se::DeviceMemoryBase output_buffer,
                                 size_t num_elements,
                                 gloo::AllreduceOptions& options) {
  options.setInput(
      reinterpret_cast<T*>(const_cast<void*>(input_buffer.opaque())),
      num_elements);
  options.setOutput(
      reinterpret_cast<T*>(const_cast<void*>(output_buffer.opaque())),
      num_elements);

  using ReductionFn = void (*)(void*, const void*, const void*, size_t);

  struct Selector {
    static auto sum() {
      if constexpr (zk_dtypes::IsEcPoint<T>)
        return static_cast<ReductionFn>(&gloo::EcPointSum<T>);
      else
        return static_cast<ReductionFn>(&gloo::sum<T>);
    }

    static auto product() {
      return static_cast<ReductionFn>(&gloo::product<T>);
    }

    static auto min() { return static_cast<ReductionFn>(&gloo::min<T>); }

    static auto max() { return static_cast<ReductionFn>(&gloo::max<T>); }
  };

  absl::StatusOr<ReductionFn> fn_or =
      SelectReduction<T, ReductionFn>(reduction_kind, Selector{});
  if (!fn_or.ok()) return fn_or.status();
  options.setReduceFunction(fn_or.value());
  return absl::OkStatus();
}

template <typename T>
absl::Status ReduceScatterHelper(std::shared_ptr<gloo::Context> context,
                                 ReductionKind reduction_kind, void* buffer,
                                 size_t chunk_elems) {
  const gloo::ReductionFunction<T>* reduction_function = nullptr;

  struct Selector {
    static auto sum() { return gloo::ReductionFunction<T>::sum; }

    static auto product() { return gloo::ReductionFunction<T>::product; }

    static auto min() { return gloo::ReductionFunction<T>::min; }

    static auto max() { return gloo::ReductionFunction<T>::max; }
  };

  absl::StatusOr<const gloo::ReductionFunction<T>*> fn_or =
      SelectReduction<T, const gloo::ReductionFunction<T>*>(reduction_kind,
                                                            Selector{});
  if (!fn_or.ok()) return fn_or.status();
  reduction_function = fn_or.value();

  try {
    std::vector<int> recv_elems(context->size, chunk_elems);
    gloo::ReduceScatterHalvingDoubling<T> algorithm(
        context, std::vector<T*>{reinterpret_cast<T*>(buffer)},
        chunk_elems * context->size, recv_elems, reduction_function);
    algorithm.run();
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo reduce-scatter failed: ", e.what()));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status GlooCommunicator::AllReduce(se::DeviceMemoryBase send_buffer,
                                         se::DeviceMemoryBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         ReductionKind reduction_kind,
                                         const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));

  gloo::AllreduceOptions options(context_);
  // TODO(phawkins): how to do tags?
  // options.setTag(tag);
  switch (dtype) {
    case S8:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<int8_t>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case PRED:
    case U8:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<uint8_t>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case S16:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<int16_t>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case U16:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<uint16_t>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case S32:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<int32_t>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case U32:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<uint32_t>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case S64:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<int64_t>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case U64:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<uint64_t>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case S128:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<s128>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case S256:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<s256>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case U128:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<u128>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
    case U256:
      TF_RETURN_IF_ERROR(SetAllReduceOptions<u256>(
          reduction_kind, send_buffer, recv_buffer, count, options));
      break;
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2)             \
  case enum:                                                        \
    TF_RETURN_IF_ERROR(SetAllReduceOptions<cpp_type>(               \
        reduction_kind, send_buffer, recv_buffer, count, options)); \
    break;
      ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      return absl::InvalidArgumentError("Unknown datatype in allreduce");
  }
  options.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING);
  options.setTimeout(absl::ToChronoMilliseconds(cpu_executor->timeout()));

  try {
    gloo::allreduce(options);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo all-reduce failed: ", e.what()));
  }
  return absl::OkStatus();
}

absl::Status GlooCommunicator::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  uint32_t tag = 0;  // TODO(phawkins): come up with better tags.
  const auto slot = gloo::Slot::build(kCollectivePermuteSlotPrefix, tag);

  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  size_t num_bytes = count * primitive_util::ByteWidth(dtype);

  try {
    std::unique_ptr<gloo::transport::UnboundBuffer> in;
    std::unique_ptr<gloo::transport::UnboundBuffer> out;
    for (RankId target : target_ranks) {
      if (target != context_->rank) {
        VLOG(1) << "send from " << context_->rank << " to " << target.value();
        if (!in) {
          in = context_->createUnboundBuffer(send_buffer.opaque(), num_bytes);
        }
        in->send(target.value(), slot);
      }
    }
    if (source_rank) {
      if (*source_rank == context_->rank) {
        std::memcpy(recv_buffer.opaque(), send_buffer.opaque(), num_bytes);
      } else {
        VLOG(1) << "recv at " << context_->rank << " from "
                << source_rank->value();
        out = context_->createUnboundBuffer(recv_buffer.opaque(), num_bytes);
        out->recv(source_rank->value(), slot);
      }
    } else {
      std::memset(recv_buffer.opaque(), 0, num_bytes);
    }
    VLOG(1) << "wait for send at " << context_->rank;
    auto deadline = absl::ToChronoTime(absl::Now() + cpu_executor->timeout());
    if (in) {
      in->waitSend(deadline);
    }
    VLOG(1) << "wait for recv at " << context_->rank;
    if (out) {
      out->waitRecv(deadline);
    }
    VLOG(1) << "done waiting at " << context_->rank;
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo collective-permute failed: ", e.what()));
  }
  return absl::OkStatus();
}

absl::Status GlooCommunicator::AllToAll(
    absl::Span<const se::DeviceMemoryBase> send_buffers,
    absl::Span<const se::DeviceMemoryBase> recv_buffers, PrimitiveType dtype,
    size_t count, const Executor& executor) {
  // We can't use Gloo's all-to-all implementation directly because it assumes
  // that the inputs and outputs are contiguous. No big deal; it's just built
  // on top of send/recv and we can do the same as it.
  uint32_t tag = 0;  // TODO(phawkins): use better tags.
  int my_rank = context_->rank;
  int world_size = context_->size;

  TF_RET_CHECK(world_size == send_buffers.size());
  TF_RET_CHECK(world_size == recv_buffers.size());

  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  size_t chunk_bytes = count * primitive_util::ByteWidth(dtype);

  try {
    const auto slot = gloo::Slot::build(gloo::kAlltoallSlotPrefix, tag);
    std::vector<std::unique_ptr<gloo::transport::UnboundBuffer>> ins(
        context_->size);
    std::vector<std::unique_ptr<gloo::transport::UnboundBuffer>> outs(
        context_->size);
    for (size_t i = 0; i < world_size; ++i) {
      if (i != my_rank) {
        ins[i] = context_->createUnboundBuffer(
            const_cast<void*>(send_buffers[i].opaque()), chunk_bytes);
        outs[i] = context_->createUnboundBuffer(
            const_cast<void*>(recv_buffers[i].opaque()), chunk_bytes);
      }
    }

    for (int i = 1; i < world_size; i++) {
      int send_rank = (my_rank + i) % world_size;
      int recv_rank = (my_rank + world_size - i) % world_size;
      ins[send_rank]->send(send_rank, slot);
      outs[recv_rank]->recv(recv_rank, slot);
    }

    std::memcpy(const_cast<void*>(recv_buffers[my_rank].opaque()),
                send_buffers[my_rank].opaque(), chunk_bytes);

    auto deadline = absl::ToChronoTime(absl::Now() + cpu_executor->timeout());
    for (int i = 0; i < world_size; i++) {
      if (i != my_rank) {
        ins[i]->waitSend(deadline);
        outs[i]->waitRecv(deadline);
      }
    }
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo all-to-all failed: ", e.what()));
  }
  return absl::OkStatus();
}

absl::Status GlooCommunicator::AllGather(se::DeviceMemoryBase send_buffer,
                                         se::DeviceMemoryBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         const Executor& executor) {
  uint32_t tag = 0;  // TODO(phawkins): use better tags.

  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  size_t chunk_bytes = count * primitive_util::ByteWidth(dtype);

  gloo::AllgatherOptions options(context_);
  options.setTag(tag);
  options.setTimeout(absl::ToChronoMilliseconds(cpu_executor->timeout()));
  options.setInput(reinterpret_cast<char*>(send_buffer.opaque()), chunk_bytes);
  options.setOutput(reinterpret_cast<char*>(recv_buffer.opaque()),
                    chunk_bytes * context_->size);

  try {
    gloo::allgather(options);
  } catch (std::exception& e) {
    return absl::UnknownError(
        absl::StrCat("Gloo all-gather failed: ", e.what()));
  }
  return absl::OkStatus();
}

absl::Status GlooCommunicator::ReduceScatter(se::DeviceMemoryBase send_buffer,
                                             se::DeviceMemoryBase recv_buffer,
                                             PrimitiveType dtype, size_t count,
                                             ReductionKind reduction_kind,
                                             const Executor& executor) {
  size_t chunk_bytes = count * primitive_util::ByteWidth(dtype);
  std::unique_ptr<char[]> temp(new char[chunk_bytes * context_->size]);
  std::memcpy(temp.get(), send_buffer.opaque(), chunk_bytes * context_->size);
  switch (dtype) {
    case S8:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<int8_t>(context_, reduction_kind,
                                                     temp.get(), count));
      break;
    case PRED:
    case U8:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<uint8_t>(context_, reduction_kind,
                                                      temp.get(), count));
      break;
    case S16:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<int16_t>(context_, reduction_kind,
                                                      temp.get(), count));
      break;
    case U16:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<uint16_t>(context_, reduction_kind,
                                                       temp.get(), count));
      break;
    case S32:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<int32_t>(context_, reduction_kind,
                                                      temp.get(), count));
      break;
    case U32:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<uint32_t>(context_, reduction_kind,
                                                       temp.get(), count));
      break;
    case S64:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<int64_t>(context_, reduction_kind,
                                                      temp.get(), count));
      break;
    case U64:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<uint64_t>(context_, reduction_kind,
                                                       temp.get(), count));
      break;
    case S128:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<s128>(context_, reduction_kind,
                                                   temp.get(), count));
      break;
    case S256:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<s256>(context_, reduction_kind,
                                                   temp.get(), count));
      break;
    case U128:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<u128>(context_, reduction_kind,
                                                   temp.get(), count));
      break;
    case U256:
      TF_RETURN_IF_ERROR(ReduceScatterHelper<u256>(context_, reduction_kind,
                                                   temp.get(), count));
      break;
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2)                        \
  case enum:                                                                   \
    TF_RETURN_IF_ERROR(ReduceScatterHelper<cpp_type>(context_, reduction_kind, \
                                                     temp.get(), count));      \
    break;
      ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      return absl::InvalidArgumentError("Unknown datatype in reduce-scatter");
  }
  std::memcpy(recv_buffer.opaque(), temp.get(), chunk_bytes);
  return absl::OkStatus();
}

}  // namespace zkx::cpu
