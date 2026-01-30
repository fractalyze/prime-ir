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

#include "zkx/backends/cpu/collectives/in_process_communicator.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/bind_front.h"
#include "absl/log/log.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/comparable_traits.h"
#include "zk_dtypes/include/field/field.h"
#include "zk_dtypes/include/group/group.h"
#include "zkx/backends/cpu/collectives/cpu_collectives.h"
#include "zkx/base/types/always_false.h"
#include "zkx/core/collectives/reduction_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/collective_ops_utils.h"
#include "zkx/service/rendezvous.h"

namespace zkx::cpu {
namespace {

template <typename Participant>
bool ByRank(const Participant* a, const Participant* b) {
  return a->rank < b->rank;
}

template <typename T>
T GetInitialValue(ReductionKind reduction_kind) {
  struct Selector {
    static auto sum() {
      if constexpr (zk_dtypes::IsAdditiveGroup<T>) {
        return T::Zero();
      } else {
        return T{0};
      }
    }

    static auto product() {
      if constexpr (zk_dtypes::IsMultiplicativeGroup<T>) {
        return T::One();
      } else {
        return T{1};
      }
    }

    static auto min() {
      if constexpr (zk_dtypes::IsComparable<T>) {
        return T::Max();
      } else {
        return std::numeric_limits<T>::max();
      }
    }

    static auto max() {
      if constexpr (zk_dtypes::IsComparable<T>) {
        return T::Min();
      } else {
        return std::numeric_limits<T>::lowest();
      }
    }
  };

  absl::StatusOr<T> value_or =
      SelectReduction<T, T>(reduction_kind, Selector{});
  CHECK_OK(value_or.status());
  return value_or.value();
}

template <ReductionKind kReductionKind, typename T>
void ReduceHelper(absl::Span<T> acc, absl::Span<T const* const> inputs) {
  // TODO(penporn): make sure this gets vectorized.
  if constexpr (kReductionKind == ReductionKind::kSum) {
    if constexpr (zk_dtypes::IsAdditiveGroup<T> || std::is_integral_v<T>) {
      if constexpr (zk_dtypes::IsAffinePoint<T>) {
        LOG(FATAL) << "Sum reduction is not supported for AffinePoint types.";
      } else {
        for (size_t j = 0; j < inputs.size(); ++j) {
          for (size_t i = 0; i < acc.size(); ++i) {
            acc[i] += inputs[j][i];
          }
        }
      }
    } else {
      LOG(FATAL) << "Sum reduction is only supported for additive groups or "
                    "integral types.";
    }
  } else if constexpr (kReductionKind == ReductionKind::kProduct) {
    if constexpr (zk_dtypes::IsMultiplicativeGroup<T> ||
                  std::is_integral_v<T>) {
      for (size_t j = 0; j < inputs.size(); ++j) {
        for (size_t i = 0; i < acc.size(); ++i) {
          acc[i] *= inputs[j][i];
        }
      }
    } else {
      LOG(FATAL) << "Product reduction is only supported for "
                    "MultiplicativeGroup types or integral types.";
    }
  } else if constexpr (kReductionKind == ReductionKind::kMin) {
    constexpr PrimitiveType PT = primitive_util::NativeToPrimitiveType<T>();
    if constexpr (!primitive_util::IsComparableType(PT)) {
      LOG(FATAL) << "Min reduction is not supported for non-comparable types.";
    } else {
      for (size_t j = 0; j < inputs.size(); ++j) {
        for (size_t i = 0; i < acc.size(); ++i) {
          acc[i] = std::min(acc[i], inputs[j][i]);
        }
      }
    }
  } else if constexpr (kReductionKind == ReductionKind::kMax) {
    constexpr PrimitiveType PT = primitive_util::NativeToPrimitiveType<T>();
    if constexpr (!primitive_util::IsComparableType(PT)) {
      LOG(FATAL) << "Max reduction is not supported for non-comparable types.";
    } else {
      for (size_t j = 0; j < inputs.size(); ++j) {
        for (size_t i = 0; i < acc.size(); ++i) {
          acc[i] = std::max(acc[i], inputs[j][i]);
        }
      }
    }
  } else {
    static_assert(base::AlwaysFalse<T>, "Unsupported reduction kind");
  }
}

template <PrimitiveType PT>
absl::Status ReduceScatter(ReductionKind reduction_kind,
                           absl::Span<const void* const> inputs, void* output,
                           size_t num_elems) {
  using T = primitive_util::NativeTypeOf<PT>;

  // Types without natural ordering don't support min/max operations.
  if constexpr (!primitive_util::IsComparableType(PT)) {
    if (reduction_kind == ReductionKind::kMin ||
        reduction_kind == ReductionKind::kMax) {
      return absl::InvalidArgumentError(
          "Min/Max reduction is not supported for non-comparable types");
    }
  }

  T initial_value = GetInitialValue<T>(reduction_kind);

  absl::Span<T> out_chunk =
      absl::MakeSpan(reinterpret_cast<T*>(output), num_elems);
  for (size_t i = 0; i < num_elems; ++i) {
    out_chunk[i] = initial_value;
  }

  absl::Span<T const* const> input_chunks(
      reinterpret_cast<T const* const*>(inputs.data()), inputs.size());

  struct Selector {
    Selector(absl::Span<T const* const> input_chunks, absl::Span<T> out_chunk)
        : input_chunks(input_chunks), out_chunk(out_chunk) {}

    auto sum() const {
      ReduceHelper<ReductionKind::kSum, T>(out_chunk, input_chunks);
      return absl::OkStatus();
    }

    auto product() const {
      ReduceHelper<ReductionKind::kProduct, T>(out_chunk, input_chunks);
      return absl::OkStatus();
    }

    auto min() const {
      if constexpr (zk_dtypes::IsComparable<T> || std::is_integral_v<T>) {
        ReduceHelper<ReductionKind::kMin, T>(out_chunk, input_chunks);
        return absl::OkStatus();
      } else {
        return absl::InvalidArgumentError(
            "Min reduction is not supported for non-comparable types");
      }
    }

    auto max() const {
      if constexpr (zk_dtypes::IsComparable<T> || std::is_integral_v<T>) {
        ReduceHelper<ReductionKind::kMax, T>(out_chunk, input_chunks);
        return absl::OkStatus();
      } else {
        return absl::InvalidArgumentError(
            "Max reduction is not supported for non-comparable types");
      }
    }

    absl::Span<T const* const> input_chunks;
    absl::Span<T> out_chunk;
  };

  return SelectReduction<T>(reduction_kind, Selector{input_chunks, out_chunk});
}

//===----------------------------------------------------------------------===//
// AllReduce
//===----------------------------------------------------------------------===//

struct AllReduceParticipant {
  size_t rank;
  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;
};

absl::Status AllReduceOp(PrimitiveType primitive_type, size_t count,
                         ReductionKind reduction_kind,
                         absl::Span<const AllReduceParticipant*> participants) {
  absl::c_sort(participants, ByRank<AllReduceParticipant>);

  if (!primitive_util::IsArrayType(primitive_type)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unexpected datatype: %s",
        primitive_util::LowercasePrimitiveTypeName(primitive_type)));
  }

  // Collect reduction inputs from all participants.
  std::vector<const void*> inputs(participants.size());
  for (auto* participant : participants) {
    inputs[participant->rank] = participant->src.opaque();
  }

  // Reduce all inputs into the destination buffer at rank 0.
  void* output = participants[0]->dest.opaque();

  TF_RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch<absl::Status>(
      [&](const auto type_tag) {
        return ReduceScatter<type_tag>(reduction_kind, inputs, output, count);
      },
      primitive_type));

  // Copy all-reduced output to all other participants.
  for (size_t i = 1; i < participants.size(); ++i) {
    std::memcpy(participants[i]->dest.opaque(), participants[0]->dest.opaque(),
                count * primitive_util::ByteWidth(primitive_type));
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// ReduceScatter
//===----------------------------------------------------------------------===//

struct ReduceScatterParticipant {
  size_t rank;
  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;
};

absl::Status ReduceScatterOp(
    PrimitiveType primitive_type, size_t count, ReductionKind reduction_kind,
    absl::Span<const ReduceScatterParticipant*> participants) {
  absl::c_sort(participants, ByRank<ReduceScatterParticipant>);

  if (!primitive_util::IsArrayType(primitive_type)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unexpected datatype: %s",
        primitive_util::LowercasePrimitiveTypeName(primitive_type)));
  }

  size_t num_participants = participants.size();
  size_t num_bytes = count * primitive_util::ByteWidth(primitive_type);

  for (size_t i = 0; i < num_participants; ++i) {
    size_t offset = i * num_bytes;

    // Collect reduction inputs from all participants.
    std::vector<const void*> inputs(num_participants);
    for (size_t j = 0; j < num_participants; ++j) {
      std::byte* src = static_cast<std::byte*>(participants[j]->src.opaque());
      inputs[j] = src + offset;
    }

    // Reduce all inputs into the destination buffer.
    void* output = participants[i]->dest.opaque();

    TF_RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch<absl::Status>(
        [&](const auto type_tag) {
          return ReduceScatter<type_tag>(reduction_kind, inputs, output, count);
        },
        primitive_type));
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// AllGather
//===----------------------------------------------------------------------===//

struct AllGatherParticipant {
  size_t rank;
  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;
};

absl::Status AllGatherOp(size_t num_bytes,
                         absl::Span<const AllGatherParticipant*> participants) {
  absl::c_sort(participants, ByRank<AllGatherParticipant>);

  size_t num_participants = participants.size();

  for (size_t i = 0; i < num_participants; ++i) {
    for (size_t j = 0; j < num_participants; ++j) {
      std::byte* dest = static_cast<std::byte*>(participants[i]->dest.opaque());
      size_t offset = j * num_bytes;
      std::memcpy(dest + offset, participants[j]->src.opaque(), num_bytes);
    }
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// AllToAll
//===----------------------------------------------------------------------===//

struct AllToAllParticipant {
  size_t rank;

  std::vector<se::DeviceMemoryBase> src;
  std::vector<se::DeviceMemoryBase> dest;
};

absl::Status AllToAllOp(size_t num_bytes,
                        absl::Span<const AllToAllParticipant*> participants) {
  absl::c_sort(participants, ByRank<AllToAllParticipant>);

  size_t num_participants = participants.size();

  for (size_t i = 0; i < num_participants; ++i) {
    for (size_t j = 0; j < num_participants; ++j) {
      std::memcpy(participants[j]->dest[i].opaque(),
                  participants[i]->src[j].opaque(), num_bytes);
    }
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// CollectivePermute
//===----------------------------------------------------------------------===//

struct CollectivePermuteParticipant {
  size_t rank;
  std::optional<RankId> src_rank;

  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;
};

absl::Status CollectivePermuteOp(
    size_t num_bytes,
    absl::Span<const CollectivePermuteParticipant*> participants) {
  absl::c_sort(participants, ByRank<CollectivePermuteParticipant>);

  for (const CollectivePermuteParticipant* participant : participants) {
    void* dest = participant->dest.opaque();

    if (participant->src_rank) {
      size_t src_rank = participant->src_rank->value();
      std::memcpy(dest, participants.at(src_rank)->src.opaque(), num_bytes);
    } else {
      std::memset(dest, 0, num_bytes);
    }
  }
  return absl::OkStatus();
}

}  // namespace

//===----------------------------------------------------------------------===//

absl::Status InProcessCommunicator::AllReduce(se::DeviceMemoryBase send_buffer,
                                              se::DeviceMemoryBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              ReductionKind reduction_kind,
                                              const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("all reduce ", key.ToString());
  AllReduceParticipant participant{rank_, send_buffer, recv_buffer};

  return Rendezvous<absl::Status>(
      name, key, participant, key.num_local_participants,
      absl::bind_front(AllReduceOp, dtype, count, reduction_kind));
}

absl::Status InProcessCommunicator::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("collective permute ", key.ToString());
  CollectivePermuteParticipant participant{rank_, source_rank, send_buffer,
                                           recv_buffer};

  size_t num_bytes = count * primitive_util::ByteWidth(dtype);
  return Rendezvous<absl::Status>(
      name, key, participant, key.num_local_participants,
      absl::bind_front(CollectivePermuteOp, num_bytes));
}

absl::Status InProcessCommunicator::AllToAll(
    absl::Span<const se::DeviceMemoryBase> send_buffers,
    absl::Span<const se::DeviceMemoryBase> recv_buffers, PrimitiveType dtype,
    size_t count, const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("all to all ", key.ToString());
  AllToAllParticipant participant{rank_,
                                  {send_buffers.begin(), send_buffers.end()},
                                  {recv_buffers.begin(), recv_buffers.end()}};

  size_t num_bytes = count * primitive_util::ByteWidth(dtype);
  return Rendezvous<absl::Status>(name, key, participant,
                                  key.num_local_participants,
                                  absl::bind_front(AllToAllOp, num_bytes));
}

absl::Status InProcessCommunicator::AllGather(se::DeviceMemoryBase send_buffer,
                                              se::DeviceMemoryBase recv_buffer,
                                              PrimitiveType dtype, size_t count,
                                              const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("all gather ", key.ToString());
  AllGatherParticipant participant{rank_, send_buffer, recv_buffer};

  size_t num_bytes = count * primitive_util::ByteWidth(dtype);
  return Rendezvous<absl::Status>(name, key, participant,
                                  key.num_local_participants,
                                  absl::bind_front(AllGatherOp, num_bytes));
}

absl::Status InProcessCommunicator::ReduceScatter(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("reduce scatter ", key.ToString());
  ReduceScatterParticipant participant{rank_, send_buffer, recv_buffer};

  return Rendezvous<absl::Status>(
      name, key, participant, key.num_local_participants,
      absl::bind_front(ReduceScatterOp, dtype, count, reduction_kind));
}

}  // namespace zkx::cpu
