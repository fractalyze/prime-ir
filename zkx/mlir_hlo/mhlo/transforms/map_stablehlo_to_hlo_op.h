/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_STABLEHLO_TO_HLO_OP_H_
#define ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_STABLEHLO_TO_HLO_OP_H_

#include "stablehlo/dialect/StablehloOps.h"

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::stablehlo {

template <typename HloOpTy>
struct HloToStablehloOpImpl;
template <typename HloOpTy>
using HloToStablehloOp = typename HloToStablehloOpImpl<HloOpTy>::Type;

template <typename StablehloOpTy>
struct StablehloToHloOpImpl;
template <typename StablehloOpTy>
using StablehloToHloOp = typename StablehloToHloOpImpl<StablehloOpTy>::Type;

#define MAP_STABLEHLO_TO_HLO(OpName)                                           \
  template <>                                                                  \
  struct HloToStablehloOpImpl<mhlo::OpName> {                                  \
    using Type = stablehlo::OpName;                                            \
  };                                                                           \
  template <>                                                                  \
  struct StablehloToHloOpImpl<stablehlo::OpName> {                             \
    using Type = mhlo::OpName;                                                 \
  };

MAP_STABLEHLO_TO_HLO(AbsOp)
MAP_STABLEHLO_TO_HLO(AddOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::AfterAllOp
// MAP_STABLEHLO_TO_HLO(AfterAllOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::AllGatherOp
// MAP_STABLEHLO_TO_HLO(AllGatherOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::AllReduceOp
// MAP_STABLEHLO_TO_HLO(AllReduceOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::AllToAllOp
// MAP_STABLEHLO_TO_HLO(AllToAllOp)
MAP_STABLEHLO_TO_HLO(AndOp)
MAP_STABLEHLO_TO_HLO(BitcastConvertOp)
MAP_STABLEHLO_TO_HLO(BroadcastInDimOp)
MAP_STABLEHLO_TO_HLO(BroadcastOp)
MAP_STABLEHLO_TO_HLO(CaseOp)
MAP_STABLEHLO_TO_HLO(ClampOp)
MAP_STABLEHLO_TO_HLO(ClzOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::CollectiveBroadcastOp
// MAP_STABLEHLO_TO_HLO(CollectiveBroadcastOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::CollectivePermuteOp
// MAP_STABLEHLO_TO_HLO(CollectivePermuteOp)
MAP_STABLEHLO_TO_HLO(CompareOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::CompositeOp
// MAP_STABLEHLO_TO_HLO(CompositeOp)
MAP_STABLEHLO_TO_HLO(ConcatenateOp)
MAP_STABLEHLO_TO_HLO(ConstantOp)
MAP_STABLEHLO_TO_HLO(ConvertOp)
MAP_STABLEHLO_TO_HLO(CreateTokenOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::CrossReplicaSumOp
// MAP_STABLEHLO_TO_HLO(CrossReplicaSumOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::CustomCallOp
// MAP_STABLEHLO_TO_HLO(CustomCallOp)
MAP_STABLEHLO_TO_HLO(DivOp)
MAP_STABLEHLO_TO_HLO(DotGeneralOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::DotOp
// MAP_STABLEHLO_TO_HLO(DotOp)
MAP_STABLEHLO_TO_HLO(DynamicBroadcastInDimOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::DynamicGatherOp
// MAP_STABLEHLO_TO_HLO(DynamicGatherOp)
MAP_STABLEHLO_TO_HLO(DynamicIotaOp)
MAP_STABLEHLO_TO_HLO(DynamicPadOp)
MAP_STABLEHLO_TO_HLO(DynamicReshapeOp)
MAP_STABLEHLO_TO_HLO(DynamicSliceOp)
MAP_STABLEHLO_TO_HLO(DynamicUpdateSliceOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::EinsumOp
// MAP_STABLEHLO_TO_HLO(EinsumOp)
MAP_STABLEHLO_TO_HLO(GatherOp)
MAP_STABLEHLO_TO_HLO(GetDimensionSizeOp)
MAP_STABLEHLO_TO_HLO(GetTupleElementOp)
MAP_STABLEHLO_TO_HLO(IfOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::InfeedOp
// MAP_STABLEHLO_TO_HLO(InfeedOp)
MAP_STABLEHLO_TO_HLO(IotaOp)
MAP_STABLEHLO_TO_HLO(MapOp)
MAP_STABLEHLO_TO_HLO(MaxOp)
MAP_STABLEHLO_TO_HLO(MinOp)
MAP_STABLEHLO_TO_HLO(MulOp)
MAP_STABLEHLO_TO_HLO(NegOp)
MAP_STABLEHLO_TO_HLO(NotOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::OptimizationBarrierOp
// MAP_STABLEHLO_TO_HLO(OptimizationBarrierOp)
MAP_STABLEHLO_TO_HLO(OrOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::OutfeedOp
// MAP_STABLEHLO_TO_HLO(OutfeedOp)
MAP_STABLEHLO_TO_HLO(PadOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::PartitionIdOp
// MAP_STABLEHLO_TO_HLO(PartitionIdOp)
MAP_STABLEHLO_TO_HLO(PopulationCountOp)
MAP_STABLEHLO_TO_HLO(PowOp)
MAP_STABLEHLO_TO_HLO(RealDynamicSliceOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::RealOp
// MAP_STABLEHLO_TO_HLO(RecvOp)
MAP_STABLEHLO_TO_HLO(ReduceOp)
MAP_STABLEHLO_TO_HLO(ReduceWindowOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::ReduceScatterOp
// MAP_STABLEHLO_TO_HLO(ReduceScatterOp)
MAP_STABLEHLO_TO_HLO(RemOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::ReplicaIdOp
// MAP_STABLEHLO_TO_HLO(ReplicaIdOp)
MAP_STABLEHLO_TO_HLO(ReshapeOp)
MAP_STABLEHLO_TO_HLO(ReturnOp)
MAP_STABLEHLO_TO_HLO(ReverseOp)
MAP_STABLEHLO_TO_HLO(ScatterOp)
MAP_STABLEHLO_TO_HLO(SelectOp)
// TODO(chokobole): Uncomment this. Dependency: stablehlo::SendOp
// MAP_STABLEHLO_TO_HLO(SendOp)
MAP_STABLEHLO_TO_HLO(SetDimensionSizeOp)
MAP_STABLEHLO_TO_HLO(ShiftLeftOp)
MAP_STABLEHLO_TO_HLO(ShiftRightArithmeticOp)
MAP_STABLEHLO_TO_HLO(ShiftRightLogicalOp)
MAP_STABLEHLO_TO_HLO(SignOp)
MAP_STABLEHLO_TO_HLO(SliceOp)
MAP_STABLEHLO_TO_HLO(SortOp)
MAP_STABLEHLO_TO_HLO(SubtractOp)
MAP_STABLEHLO_TO_HLO(TransposeOp)
MAP_STABLEHLO_TO_HLO(TupleOp)
// (deprecated) MAP_STABLEHLO_TO_HLO(UnaryEinsumOp)
MAP_STABLEHLO_TO_HLO(WhileOp)
MAP_STABLEHLO_TO_HLO(XorOp)

#undef MAP_STABLEHLO_TO_HLO

} // namespace mlir::stablehlo

#endif // ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_STABLEHLO_TO_HLO_OP_H_
