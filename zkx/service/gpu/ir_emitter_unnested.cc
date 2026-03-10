/*Copyright 2022 The OpenXLA Authors.

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

#include "zkx/service/gpu/ir_emitter_unnested.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/gpu/codegen/fusion_emitter.h"
#include "zkx/backends/gpu/codegen/fusions.h"
#include "zkx/backends/gpu/runtime/command_buffer_cmd.h"
#include "zkx/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "zkx/backends/gpu/runtime/command_buffer_thunk.h"
#include "zkx/backends/gpu/runtime/msm/msm_thunk.h"
#include "zkx/backends/gpu/runtime/wait_for_streams_thunk.h"
#include "zkx/backends/gpu/runtime/while_thunk.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/ir_emission_utils.h"
#include "zkx/service/gpu/stream_executor_util.h"
#include "zkx/service/llvm_ir/buffer_assignment_util.h"
#include "zkx/status_macros.h"

namespace zkx::gpu {

IrEmitterUnnested::IrEmitterUnnested(IrEmitterContext* ir_emitter_context)
    : IrEmitter(ir_emitter_context, /*is_nested=*/false),
      // TODO(chokobole): Uncomment this. Dependency: SendRecvAsyncEvents
      // send_recv_events_(std::make_shared<SendRecvAsyncEvents>()),
      copy_events_(std::make_shared<CopyThunk::AsyncEvents>()) {}

std::unique_ptr<IrEmitterUnnested> IrEmitterUnnested::Create(
    IrEmitterContext* ir_emitter_context) {
  return std::unique_ptr<IrEmitterUnnested>(
      new IrEmitterUnnested(ir_emitter_context));
}

absl::Status IrEmitterUnnested::EmitConstant(
    const HloConstantInstruction* instr) {
  TF_ASSIGN_OR_RETURN(DenseDataIntermediate content,
                      LiteralToZkxFormat(instr->literal()));

  int element_bytes =
      primitive_util::ByteWidth(instr->literal().shape().element_type());
  TF_RET_CHECK(content.span().size() % element_bytes == 0);
  // Treat packed constants as a byte constant.
  int num_elements = content.span().size() / element_bytes;

  std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      GetAllocationSliceForHlo(instr, {}));

  ir_emitter_context_->emit_constant(num_elements, element_bytes, global_name,
                                     slice.index(), std::move(content), &b_);
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitConditional(const HloInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitConditional");
}

// TODO(chokobole): Uncomment this. Dependency: llvm builder
// llvm::Value* IrEmitterUnnested::CreateLoad(llvm::Value* address,
//                                            llvm::Type* data_type,
//                                            int alignment_bytes) {
//   int data_bytes = data_type->getPrimitiveSizeInBits() /
//                    primitive_util::BitWidth(PrimitiveType::U8);
//   if (alignment_bytes == 0) {
//     return b_.CreateLoad(data_type, address);
//   }

//   int alignment_bitwidth =
//       alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

//   llvm::Value* output = llvm::ConstantInt::get(data_type, 0);
//   for (int offset_bytes = 0; offset_bytes < data_bytes;
//        offset_bytes += alignment_bytes) {
//     llvm::Value* offset_address = b_.CreateConstInBoundsGEP1_32(
//         b_.getInt8Ty(), address, offset_bytes, "offset_address");
//     llvm::Value* partial_value =
//     b_.CreateLoad(b_.getIntNTy(alignment_bitwidth),
//                                                offset_address,
//                                                "partial_value");
//     llvm::Value* zextd =
//         b_.CreateZExt(partial_value, output->getType(),
//         "partial_value_zextd");
//     llvm::Value* shifted = b_.CreateShl(
//         zextd, llvm::ConstantInt::get(b_.getInt32Ty(), offset_bytes),
//         "partial_input_shifted");
//     output = b_.CreateAdd(output, shifted, "output_updated");
//   }
//   return output;
// }

// TODO(chokobole): Uncomment this. Dependency: llvm builder
// void IrEmitterUnnested::CreateStore(llvm::Value* data, llvm::Value* address,
//                                     int alignment_bytes) {
//   int data_bytes = data->getType()->getPrimitiveSizeInBits() /
//                    primitive_util::BitWidth(PrimitiveType::U8);
//   CHECK_GE(data_bytes, alignment_bytes);
//   if (alignment_bytes == 0) {
//     b_.CreateStore(data, address);
//     return;
//   }

//   int alignment_bitwidth =
//       alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

//   for (int offset_bytes = 0; offset_bytes < data_bytes;
//        offset_bytes += alignment_bytes) {
//     llvm::Value* offset_address = b_.CreateConstInBoundsGEP1_32(
//         b_.getInt8Ty(), address, offset_bytes, "offset_address");
//     llvm::Value* shifted_partial = b_.CreateTrunc(
//         b_.CreateLShr(data,
//                       llvm::ConstantInt::get(b_.getInt32Ty(), offset_bytes)),
//         b_.getIntNTy(alignment_bitwidth), "truncated_value");
//     b_.CreateStore(shifted_partial, offset_address);
//   }
// }

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
// TODO(chokobole): Implement this. Dependency: HloCustomCallInstruction.
// absl::Status IrEmitterUnnested::EmitPadToStatic(

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
// TODO(chokobole): Implement this. Dependency: HloCustomCallInstruction.
// absl::Status IrEmitterUnnested::EmitSliceToDynamic(

absl::Status IrEmitterUnnested::EmitCommandBufferThunk(
    const HloInstruction* instr) {
  // Spawn a new IrEmitterUnnested to emit thunks for the command buffer
  // computation. Then convert emitted thunks to a sequence of CommandBufferCmd.
  // The resulting thunk added to the thunk sequence is a CommandBufferThunk.
  // Thunks emitted from the command buffer computation are discarded.
  DCHECK_EQ(instr->called_computations().size(), 1);
  const HloComputation* command_buffer = instr->called_computations().front();
  auto ir_emitter = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(command_buffer));
  std::unique_ptr<SequentialThunk> thunk_sequence =
      ir_emitter->ConsumeThunkSequence();

  // Maybe serialize all commands in a sequence by forcing barriers between all
  // recorded commands. This guarantees that we execute all device operations
  // in the exact same order as a thunk sequence.
  CommandBufferCmdSequence::SynchronizationMode synchronization_mode =
      ir_emitter_context_->debug_options()
              .zkx_gpu_graph_enable_concurrent_region()
          ? CommandBufferCmdSequence::SynchronizationMode::kAutomatic
          : CommandBufferCmdSequence::SynchronizationMode::kSerialize;

  TF_ASSIGN_OR_RETURN(
      CommandBufferCmdSequence cmd_sequence,
      ConvertToCommands(thunk_sequence->thunks(), synchronization_mode));

  AddThunkToThunkSequence(std::make_unique<CommandBufferThunk>(
      std::move(cmd_sequence), Thunk::ThunkInfo::WithProfileAnnotation(instr),
      std::move(thunk_sequence),
      ir_emitter_context_->debug_options()
          .zkx_enable_command_buffers_during_profiling()));

  return absl::OkStatus();
}

// TODO(chokobole): Implement this. Dependency: HloCustomCallInstruction.
// absl::Status IrEmitterUnnested::EmitGemmThunk(
//     const HloCustomCallInstruction* instr) {

// TODO(chokobole): Implement this. Dependency: HloCustomCallInstruction.
// absl::Status IrEmitterUnnested::EmitCublasLtMatmulThunk(
//     const HloCustomCallInstruction* instr);

absl::StatusOr<BufferAllocation::Slice>
IrEmitterUnnested::GetAllocationSliceForHlo(const HloInstruction* instr,
                                            const ShapeIndex& index) const {
  return GetAllocationSlice(ir_emitter_context_->buffer_assignment(), instr,
                            index);
}

// TODO(chokobole): Implement this. Dependency: HloCustomCallInstruction.
// absl::Status IrEmitterUnnested::EmitCubDeviceRadixSort(
//     const HloCustomCallInstruction* instr)

// TODO(chokobole): Implement this. Dependency: HloCustomCallInstruction.
// absl::Status IrEmitterUnnested::EmitCustomCallThunk(
//     const HloCustomCallInstruction* instr) {

// TODO(chokobole): Implement this. Dependency: HloCustomCallInstruction.
// absl::Status IrEmitterUnnested::EmitTritonCustomCall(
//     const HloCustomCallInstruction* instr) {

absl::Status IrEmitterUnnested::EmitAsyncComputation(
    const HloInstruction* instr) {
  const HloInstruction* wrapped = instr->async_wrapped_instruction();
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(auto stream,
                      stream_assignment.GetSyncExecutionStreamId(wrapped));
  TF_RET_CHECK(wrapped->called_computations().size() == 1);
  auto computation = wrapped->called_computations().front();
  auto ir_emitter = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter->EmitHloComputation(computation));
  std::unique_ptr<SequentialThunk> thunk_sequence =
      ir_emitter->ConsumeThunkSequence();
  for (auto& thunk : thunk_sequence->thunks()) {
    thunk->set_execution_stream_id(stream);
  }
  auto* async_start = Cast<HloAsyncInstruction>(instr);
  TF_ASSIGN_OR_RETURN(
      ExecutionStreamAssignment::AsyncExecutionStreamIds async_streams,
      stream_assignment.GetAsyncExecutionStreamIds(async_start));
  // We launch the thunk sequence computation on a concurrent stream.
  // The concurrent stream needs to first wait until the main stream has
  // finished calculating any values that may be used as input.
  // We enforce this by inlining a `WaitForStreams` thunk on the main
  // stream.
  AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      async_streams.destination_stream_id, async_streams.source_stream_id));
  AddThunkToThunkSequence(std::move(thunk_sequence));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitFusion(const HloFusionInstruction* instr) {
  const se::DeviceDescription& device_info =
      ir_emitter_context_->gpu_device_info();
  const HloFusionAnalysis fusion_analysis =
      HloFusionAnalysis::Create(*instr, device_info);
  VLOG(3) << "IrEmitterUnnested::EmitFusion:start";
  std::unique_ptr<FusionInterface> emitter = GetFusionEmitter(HloFusionInfo(
      fusion_analysis, instr, &ir_emitter_context_->buffer_assignment()));
  TF_ASSIGN_OR_RETURN(auto result, emitter->Emit(*ir_emitter_context_, *instr));

  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  for (std::unique_ptr<Thunk>& thunk : result.thunks) {
    TF_ASSIGN_OR_RETURN(ExecutionStreamId execution_stream_id,
                        stream_assignment.GetSyncExecutionStreamId(instr));
    thunk->set_execution_stream_id(execution_stream_id);
    AddThunkToThunkSequence(std::move(thunk));
  }
  VLOG(3) << "IrEmitterUnnested::EmitFusion:complete";
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCopy(const HloInstruction* instr) {
  TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
      instr->operand(0)->shape(), instr->shape(),
      Layout::Equal().MinorToMajorOnly()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(instr));
  AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      /*source_buffer=*/src_buffer,
      /*destination_buffer=*/dst_buffer,
      /*mem_size=*/src_buffer.size()));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitMsm(const HloMsmInstruction* instr) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice scalars_slice,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bases_slice,
                      GetAllocationSliceForHlo(instr->operand(1)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSliceForHlo(instr, {}));

  // MSM size = number of scalar elements.
  const Shape& scalar_shape = instr->operand(0)->shape();
  int32_t msm_size = scalar_shape.dimensions(0);

  PrimitiveType scalars_et = instr->operand(0)->shape().element_type();
  PrimitiveType bases_et = instr->operand(1)->shape().element_type();
  PrimitiveType result_et = instr->shape().element_type();
  AddThunkToThunkSequence(std::make_unique<MsmThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), scalars_slice,
      bases_slice, result_slice, msm_size, instr->window_bits(), scalars_et,
      bases_et, result_et));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitSlice(const HloInstruction* instr) {
  const Shape& src_shape = instr->operand(0)->shape();

  // Only support unit strides.
  for (int64_t i = 0; i < src_shape.rank(); ++i) {
    if (instr->slice_strides(i) != 1) {
      return absl::UnimplementedError(
          "Non-unit stride slice not supported as standalone instruction");
    }
  }

  // Check contiguity: for all but the most-major dimension, the slice must
  // cover the full extent. Otherwise, the slice is non-contiguous in memory
  // and cannot be emitted as a single memcpy.
  const auto& minor_to_major = src_shape.layout().minor_to_major();
  for (int64_t i = 0; i < src_shape.rank() - 1; ++i) {
    int64_t dim = minor_to_major[i];
    if (instr->slice_starts(dim) != 0 ||
        instr->slice_limits(dim) != src_shape.dimensions(dim)) {
      return absl::UnimplementedError(
          "Non-contiguous slice not supported as standalone instruction");
    }
  }

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(instr));

  // Compute byte offset. Only the most-major dimension can have non-zero start
  // (verified by the contiguity check above).
  int64_t byte_offset = 0;
  if (src_shape.rank() > 0) {
    int64_t most_major_dim = minor_to_major.back();
    int64_t element_size =
        ShapeUtil::ByteSizeOfPrimitiveType(src_shape.element_type());
    int64_t stride = 1;
    for (int64_t i = 0; i < src_shape.rank() - 1; ++i) {
      stride *= src_shape.dimensions(minor_to_major[i]);
    }
    byte_offset = instr->slice_starts(most_major_dim) * stride * element_size;
  }

  BufferAllocation::Slice adjusted_src(src_buffer.allocation(),
                                       src_buffer.offset() + byte_offset,
                                       dst_buffer.size());

  AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr),
      /*source_buffer=*/adjusted_src,
      /*destination_buffer=*/dst_buffer,
      /*mem_size=*/dst_buffer.size()));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitAsyncCustomCallStart(
    const HloInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError(
      "Not implemented for EmitAsyncCustomCallStart");
}

absl::Status IrEmitterUnnested::AssertNonDeterminismIsOkay(
    const std::string& op_name) {
  if (RequireDeterminism(ir_emitter_context_->hlo_module().config())) {
    return absl::UnimplementedError(absl::StrFormat(
        "HLO instruction %s does not have a deterministic implementation, "
        "but run-to-run determinism is required.",
        op_name));
  }
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitWhile(const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto config,
                      instr->backend_config<WhileLoopBackendConfig>());

  std::optional<int64_t> trip_count = std::nullopt;
  if (config.has_known_trip_count()) trip_count = config.known_trip_count().n();

  TF_ASSIGN_OR_RETURN(
      auto thunk,
      BuildWhileThunk(instr, Thunk::ThunkInfo::WithProfileAnnotation(instr),
                      trip_count));

  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

// TODO(chokobole): Implement this. Dependency: HloSortInstruction.
// absl::Status IrEmitterUnnested::EmitSort(const HloSortInstruction* sort) {

template <typename ThunkType>
absl::Status IrEmitterUnnested::EmitReplicaOrPartitionId(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSliceForHlo(instr, {}));
  auto thunk = std::make_unique<ThunkType>(
      Thunk::ThunkInfo::WithProfileAnnotation(instr), result_slice);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCollectivePermute(
    const HloCollectivePermuteInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError(
      "Not implemented for EmitReplicaOrPartitionId");
}

template <typename NcclThunkType, typename HloInstType>
absl::Status IrEmitterUnnested::EmitNcclThunk(
    Thunk::Kind kind, const HloInstruction* async_start,
    const HloInstType* inst, std::optional<bool> use_global_device_ids) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitNcclThunk");
}

absl::Status IrEmitterUnnested::EmitNcclGroupStartThunk(
    const HloInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitNcclThunk");
}

absl::Status IrEmitterUnnested::EmitNcclAsyncDone(Thunk::Kind kind,
                                                  const HloInstruction* inst) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitNcclAsyncDone");
}

absl::Status IrEmitterUnnested::EmitInfeed(const HloInfeedInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitInfeed");
}

absl::Status IrEmitterUnnested::EmitOutfeed(
    const HloOutfeedInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitOutfeed");
}

// TODO(chokobole): Uncomment this. Dependency: IrArray.
// absl::StatusOr<std::pair<std::vector<llvm_ir::IrArray> /*inputs*/,
//                          std::vector<llvm_ir::IrArray> /*outputs*/>>
// IrEmitterUnnested::BuildKernelThunkForNonFusionOp(

absl::StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildWhileThunk(
    const HloInstruction* instr, const Thunk::ThunkInfo& thunk_info,
    std::optional<int64_t> trip_count) {
  HloComputation* condition = instr->while_condition();
  HloComputation* body = instr->while_body();

  // Generate thunk sequence for while 'condition'.
  auto ir_emitter_condition = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter_condition->EmitHloComputation(condition));

  // Generate thunk sequence for while 'body'.
  auto ir_emitter_body = IrEmitterUnnested::Create(ir_emitter_context_);
  TF_RETURN_IF_ERROR(ir_emitter_body->EmitHloComputation(body));

  // Buffer slice holding while loop predicate.
  TF_ASSIGN_OR_RETURN(
      auto pred, GetAllocationSliceForHlo(condition->root_instruction(), {}));

  Thunk::ThunkInfo cond_thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr);
  cond_thunk_info.profile_annotation += "_condition";
  Thunk::ThunkInfo body_thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr);
  body_thunk_info.profile_annotation += "_body";

  return std::unique_ptr<Thunk>(new WhileThunk(
      thunk_info, pred,
      ir_emitter_condition->ConsumeThunkSequence(cond_thunk_info),
      ir_emitter_body->ConsumeThunkSequence(body_thunk_info), trip_count));
}

// TODO(chokobole): Uncomment this. Dependency: ElementGenerator.
// absl::Status IrEmitterUnnested::EmitTargetElementLoop(
//     const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter)
//     {
//   return absl::InternalError("This should be unreachable");
// }
namespace {

absl::StatusOr<bool> ShapeHasHostMemorySpace(Shape shape, int index,
                                             int host_memory_space) {
  return shape.tuple_shapes(index).has_layout() &&
         shape.tuple_shapes(index).layout().memory_space() == host_memory_space;
}

}  // namespace

absl::Status IrEmitterUnnested::EmitCopyStartThunk(
    const HloCopyStartInstruction* copy_start_instr) {
  // copy-start has a tuple shape: {host, device, context},
  // or {device, host, context}.
  // Only the destination shape is needed to get the output buffer.
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      GetAllocationSliceForHlo(copy_start_instr,
                                               /*index=*/{0}));

  const HloInstruction* src = copy_start_instr->operand(0);
  const Shape& input_shape = src->shape();
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      GetAllocationSliceForHlo(src, {}));
  const Shape& shape = copy_start_instr->shape();
  CHECK(shape.IsTuple());
  int host_memory_space = static_cast<int>(se::MemoryType::kHost);
  TF_ASSIGN_OR_RETURN(bool is_dst_host_memory,
                      ShapeHasHostMemorySpace(shape, 0, host_memory_space));
  TF_ASSIGN_OR_RETURN(bool is_src_host_memory,
                      ShapeHasHostMemorySpace(shape, 1, host_memory_space));
  if (is_dst_host_memory == is_src_host_memory) {
    return absl::InternalError(absl::StrFormat(
        "Copy-start %s doesn't have correct host memory space color S(%d)",
        copy_start_instr->ToString(), static_cast<int>(se::MemoryType::kHost)));
  }
  const ExecutionStreamAssignment& stream_assignment =
      ir_emitter_context_->execution_stream_assignment();
  TF_ASSIGN_OR_RETURN(
      ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
      stream_assignment.GetAsyncExecutionStreamIds(copy_start_instr));
  // Insert a waitFor() thunk for asynchronous memcpy only when the source
  // and destination stream IDs differ. If the IDs are the same, the memcpy
  // operation is synchronous within that stream.
  if (streams.destination_stream_id != streams.source_stream_id) {
    AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
        streams.destination_stream_id, streams.source_stream_id));
  }
  if (is_dst_host_memory) {
    auto thunk = std::make_unique<DeviceToHostCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
        /*source_buffer=*/src_buffer,
        /*destination_buffer=*/dst_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape),
        /*copy_events=*/copy_events_,
        /*copy_start_instr=*/copy_start_instr);
    thunk->set_execution_stream_id(streams.destination_stream_id);
    AddThunkToThunkSequence(std::move(thunk));
  } else {
    auto thunk = std::make_unique<HostToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
        /*source_buffer=*/src_buffer,
        /*destination_buffer=*/dst_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(input_shape),
        /*copy_events=*/copy_events_,
        /*copy_start_instr=*/copy_start_instr);
    thunk->set_execution_stream_id(streams.destination_stream_id);
    AddThunkToThunkSequence(std::move(thunk));
  }

  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitCopyDoneThunk(const HloInstruction* instr) {
  const HloInstruction* copy_start_instr = instr->operand(0);
  CHECK(copy_start_instr->opcode() == HloOpcode::kCopyStart);

  auto thunk = std::make_unique<CopyDoneThunk>(
      Thunk::kCopyDone,
      Thunk::ThunkInfo::WithProfileAnnotation(copy_start_instr),
      /*copy_events=*/copy_events_,
      /*copy_start_instr=*/copy_start_instr);
  AddThunkToThunkSequence(std::move(thunk));
  return absl::OkStatus();
}

absl::Status IrEmitterUnnested::EmitSendThunk(const HloSendInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitSendThunk");
}

absl::Status IrEmitterUnnested::EmitSendDoneThunk(
    const HloSendDoneInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitSendDoneThunk");
}

absl::Status IrEmitterUnnested::EmitRecvThunk(const HloRecvInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitRecvThunk");
}

absl::Status IrEmitterUnnested::EmitRecvDoneThunk(
    const HloRecvDoneInstruction* instr) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented for EmitRecvDoneThunk");
}

absl::Status IrEmitterUnnested::EmitHloInstruction(
    const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kAllGatherDone:
      return EmitNcclAsyncDone(Thunk::kNcclAllGatherDone, instr);
    case HloOpcode::kAllGatherStart: {
      // TODO(chokobole): Implement this.
      return absl::UnimplementedError(
          "Not implemented for HloOpcode::kAllGatherStart");
    }

    case HloOpcode::kAllReduceDone:
      return EmitNcclAsyncDone(Thunk::kNcclAllReduceDone, instr);
    case HloOpcode::kAllReduceStart: {
      // TODO(chokobole): Implement this.
      return absl::UnimplementedError(
          "Not implemented for HloOpcode::kAllReduceStart");
    }
    case HloOpcode::kAsyncDone: {
      if (!instr->async_wrapped_computation()
               ->CanExpandIntoSingleInstruction()) {
        return EmitNcclAsyncDone(Thunk::kNcclGroupDone, instr);
      }
      const HloInstruction* wrapped = instr->async_wrapped_instruction();
      switch (wrapped->opcode()) {
        case HloOpcode::kReduceScatter:
          return EmitNcclAsyncDone(Thunk::kNcclReduceScatterDone, instr);
        case HloOpcode::kAllToAll:
          return EmitNcclAsyncDone(Thunk::kNcclAllToAllDone, instr);
        case HloOpcode::kRaggedAllToAll:
          return EmitNcclAsyncDone(Thunk::kNcclRaggedAllToAllDone, instr);
        case HloOpcode::kCollectiveBroadcast:
          return EmitNcclAsyncDone(Thunk::kNcclCollectiveBroadcastDone, instr);
        case HloOpcode::kFusion:
        case HloOpcode::kCall:
        case HloOpcode::kCustomCall: {
          // Wait until the concurrent stream has finished.
          auto* async_done = Cast<HloAsyncInstruction>(instr);
          const ExecutionStreamAssignment& stream_assignment =
              ir_emitter_context_->execution_stream_assignment();
          TF_ASSIGN_OR_RETURN(
              ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
              stream_assignment.GetAsyncExecutionStreamIds(async_done));
          AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
              Thunk::ThunkInfo::WithProfileAnnotation(instr),
              streams.source_stream_id, streams.destination_stream_id));
          return absl::OkStatus();
        }
        default:
          return absl::InternalError(
              absl::StrFormat("Unsupported async done wrapped instruction: %s",
                              HloOpcodeString(wrapped->opcode())));
      }
    }
    case HloOpcode::kAsyncStart: {
      // Multi-op async start will emit a NCCL group thunk.
      if (!instr->async_wrapped_computation()
               ->CanExpandIntoSingleInstruction()) {
        return EmitNcclGroupStartThunk(instr);
      }
      const HloInstruction* wrapped = instr->async_wrapped_instruction();
      switch (wrapped->opcode()) {
        case HloOpcode::kReduceScatter: {
          // clang-format off
          // TODO(chokobole): Uncomment this. Dependency: NcclReduceScatterStartThunk.
          // clang-format on
          // auto* reduce_scatter = Cast<HloReduceScatterInstruction>(wrapped);
          // return EmitNcclThunk<NcclReduceScatterStartThunk,
          //                      HloReduceScatterInstruction>(
          //     Thunk::kNcclReduceScatter, instr, reduce_scatter,
          //     reduce_scatter->use_global_device_ids());
          return absl::UnimplementedError(
              "Not implemented for HloOpcode::kReduceScatter");
        }
        case HloOpcode::kAllToAll: {
          // clang-format off
          // TODO(chokobole): Uncomment this. Dependency: NcclAllToAllStartThunk.
          // clang-format on
          //   auto* all_to_all = Cast<HloAllToAllInstruction>(wrapped);
          //   return EmitNcclThunk<NcclAllToAllStartThunk,
          //   HloAllToAllInstruction>(
          //       Thunk::kNcclAllToAll, instr, all_to_all, std::nullopt);
          // }
          return absl::UnimplementedError(
              "Not implemented for HloOpcode::kAllToAll");
        }
        case HloOpcode::kRaggedAllToAll: {
          // clang-format off
          // TODO(chokobole): Uncomment this. Dependency: NcclRaggedAllToAllStartThunk.
          // clang-format on
          // auto* ragged_all_to_all =
          // Cast<HloRaggedAllToAllInstruction>(wrapped); return
          // EmitNcclThunk<NcclRaggedAllToAllStartThunk,
          //                      HloRaggedAllToAllInstruction>(
          //     Thunk::kNcclRaggedAllToAll, instr, ragged_all_to_all,
          //     std::nullopt);
          return absl::UnimplementedError(
              "Not implemented for HloOpcode::kRaggedAllToAll");
        }
        case HloOpcode::kCollectiveBroadcast: {
          // clang-format off
          // TODO(chokobole): Uncomment this. Dependency: NcclCollectiveBroadcastStartThunk.
          // clang-format on
          // auto* collective_broadcast =
          //     Cast<HloCollectiveBroadcastInstruction>(wrapped);
          // return EmitNcclThunk<NcclCollectiveBroadcastStartThunk,
          //                      HloCollectiveBroadcastInstruction>(
          //     Thunk::kNcclCollectiveBroadcast, instr, collective_broadcast,
          //     std::nullopt);
          return absl::UnimplementedError(
              "Not implemented for HloOpcode::kCollectiveBroadcast");
        }
        case HloOpcode::kFusion: {
          // We'll launch the fusion computation on a concurrent stream. The
          // concurrent stream needs to first wait until the main stream has
          // finished calculating any values that may be used as inputs to the
          // fusion computation. We enforce this by inlining a
          // `WaitForStreams` thunk.
          auto* async_start = Cast<HloAsyncInstruction>(instr);
          const ExecutionStreamAssignment& stream_assignment =
              ir_emitter_context_->execution_stream_assignment();
          TF_ASSIGN_OR_RETURN(
              ExecutionStreamAssignment::AsyncExecutionStreamIds streams,
              stream_assignment.GetAsyncExecutionStreamIds(async_start));
          AddThunkToThunkSequence(std::make_unique<WaitForStreamsThunk>(
              Thunk::ThunkInfo::WithProfileAnnotation(instr),
              streams.destination_stream_id, streams.source_stream_id));
          return EmitFusion(Cast<HloFusionInstruction>(wrapped));
        }
        case HloOpcode::kCall: {
          return EmitAsyncComputation(instr);
        }
        case HloOpcode::kCustomCall: {
          return EmitAsyncCustomCallStart(instr);
        }
        default:
          return absl::InternalError(
              absl::StrFormat("Unsupported async start wrapped instruction: %s",
                              HloOpcodeString(wrapped->opcode())));
      }
    }

    case HloOpcode::kCall:
      return EmitCommandBufferThunk(instr);
    case HloOpcode::kCollectivePermuteDone:
      return EmitNcclAsyncDone(Thunk::kNcclCollectivePermuteDone, instr);
    case HloOpcode::kCollectivePermuteStart:
      return EmitCollectivePermute(
          Cast<HloCollectivePermuteInstruction>(instr));
    case HloOpcode::kConditional:
      return EmitConditional(instr);
    case HloOpcode::kConstant:
      return EmitConstant(Cast<HloConstantInstruction>(instr));
    case HloOpcode::kCustomCall: {
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: HloCustomCallInstruction.
      // clang-format on
      // auto* custom_call = Cast<HloCustomCallInstruction>(instr);
      // TODO(chokobole): Uncomment this. Dependency: IsLegacyCublasMatmul.
      // if (IsLegacyCublasMatmul(*instr)) {
      //   return EmitGemmThunk(custom_call);
      // }
      // TODO(chokobole): Uncomment this. Dependency: IsCublasLtMatmul.
      // if (IsCublasLtMatmul(*instr)) {
      //   return EmitCublasLtMatmulThunk(custom_call);
      // }
      // TODO(chokobole): Uncomment this. Dependency: IsCubDeviceRadixSort.
      // if (IsCubDeviceRadixSort(*instr)) {
      //   return EmitCubDeviceRadixSort(custom_call);
      // }
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: HloInstruction::custom_call_target
      // clang-format on
      // if (custom_call->custom_call_target() == "PadToStatic") {
      //   return EmitPadToStatic(custom_call);
      // }
      // if (instr->custom_call_target() == "SliceToDynamic") {
      //   return EmitSliceToDynamic(custom_call);
      // }
      // if (instr->custom_call_target() == "__gpu$xla.gpu.triton") {
      //   // TODO(slebedev): Remove this after June 15th 2025.
      //   return EmitTritonCustomCall(custom_call);
      // }
      // if (instr->custom_call_target() == kNopCustomCallTarget) {
      //   return absl::OkStatus();
      // }
      // return EmitCustomCallThunk(custom_call);
      return absl::UnimplementedError(
          "Not implemented for HloOpcode::kCustomCall");
    }
    case HloOpcode::kFusion:
      return EmitFusion(Cast<HloFusionInstruction>(instr));
    case HloOpcode::kMsm:
      return EmitMsm(Cast<HloMsmInstruction>(instr));
    case HloOpcode::kCopy:
      return EmitCopy(instr);
    case HloOpcode::kInfeed:
      return EmitInfeed(Cast<HloInfeedInstruction>(instr));
    case HloOpcode::kOutfeed:
      return EmitOutfeed(Cast<HloOutfeedInstruction>(instr));
    case HloOpcode::kPartitionId:
      // TODO(chokobole): Uncomment this. Dependency: PartitionIdThunk.
      // return EmitReplicaOrPartitionId<PartitionIdThunk>(instr);
      return absl::UnimplementedError(
          "Not implemented for HloOpcode::kPartitionId");

    case HloOpcode::kRecv:
      return EmitRecvThunk(Cast<HloRecvInstruction>(instr));
    case HloOpcode::kRecvDone:
      return EmitRecvDoneThunk(Cast<HloRecvDoneInstruction>(instr));

    case HloOpcode::kReplicaId:
      // TODO(chokobole): Uncomment this. Dependency: ReplicaIdThunk.
      // return EmitReplicaOrPartitionId<ReplicaIdThunk>(instr);
      return absl::UnimplementedError(
          "Not implemented for HloOpcode::kReplicaId");

    case HloOpcode::kSend:
      return EmitSendThunk(Cast<HloSendInstruction>(instr));
    case HloOpcode::kSendDone:
      return EmitSendDoneThunk(Cast<HloSendDoneInstruction>(instr));

    case HloOpcode::kSlice:
      return EmitSlice(instr);
    case HloOpcode::kSort:
      // TODO(chokobole): Uncomment this. Dependency: HloSortInstruction.
      // return EmitSort(Cast<HloSortInstruction>(instr));
      return absl::UnimplementedError("Not implemented for HloOpcode::kSort");
    case HloOpcode::kWhile:
      return EmitWhile(instr);
    case HloOpcode::kCopyStart:
      return EmitCopyStartThunk(Cast<HloCopyStartInstruction>(instr));
    case HloOpcode::kCopyDone:
      return EmitCopyDoneThunk(instr);

    // HLO module is already scheduled, so instructions for ordering are
    // noops.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    // We don't need to emit thunks for these operations because their
    // semantics are encoded by buffers.
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return absl::OkStatus();
    default:
      LOG(ERROR) << "Unsupported instruction opcode: "
                 << HloOpcodeString(instr->opcode()) << "\nHLO module:\n"
                 << instr->parent()->parent()->ToString();
      return absl::InternalError(
          absl::StrFormat("Unsupported instruction opcode: %s",
                          HloOpcodeString(instr->opcode())));
  }

  return absl::InternalError("Unhandled HLO instruction");
}

absl::Status IrEmitterUnnested::EmitHloComputation(
    const HloComputation* computation) {
  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation))
    return absl::InternalError(absl::StrFormat(
        "Sequence not found for computation: %s", computation->name()));

  const HloInstructionSequence& sequence = schedule.sequence(computation);
  for (HloInstruction* instr : sequence.instructions()) {
    TF_RETURN_IF_ERROR(EmitHloInstruction(instr));
  }
  return absl::OkStatus();
}

}  // namespace zkx::gpu
