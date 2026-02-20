/* Copyright 2023 The OpenXLA Authors.
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

#include "zkx/service/gpu/prepare_hlo_for_ir_emitting_pipeline.h"

#include "zkx/hlo/transforms/simplifiers/hlo_dce.h"
#include "zkx/service/copy_insertion.h"
#include "zkx/service/gpu/transforms/copy_fusion.h"
#include "zkx/service/gpu/transforms/horizontal_loop_fusion.h"

namespace zkx::gpu {
namespace {

// TODO(chokobole): Uncomment this. Dependency: CpuGpuVerifierMetadata
// std::unique_ptr<TargetVerifierMetadata> CreateVerifierMetadata(
//     const HloModuleConfig& config) {
//   HloVerifierOpts opts =
//       HloVerifierOpts{}.MakeLayoutSensitive().WithInstructionCanChangeLayout(
//           LayoutAssignment::InstructionCanChangeLayout);
//   opts.verify_unique_channel_ids =
//       !config.debug_options().zkx_ignore_channel_id();
//   return std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
// }

}  // namespace

HloPassPipeline PrepareHloModuleForIrEmittingPipeline(
    HloModule& hlo_module, HloDataflowAnalysis::CanShareBuffer can_share_buffer,
    const se::DeviceDescription& device_description) {
  const DebugOptions& debug_options = hlo_module.config().debug_options();

  // TODO(chokobole): Use |module| once CpuGpuVerifierMetadata and
  // AliasPassthroughParams are ported.
  (void)hlo_module;

  // In some cases, we have to place the result of an instruction in a temporary
  // buffer. For instance, the buffer that holds an external parameter is
  // assumed immutable at this point, and should not be reused for output
  // (b/27180329). Therefore, in that case, we set the output to be a copy of
  // the parameter.
  HloPassPipeline pipeline("GPU-ir-emit-prepare");

  // TODO(chokobole): Uncomment this. Dependency: CpuGpuVerifierMetadata
  // pipeline.AddInvariantCheckerDebug<HloVerifier>(
  //     CreateVerifierMetadata(config), "hlo verifier (debug)");

  // Copy insertion should be performed immediately before IR emission to avoid
  // inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes an
  // instruction which materializes a value). DCE must be run immediately before
  // (and sometime after) copy insertion, to avoid dead code from interfering
  // with the rewrites.
  pipeline.AddPass<HloDCE>();

  // TODO(chokobole): Uncomment this. Dependency: AliasPassthroughParams
  // if (hlo_module.config().alias_passthrough_params()) {
  //   pipeline.AddPass<AliasPassthroughParams>();
  // }

  // TODO(chokobole): Uncomment this. Dependency: LoopScheduleLinearizer
  // pipeline.AddPass<LoopScheduleLinearizer>(can_share_buffer);

  if (debug_options.zkx_gpu_copy_insertion_use_region_analysis()) {
    constexpr int64_t kNoRegionBasedLiveRangeAnalysisLimit = -1;
    pipeline.AddPass<CopyInsertion>(can_share_buffer,
                                    kNoRegionBasedLiveRangeAnalysisLimit);
  } else {
    pipeline.AddPass<CopyInsertion>(can_share_buffer);
  }

  auto& sub_pipeline =
      pipeline.AddPass<HloPassPipeline>("horizontal-loop-fusion-for-copy");
  sub_pipeline.AddPass<CopyFusion>(device_description);
  sub_pipeline.AddPass<HorizontalLoopFusion>(device_description, "copy_",
                                             /*only_entry_computation=*/true);
  sub_pipeline.AddPass<HloDCE>();

  // TODO(chokobole): Uncomment this. Dependency: SanitizeConstantNames
  // pipeline.AddPass<SanitizeConstantNames>();

  return pipeline;
}

}  // namespace zkx::gpu
