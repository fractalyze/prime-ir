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

#include "zkx/service/gpu/fusion_pipeline.h"

#include <utility>

#include "zkx/hlo/pass/hlo_pass_fix.h"
#include "zkx/hlo/transforms/simplifiers/hlo_cse.h"
#include "zkx/hlo/transforms/simplifiers/hlo_dce.h"
#include "zkx/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "zkx/service/gpu/transforms/horizontal_input_fusion.h"
#include "zkx/service/gpu/transforms/horizontal_loop_fusion.h"
#include "zkx/service/gpu/transforms/multi_output_fusion.h"
#include "zkx/service/gpu/transforms/poseidon2_fusion_rewriter.h"
#include "zkx/service/gpu/transforms/priority_fusion.h"

namespace zkx::gpu {

HloPassPipeline FusionPipeline(
    const DebugOptions& debug_options,
    HloCostAnalysis::ShapeSizeFunction shape_size_bytes_function,
    tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& gpu_device_info) {
  HloPassFix<HloPassPipeline> fusion("fusion");

  // We try to split variadic ops with many parameters into several such ops
  // to avoid exceeding the parameter space.
  // TODO(batzor): Uncomment this. Dependency: VariadicOpSplitter
  // fusion.AddPass<VariadicOpSplitter>();

  // TODO(batzor): Uncomment this. Dependency: HloVerifier with
  // CpuGpuVerifierMetadata
  // HloVerifierOpts opts =
  //     HloVerifierOpts().MakeLayoutSensitive().WithInstructionCanChangeLayout(
  //         LayoutAssignment::InstructionCanChangeLayout);
  // opts.verify_unique_channel_ids = !debug_options.zkx_ignore_channel_id();
  // fusion.AddInvariantCheckerDebug<HloVerifier>(
  //     std::make_unique<CpuGpuVerifierMetadata>(std::move(opts)),
  //     "hlo verifier (debug)");

  // Detect Poseidon2 permutation pattern and wrap into a single kCustom fusion
  // before PriorityFusion can split it into multiple kernels.
  fusion.AddPass<Poseidon2FusionRewriter>();
  fusion.AddPass<HloDCE>();

  GpuHloCostAnalysis::Options cost_analysis_options{
      shape_size_bytes_function,
      /*per_second_rates=*/{},
      /*min_latencies_seconds=*/{},
      /*count_multiple_input_accesses=*/true};
  fusion.AddPass<PriorityFusion>(thread_pool, gpu_device_info,
                                 std::move(cost_analysis_options));

  // Running CSE affects how many users an op has. This plays a role in what
  // we detect as a tiled transpose fusion.
  fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                         /*only_fusion_computations=*/true);
  fusion.AddPass<HloDCE>();

  fusion.AddPass<MultiOutputFusion>(gpu_device_info, shape_size_bytes_function);

  fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                         /*only_fusion_computations=*/true);
  fusion.AddPass<HloDCE>();

  return std::move(fusion);
}

HloPassPipeline HorizontalFusionPipeline(
    const se::DeviceDescription& gpu_device_info) {
  HloPassFix<HloPassPipeline> horizontal_fusion("horizontal fusion");

  horizontal_fusion.AddPass<HorizontalLoopFusion>(gpu_device_info);
  horizontal_fusion.AddPass<HorizontalInputFusion>(gpu_device_info);

  horizontal_fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                                    /*only_fusion_computations=*/true);
  horizontal_fusion.AddPass<HloDCE>();

  return std::move(horizontal_fusion);
}

}  // namespace zkx::gpu
