/* Copyright 2024 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_GPU_FUSION_PROCESS_DUMP_H_
#define ZKX_SERVICE_GPU_FUSION_PROCESS_DUMP_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/gpu/fusion_process_dump.pb.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// Helper class to work with fusion process dump.
class FusionProcessDump {
 public:
  static absl::StatusOr<FusionProcessDump> LoadFromFile(
      const std::string& path);
  static absl::StatusOr<FusionProcessDump> LoadFromData(
      const std::string& data, std::string_view format);
  static absl::StatusOr<FusionProcessDump> LoadFromProto(
      const FusionProcessDumpProto& fusion_process_dump_proto);

  const FusionProcessDumpProto& proto() { return fusion_process_dump_proto_; }

  HloModule* module() { return hlo_module_.get(); }

  const se::DeviceDescription& device_info() { return device_info_; }

  int64_t current_step_idx() { return current_step_idx_; }

  // Returns computation that contains producer (and other instructions) of the
  // current step.
  HloComputation* GetCurrentComputation();

  // Returns the instruction with `name`.
  HloInstruction* GetInstructionWithName(std::string_view name);

  // Returns producer of the current step. Should not be null, since all step
  // types have a producer.
  HloInstruction* GetProducer();

  // Returns a list of consumers of the current step. The list contains one
  // instruction is the current step is fusion. The list is empty if the current
  // step is `producer_ineligible`.
  absl::InlinedVector<HloInstruction*, 2> GetConsumers();

  // Returns result instruction of the last fusion step. Returns nullptr before
  // the first fusion.
  HloInstruction* GetLastFusion() { return last_fusion_; }

  // Returns current step. If current step is `fusion`, the `module` is in the
  // state *before* the fusion. Next call to `FusionProcessDump::Advance` will
  // actually perform the fusion.
  const FusionStep& CurrentStep();

  // Returns true if there are fusion steps.
  bool HasNext();

  // Advances to the next fusion step. If current step is `fusion`, modifies the
  // `module` accordingly.
  void Advance();

 private:
  FusionProcessDump(FusionProcessDumpProto fusion_process_dump_proto,
                    std::unique_ptr<HloModule> hlo_module,
                    se::DeviceDescription device_info,
                    absl::flat_hash_map<std::string, HloComputation*>
                        instruction_name_to_computation_map)
      : fusion_process_dump_proto_(std::move(fusion_process_dump_proto)),
        hlo_module_(std::move(hlo_module)),
        device_info_(std::move(device_info)),
        instruction_name_to_computation_map_(
            std::move(instruction_name_to_computation_map)) {}

  FusionProcessDumpProto fusion_process_dump_proto_;
  std::unique_ptr<HloModule> hlo_module_;
  se::DeviceDescription device_info_;

  // A map from instructions to computations. HLO module doesn't have a
  // convenient way to get an instruction by name. This map saves the need to
  // iterator over all computations in the module.
  absl::flat_hash_map<std::string, HloComputation*>
      instruction_name_to_computation_map_;

  // Index of the current step.
  int64_t current_step_idx_ = 0;

  // Tracks result of the last fusion step.
  HloInstruction* last_fusion_ = nullptr;  // not owned
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_FUSION_PROCESS_DUMP_H_
