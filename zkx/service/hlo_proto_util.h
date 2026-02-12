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

// Utilities to manipulate data in hlo.proto.

#ifndef ZKX_SERVICE_HLO_PROTO_UTIL_H_
#define ZKX_SERVICE_HLO_PROTO_UTIL_H_

#include <vector>

#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/hlo.pb.h"

namespace zkx {

// Returns a serialized representation of the HLO state.
HloProto MakeHloProto(const HloModule& module,
                      const BufferAssignment& assignment);

// Returns a serialized representation of the HLO state, but buffer assignment
// will not be included in the output.
HloProto MakeHloProto(const HloModule& module);

// Create an HLO state from serialized representation. In addition to
// creating the proto with HloModule::CreateFromProto(...) it also
// uses HloVerifier to ensure basic invariants are held.
absl::StatusOr<std::unique_ptr<HloModule>> CreateModuleFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config);

// Returns the shapes of the parameters of the entry computation. Shape pointers
// refer to shapes inside of the given HloProto.
absl::StatusOr<std::vector<const ShapeProto*>> EntryComputationParameterShapes(
    const HloProto& hlo_proto);

// Returns the shape of the output of the entry computation. The shape pointer
// refers to the output shape inside of the given HloProto.
absl::StatusOr<const ShapeProto*> EntryComputationOutputShape(
    const HloProto& hlo_proto);

}  // namespace zkx

#endif  // ZKX_SERVICE_HLO_PROTO_UTIL_H_
