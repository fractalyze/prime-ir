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

#ifndef ZKX_FFI_ATTRIBUTE_MAP_H_
#define ZKX_FFI_ATTRIBUTE_MAP_H_

#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "zkx/ffi/call_frame.h"

namespace zkx::ffi {

// Converts MLIR dictionary attribute attached to a custom call operation to a
// custom call handler attributes that are forwarded to the FFI handler.
absl::StatusOr<CallFrameBuilder::AttributesMap> BuildAttributesMap(
    mlir::DictionaryAttr dict);

}  // namespace zkx::ffi

#endif  // ZKX_FFI_ATTRIBUTE_MAP_H_
