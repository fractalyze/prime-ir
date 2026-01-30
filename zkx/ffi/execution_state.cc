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

#include "zkx/ffi/execution_state.h"

#include <utility>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"

#include "zkx/ffi/type_id_registry.h"

namespace zkx::ffi {

ExecutionState::ExecutionState()
    : type_id_(TypeIdRegistry::kUnknownTypeId),
      state_(nullptr),
      deleter_(nullptr) {}

ExecutionState::~ExecutionState() {
  if (deleter_) deleter_(state_);
}

absl::Status ExecutionState::Set(TypeId type_id, void* state,
                                 Deleter<void> deleter) {
  DCHECK(state && deleter) << "State and deleter must not be null";

  if (type_id_ != TypeIdRegistry::kUnknownTypeId) {
    return absl::FailedPreconditionError(
        absl::StrCat("State is already set with a type id ", type_id_.value()));
  }

  type_id_ = type_id;
  state_ = state;
  deleter_ = std::move(deleter);

  return absl::OkStatus();
}

// Returns opaque state of the given type id. If set state type id does not
// match the requested one, returns an error.
absl::StatusOr<void*> ExecutionState::Get(TypeId type_id) const {
  if (type_id_ == TypeIdRegistry::kUnknownTypeId) {
    return absl::NotFoundError("State is not set");
  }

  if (type_id_ != type_id) {
    return absl::InvalidArgumentError(
        absl::StrCat("Set state type id ", type_id_.value(),
                     " does not match the requested one ", type_id.value()));
  }

  return state_;
}

bool ExecutionState::IsSet() const {
  return type_id_ != TypeIdRegistry::kUnknownTypeId;
}

}  // namespace zkx::ffi
