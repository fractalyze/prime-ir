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

#include "zkx/ffi/type_id_registry.h"

#include <atomic>
#include <string>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

namespace zkx::ffi {

ABSL_CONST_INIT absl::Mutex type_registry_mutex(absl::kConstInit);

using ExternalTypeIdRegistry =
    absl::flat_hash_map<std::string, TypeIdRegistry::TypeId>;

namespace {

ExternalTypeIdRegistry& StaticExternalTypeIdRegistry() {
  static auto* registry = new ExternalTypeIdRegistry();
  return *registry;
}

}  // namespace

// static
TypeIdRegistry::TypeId TypeIdRegistry::GetNextTypeId() {
  static auto* counter = new std::atomic<int64_t>(1);
  return TypeId(counter->fetch_add(1));
}

absl::StatusOr<TypeIdRegistry::TypeId> TypeIdRegistry::RegisterExternalTypeId(
    std::string_view name) {
  absl::MutexLock lock(&type_registry_mutex);
  auto& registry = StaticExternalTypeIdRegistry();

  // Try to emplace with type id zero and fill it with real type id only if we
  // successfully acquired an entry for a given name.
  auto emplaced = registry.emplace(name, TypeId(0));
  if (!emplaced.second) {
    return absl::InternalError(
        absl::StrCat("Type id ", emplaced.first->second.value(),
                     " already registered for type name ", name));
  }
  return emplaced.first->second = GetNextTypeId();
}

}  // namespace zkx::ffi
