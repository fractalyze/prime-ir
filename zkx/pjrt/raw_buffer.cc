/* Copyright 2025 The OpenXLA Authors.
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

#include "zkx/pjrt/raw_buffer.h"

#include <vector>

namespace zkx {
namespace {

std::vector<RegisterRawBufferFactory::FactoryFuncT>& GetFactoryFuncs() {
  static auto* funcs = new std::vector<RegisterRawBufferFactory::FactoryFuncT>;
  return *funcs;
}

}  // namespace

absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>
PjRtRawBuffer::CreateRawAliasOfBuffer(PjRtBuffer* buffer) {
  for (auto* func : GetFactoryFuncs()) {
    auto res = (*func)(buffer);
    if (res.has_value()) {
      return *res;
    }
  }
  return absl::UnimplementedError("CreateRawAliasOfBuffer not implemented.");
}

RegisterRawBufferFactory::RegisterRawBufferFactory(
    RegisterRawBufferFactory::FactoryFuncT func) {
  GetFactoryFuncs().push_back(func);
}

}  // namespace zkx
