/* Copyright 2023 The OpenXLA Authors.
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

#include "zkx/backends/cpu/collectives/gloo_collectives.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/prefix_store.h"

#include "zkx/backends/cpu/collectives/gloo_communicator.h"

namespace zkx::cpu {

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
GlooCollectives::CreateCommunicators(const CliqueKey& clique_key,
                                     const std::optional<CliqueIds>& clique_ids,
                                     absl::Span<const DeviceRank> ranks,
                                     const Config& config) {
  std::vector<std::unique_ptr<Communicator>> communicators;
  for (auto& device_rank : ranks) {
    size_t rank = device_rank.rank.value();

    auto gloo_context = std::make_shared<gloo::rendezvous::Context>(
        rank, clique_key.num_devices());
    auto prefix_store = gloo::rendezvous::PrefixStore(
        absl::StrCat("gloo/",
                     absl::StrJoin(clique_key.devices(), ",",
                                   [](std::string* out, GlobalDeviceId id) {
                                     absl::StrAppend(out, id.value());
                                   })),
        *store_);

    try {
      gloo_context->connectFullMesh(prefix_store, device_);
    } catch (std::exception& e) {
      return absl::UnknownError(
          absl::StrCat("Gloo context initialization failed: ", e.what()));
    }

    communicators.push_back(std::make_unique<GlooCommunicator>(
        std::move(gloo_context), rank, clique_key.num_devices()));
  }

  return communicators;
}

}  // namespace zkx::cpu
