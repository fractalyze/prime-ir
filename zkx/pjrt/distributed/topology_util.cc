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

#include "zkx/pjrt/distributed/topology_util.h"

#include <fstream>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/thread_pool.h"
#include "zkx/pjrt/utils.h"

namespace zkx {

namespace {

bool SameDevice(const DeviceProto& a, const DeviceProto& b) {
  return (a.name() == b.name() && a.vendor() == b.vendor() &&
          a.local_device_ordinal() == b.local_device_ordinal() &&
          a.core_count() == b.core_count() &&
          a.device_kind() == b.device_kind() &&
          a.slice_index() == b.slice_index() &&
          // Global device ID Might not be set for LocalTopologyProto, still
          // check it for default value.
          a.global_device_id() == b.global_device_id() &&
          a.compute_capability() == b.compute_capability());
}

bool SameLocalTopology(const LocalTopologyProto& a,
                       const LocalTopologyProto& b) {
  if (a.node_id() != b.node_id() || a.devices_size() != b.devices_size()) {
    return false;
  }
  for (int i = 0; i < a.devices_size(); ++i) {
    if (!SameDevice(a.devices(i), b.devices(i))) {
      return false;
    }
  }
  return true;
}

std::string GetLocalTopologyKey(std::string_view platform, int node_id) {
  return absl::StrCat("local_topology/", platform, "/", node_id);
}

std::string GetGlobalTopologyKey(std::string_view platform) {
  return absl::StrCat("global_topology/", platform);
}

}  // namespace

// Retrieve content of /proc/sys/kernel/random/boot_id as a string.
// Note that procfs file may have file size 0 which throws off generic file
// readers such as tsl::ReadFileToString.
absl::StatusOr<std::string> GetBootIdString() {
  std::string boot_id_str;
#ifdef __linux__
  // Exists on Linux systems. Unique per OS kernel restart.
  constexpr char kBootIdPath[] = "/proc/sys/kernel/random/boot_id";

  std::ifstream file(kBootIdPath);
  if (!file) {
    return absl::NotFoundError(absl::StrFormat("%s not found.", kBootIdPath));
  }
  std::string line;
  while (std::getline(file, line)) {
    absl::StripAsciiWhitespace(&line);
    absl::StrAppend(&boot_id_str, line);
  }
#endif
  return boot_id_str;
}

namespace {

absl::StatusOr<std::vector<LocalTopologyProto>> GetAllLocalTopologies(
    std::string_view platform, int num_nodes, KeyValueStoreInterface* kv_store,
    absl::Duration timeout) {
  std::vector<absl::StatusOr<std::string>> local_topology_strs(num_nodes);

  // TODO(ezhulenev): Should a thread pool become a function argument?
  tsl::thread::ThreadPool thread_pool(
      tsl::Env::Default(), "GetAllLocalTopologies", DefaultThreadPoolSize());

  absl::BlockingCounter blocking_counter(num_nodes);
  absl::Mutex mu;
  for (int i = 0; i < num_nodes; i++) {
    thread_pool.Schedule([&, i] {
      absl::StatusOr<std::string> local_topology_str =
          kv_store->Get(GetLocalTopologyKey(platform, i), timeout);
      {
        absl::MutexLock lock(&mu);
        local_topology_strs[i] = local_topology_str;
      }
      blocking_counter.DecrementCount();
    });
  }
  blocking_counter.Wait();

  std::vector<std::string> error_messages;
  std::vector<LocalTopologyProto> local_topologies;
  int max_num_failed_message = 10;
  int failed_count = 0;
  for (const absl::StatusOr<std::string>& str : local_topology_strs) {
    if (str.ok()) {
      LocalTopologyProto local;
      local.ParseFromString(*str);
      local_topologies.push_back(local);
    } else {
      error_messages.push_back(
          absl::StrCat("Error ", ++failed_count, ": ", str.status().message()));
      if (failed_count > max_num_failed_message) {
        break;
      }
    }
  }
  if (error_messages.empty()) {
    return local_topologies;
  }
  return absl::InternalError(
      absl::StrCat("Getting local topologies failed: ",
                   absl::StrJoin(error_messages, "\n\n")));
}

}  // namespace

// Steals the contents of `local_topologies`.
GlobalTopologyProto BuildGlobalTopology(
    absl::Span<LocalTopologyProto> local_topologies,
    bool assign_global_device_ids) {
  GlobalTopologyProto global_topology;
  int next_global_device_id = 0;
  // Assign local devices of the same host to the same slice_index.
  int next_slice_index = 0;
  absl::flat_hash_map<std::string, int> boot_id_to_slice_index;
  for (LocalTopologyProto& local : local_topologies) {
    // Every new boot_id seen is treated as a new host/slice.
    std::string_view boot_id = local.boot_id();
    auto [it, inserted] =
        boot_id_to_slice_index.try_emplace(boot_id, next_slice_index);
    if (inserted) {
      ++next_slice_index;
    }
    for (DeviceProto& device : *local.mutable_devices()) {
      if (assign_global_device_ids) {
        device.set_global_device_id(next_global_device_id++);
      }
      device.set_slice_index(it->second);
    }
    global_topology.add_nodes()->Swap(&local);
  }
  if (VLOG_IS_ON(10)) {
    for (auto it = boot_id_to_slice_index.begin();
         it != boot_id_to_slice_index.end(); ++it) {
      LOG(INFO) << "BuildGlobalTopology boot_id_to_slice_index " << it->first
                << "->" << it->second;
    }
  }
  return global_topology;
}

absl::Status ExchangeTopologies(std::string_view platform, int node_id,
                                int num_nodes,
                                absl::Duration get_local_topology_timeout,
                                absl::Duration get_global_topology_timeout,
                                KeyValueStoreInterface* kv_store,
                                const LocalTopologyProto& local_topology,
                                GlobalTopologyProto* global_topology,
                                bool assign_global_device_ids) {
  VLOG(3) << "Local Topology for platform" << platform << ":\n"
          << local_topology.DebugString();
  if (num_nodes == 1) {
    LocalTopologyProto* topology = global_topology->add_nodes();
    *topology = local_topology;
    for (DeviceProto& device : *topology->mutable_devices()) {
      device.set_global_device_id(device.local_device_ordinal());
    }
    return absl::OkStatus();
  }
  CHECK_NE(kv_store, nullptr);
  const std::string local_topology_key = GetLocalTopologyKey(platform, node_id);
  const std::string serialized_local_topology =
      local_topology.SerializeAsString();

  absl::Status status = kv_store->Set(GetLocalTopologyKey(platform, node_id),
                                      serialized_local_topology);
  if (absl::IsAlreadyExists(status)) {
    // Local topology has been set previously from the same node before
    // restart.
    absl::StatusOr<std::string> existing_local_topology =
        kv_store->TryGet(local_topology_key);
    LocalTopologyProto existing_local_topology_proto;
    existing_local_topology_proto.ParseFromString(*existing_local_topology);
    if (!SameLocalTopology(existing_local_topology_proto, local_topology)) {
      return absl::InternalError(absl::Substitute(
          "Different local topology for node $0 has been set previously, "
          "possibly before a restart.\nBefore: $1\nAfter: $2",
          node_id, existing_local_topology_proto.DebugString(),
          local_topology.DebugString()));
    }
  } else if (!status.ok()) {
    return status;
  }

  // The lead node gets all local topologies, builds the global topology and
  // puts it to the key-value store.
  std::string global_topology_key = GetGlobalTopologyKey(platform);
  if (node_id == 0) {
    TF_ASSIGN_OR_RETURN(std::vector<LocalTopologyProto> local_topologies,
                        GetAllLocalTopologies(platform, num_nodes, kv_store,
                                              get_local_topology_timeout));
    *global_topology =
        BuildGlobalTopology(absl::Span<LocalTopologyProto>(local_topologies),
                            assign_global_device_ids);
    TF_RETURN_IF_ERROR(kv_store->Set(global_topology_key,
                                     global_topology->SerializeAsString()));
  } else {
    TF_ASSIGN_OR_RETURN(
        std::string global_topology_str,
        kv_store->Get(global_topology_key, get_global_topology_timeout));
    global_topology->ParseFromString(global_topology_str);
  }
  VLOG(3) << "Global topology for platform " << platform << ":\n"
          << global_topology->DebugString();
  return absl::OkStatus();
}

}  // namespace zkx
