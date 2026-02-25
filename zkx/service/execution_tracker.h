/* Copyright 2017 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_EXECUTION_TRACKER_H_
#define ZKX_SERVICE_EXECUTION_TRACKER_H_

#include <map>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

#include "zkx/executable_run_options.h"
#include "zkx/service/backend.h"
#include "zkx/service/stream_pool.h"
#include "zkx/stream_executor/stream_executor.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

// Represents an asynchronously launched execution. Owns the stream (from the
// passed run_options->stream()) on which the execution is launched and releases
// the stream when destructed.
class AsyncExecution {
 public:
  AsyncExecution(Backend* backend, std::vector<StreamPool::Ptr> streams,
                 const ExecutionProfile& profile, GlobalDataHandle result);

  absl::Status BlockUntilDone() const;

  const GlobalDataHandle& result() const { return result_; }

  const ExecutionProfile& profile() const { return profile_; }

 private:
  // Backend to execute the computation on.
  Backend* backend_;  // not owned

  // Stream on which the execution is launched.
  std::vector<StreamPool::Ptr> streams_;

  // Profile object of the execution to be returned to the user.
  ExecutionProfile profile_;

  // Data handle to the result of the execution. Data represented by this handle
  // is valid only after BlockUntilDone() is called.
  GlobalDataHandle result_;
};

// Tracks asynchronously launched executions for the ZKX service.
class ExecutionTracker {
 public:
  ExecutionTracker();

  // Registers an execution with its backend, streams, and data handle to the
  // execution result. Returns a handle for the registered execution.
  ExecutionHandle Register(Backend* backend,
                           std::vector<StreamPool::Ptr> stream,
                           const ExecutionProfile& profile,
                           GlobalDataHandle data);

  // Unregisters the execution for the given handle.
  absl::Status Unregister(const ExecutionHandle& handle);

  // Resolves the given ExecutionHandle to an AsyncExecution. Returns an
  // error status if the given handle is not found, which means that the
  // execution is not yet registered or already unregistered.
  absl::StatusOr<const AsyncExecution*> Resolve(const ExecutionHandle& handle);

 private:
  // The next handle to assign to an execution.
  int64_t next_handle_ ABSL_GUARDED_BY(execution_mutex_);

  // Mapping from ExecutionHandle handle to the corresponding registered
  // AsyncExecution object.
  std::map<int64_t, std::unique_ptr<AsyncExecution>> handle_to_execution_
      ABSL_GUARDED_BY(execution_mutex_);

  absl::Mutex execution_mutex_;  // Guards the execution mapping.

  ExecutionTracker(const ExecutionTracker&) = delete;
  ExecutionTracker& operator=(const ExecutionTracker&) = delete;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_EXECUTION_TRACKER_H_
