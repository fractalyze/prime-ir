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

#ifndef ZKX_EXECUTABLE_RUN_OPTIONS_H_
#define ZKX_EXECUTABLE_RUN_OPTIONS_H_

#include <stdint.h>

#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/statusor.h"

// These classes are forward declared so that ExecutableRunOptions can be linked
// into a ZKX-compiled binary without having to link all of the pointed-to
// objects (e.g., for an ahead-of-time compiled CPU binary, the gpu tools don't
// need to be linked).
namespace stream_executor {
class Stream;
class Event;
class Platform;
class DeviceMemoryAllocator;
class DeviceMemoryBase;
}  // namespace stream_executor

namespace se = ::stream_executor;

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace tsl {
template <typename T>
class AsyncValueRef;
}  // namespace tsl

namespace zkx {

class DeviceAssignment;
class ExecutionProfile;
class Shape;

namespace cpu {
class CpuExecutableRunOptions;
}  // namespace cpu

namespace gpu {
class GpuExecutableRunOptions;
}  // namespace gpu

namespace ffi {
class ExecutionContext;
}  // namespace ffi

// A unique identifier for a particular "logical execution" of a ZKX model.
//
// A logical execution might encompass multiple executions of one or more
// HloModules.  Runs that are part of the same logical execution can
// communicate via collective ops (e.g. kAllToAll), whereas runs that are part
// of different logical executions are isolated.
class RunId {
 public:
  // Creates a new, unique RunId.
  RunId();
  explicit RunId(int64_t value) : data_(value) {}

  RunId(const RunId&) = default;
  RunId& operator=(const RunId&) = default;
  bool operator==(const RunId& other) const { return data_ == other.data_; }
  std::string ToString() const;
  int64_t ToInt() const;

  template <typename H>
  friend H AbslHashValue(H h, const RunId& id) {
    return H::combine(std::move(h), id.data_);
  }

 private:
  int64_t data_;
};

// Callback used by the GPU backend only. This is an "one-sided" version of
// ThenDoHostCallback that enqueues a callback onto a stream. The difference
// with ThenDoHostCallback is that the device does not block waiting for the
// callback to complete; instead the callback is scheduled by the runtime.
// This functionality must be provided by the caller, and hence is provided in
// callback form.
using ThenExecuteFunction =
    std::function<void(se::Stream*, std::function<void()>)>;

// Callback for sending device buffer to a channel. Returned event will be
// recorded on a `stream` once the send operation is completed and data was
// copied from the `src` memory. `frontend_attrs` contains frontend specific
// attributes for the send.
using SendDeviceMemoryFunction = std::function<
    absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>>(
        int64_t channel_id, se::Stream* stream, const Shape& shape,
        const se::DeviceMemoryBase& src,
        const absl::flat_hash_map<std::string, std::string>& frontend_attrs)>;

// Callback for receiving device buffer from a channel. Returned event will be
// recorded on a `stream` once the recv operation is completed and data was
// copied into the `dst` memory. `frontend_attrs` contains frontend specific
// attributes for the receive.
using RecvDeviceMemoryFunction = std::function<
    absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>>(
        int64_t channel_id, se::Stream* stream, const Shape& shape,
        se::DeviceMemoryBase* dst,
        const absl::flat_hash_map<std::string, std::string>& frontend_attrs)>;

// Class containing options for running a LocalExecutable.
class ExecutableRunOptions {
 public:
  // Specifies the allocator to use during execution.
  ExecutableRunOptions& set_allocator(se::DeviceMemoryAllocator* allocator) {
    allocator_ = allocator;
    return *this;
  }
  se::DeviceMemoryAllocator* allocator() const { return allocator_; }

  // If set, this is the device to run the computation on. Valid device_ordinal
  // values are: 0 to # of devices - 1. These are the logical device ordinals,
  // since multiple logical devices could reside on the same physical device,
  // e.g., virtual GPUs. If there is only one logical device on a physical
  // device, then these values are identical to the device ordinal values used
  // by StreamExecutor. The device must be of the same type as the executable
  // was compiled for. A value of -1 indicates this option has not been set.
  ExecutableRunOptions& set_device_ordinal(int device_ordinal) {
    device_ordinal_ = device_ordinal;
    return *this;
  }
  int device_ordinal() const { return device_ordinal_; }

  // If set, this is the physical device to run the computation on. These values
  // are identical to the device ordinal values used by StreamExecutor. The
  // device must be of the same type as the executable was compiled for. A value
  // of -1 indicates this option has not been set, in which case the physical
  // device ordinal is the same as the logical device ordinal.
  ExecutableRunOptions& set_physical_device_ordinal(
      int physical_device_ordinal) {
    physical_device_ordinal_ = physical_device_ordinal;
    return *this;
  }
  int physical_device_ordinal() const { return physical_device_ordinal_; }

  // If set, this is the stream to run the computation on. The platform of the
  // stream must match the platform the executable was built for. A value of
  // nullptr indicates the option has not been set.
  ExecutableRunOptions& set_stream(se::Stream* stream) {
    stream_ = stream;
    return *this;
  }
  se::Stream* stream() const { return stream_; }

  // If set, this is the stream to perform host to device transfers on (e.g. any
  // pre-computation transfers). The platform of the stream must match the
  // platform the executable was built for. A value of nullptr indicates the
  // option has not been set.
  ExecutableRunOptions& set_host_to_device_stream(se::Stream* stream) {
    host_to_device_stream_ = stream;
    return *this;
  }
  se::Stream* host_to_device_stream() const { return host_to_device_stream_; }

  // If set, this is the stream to perform device to host transfers on.
  // The platform of the stream must match the platform the executable was
  // built for. A value of nullptr indicates the option has not been set.
  ExecutableRunOptions& set_device_to_host_stream(se::Stream* stream) {
    device_to_host_stream_ = stream;
    return *this;
  }
  se::Stream* device_to_host_stream() const { return device_to_host_stream_; }

  // Sets the thread pool device on which to run Eigen sub computations.
  //
  // This field must be set for ZKX:CPU models that call Eigen routines, but may
  // be null otherwise. Routines that use this field should always CHECK (or
  // TF_RET_CHECK) that it's not null before dereferencing it, so that users get
  // a clean crash rather than a segfault.
  //
  // Does not take ownership.
  ExecutableRunOptions& set_intra_op_thread_pool(
      const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
    intra_op_thread_pool_ = intra_op_thread_pool;
    return *this;
  }
  const Eigen::ThreadPoolDevice* intra_op_thread_pool() const {
    return intra_op_thread_pool_;
  }

  // If set, profiling information is written to 'profile'.
  ExecutableRunOptions& set_execution_profile(ExecutionProfile* profile) {
    execution_profile_ = profile;
    return *this;
  }
  ExecutionProfile* execution_profile() const { return execution_profile_; }

  ExecutableRunOptions& set_device_assignment(
      const DeviceAssignment* device_assignment) {
    device_assignment_ = device_assignment;
    return *this;
  }
  const DeviceAssignment* device_assignment() const {
    return device_assignment_;
  }

  ExecutableRunOptions& set_rng_seed(int rng_seed) {
    rng_seed_ = rng_seed;
    return *this;
  }
  int rng_seed() const { return rng_seed_; }

  ExecutableRunOptions& set_launch_id(int32_t launch_id) {
    launch_id_ = launch_id;
    return *this;
  }
  int32_t launch_id() const { return launch_id_; }

  ExecutableRunOptions& set_run_id(RunId id) {
    run_id_ = id;
    return *this;
  }
  RunId run_id() const { return run_id_; }

  // See documentation on ThenExecuteFunction.
  ExecutableRunOptions& set_then_execute_function(ThenExecuteFunction* f) {
    then_execute_function_ = f;
    return *this;
  }
  ThenExecuteFunction* then_execute_function() const {
    return then_execute_function_;
  }

  // See documentation on SendDeviceMemoryFunction.
  ExecutableRunOptions& set_send_device_memory_function(
      SendDeviceMemoryFunction* f) {
    send_device_memory_function_ = f;
    return *this;
  }
  SendDeviceMemoryFunction* send_device_memory_function() const {
    return send_device_memory_function_;
  }

  // See documentation on RecvDeviceMemoryFunction.
  ExecutableRunOptions& set_recv_device_memory_function(
      RecvDeviceMemoryFunction* f) {
    recv_device_memory_function_ = f;
    return *this;
  }
  RecvDeviceMemoryFunction* recv_device_memory_function() const {
    return recv_device_memory_function_;
  }

  // CPU-backend specific options. These are kept out-of-line to avoid bloating
  // the size of this dependency for CPU-only AOT builds.
  ExecutableRunOptions& set_cpu_executable_run_options(
      const cpu::CpuExecutableRunOptions* cpu_executable_run_options) {
    cpu_executable_run_options_ = cpu_executable_run_options;
    return *this;
  }
  const cpu::CpuExecutableRunOptions* cpu_executable_run_options() const {
    return cpu_executable_run_options_;
  }

  // GPU-backend specific options. These are kept out-of-line to avoid bloating
  // the size of this dependency for CPU-only AOT builds.
  ExecutableRunOptions& set_gpu_executable_run_options(
      const gpu::GpuExecutableRunOptions* gpu_executable_run_options) {
    gpu_executable_run_options_ = gpu_executable_run_options;
    return *this;
  }
  const gpu::GpuExecutableRunOptions* gpu_executable_run_options() const {
    return gpu_executable_run_options_;
  }

  // ZKX FFI specific execution context that allows to pass auxiliary data to
  // FFI handlers. It's a caller responsibility to ensure that the ZKX FFI
  // execution context stays alive while the executable is running.
  ExecutableRunOptions& set_ffi_execution_context(
      const ffi::ExecutionContext* ffi_execution_context) {
    ffi_execution_context_ = ffi_execution_context;
    return *this;
  }
  const ffi::ExecutionContext* ffi_execution_context() const {
    return ffi_execution_context_;
  }

  // This indicates how many local devices are used by the execution.
  // Valid values are any value greater than 0.
  // 0 means unset.
  ExecutableRunOptions& set_local_device_count(int local_device_count) {
    local_device_count_ = local_device_count;
    return *this;
  }
  int local_device_count() const { return local_device_count_; }

 private:
  se::DeviceMemoryAllocator* allocator_ = nullptr;  // not owned
  int device_ordinal_ = -1;
  int local_device_count_ = 0;
  int physical_device_ordinal_ = -1;
  const DeviceAssignment* device_assignment_ = nullptr;
  se::Stream* stream_ = nullptr;  // not owned
  const Eigen::ThreadPoolDevice* intra_op_thread_pool_ = nullptr;
  ExecutionProfile* execution_profile_ = nullptr;  // not owned
  int rng_seed_ = 0;
  int32_t launch_id_ = 0;
  se::Stream* device_to_host_stream_ = nullptr;           // not owned
  se::Stream* host_to_device_stream_ = nullptr;           // not owned
  ThenExecuteFunction* then_execute_function_ = nullptr;  // not owned
  SendDeviceMemoryFunction* send_device_memory_function_ =
      nullptr;  // not owned
  RecvDeviceMemoryFunction* recv_device_memory_function_ =
      nullptr;  // not owned
  RunId run_id_;
  const cpu::CpuExecutableRunOptions* cpu_executable_run_options_ = nullptr;
  const gpu::GpuExecutableRunOptions* gpu_executable_run_options_ = nullptr;
  const ffi::ExecutionContext* ffi_execution_context_ = nullptr;
};

}  // namespace zkx

#endif  // ZKX_EXECUTABLE_RUN_OPTIONS_H_
