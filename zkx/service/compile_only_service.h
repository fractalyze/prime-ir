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

#ifndef ZKX_SERVICE_COMPILE_ONLY_SERVICE_H_
#define ZKX_SERVICE_COMPILE_ONLY_SERVICE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/service/backend.h"
#include "zkx/service/compiler.h"
#include "zkx/service/service.h"
#include "zkx/stream_executor/stream_executor.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

// A ZKX Service specialization for ahead-of-time compilation.  This only
// instantiates a Compiler object for the relevant platform; it does not
// instantiate or require an execution backend.
class CompileOnlyService : public Service {
 public:
  // Factory for creating a CompileOnlyService. The parameter platform is the
  // platform that the service should target. If platform is null then the
  // default platform is used.
  static absl::StatusOr<std::unique_ptr<CompileOnlyService>> NewService(
      se::Platform* platform);
  static absl::StatusOr<std::unique_ptr<CompileOnlyService>> NewService(
      const ServiceOptions& options);

  // A description of a zkx computation to compile using CompileAheadOfTime.
  struct AotZkxComputationInstance {
    HloModuleProto computation;
    std::vector<const Shape*> argument_layouts;
    Shape result_layout;
  };

  // Compiles a list of zkx computations for ahead-of-time execution.  This is
  // intended for use in static compilation.  See
  // |CompileOnlyClient::CompileAheadOfTime| for additional details.
  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(absl::Span<const AotZkxComputationInstance> computations,
                     const AotCompilationOptions& options,
                     std::unique_ptr<AotCompilationMetadata>* metadata);

  absl::StatusOr<std::vector<DeviceHandle>> GetDeviceHandles(
      int64_t device_count) override {
    return absl::UnimplementedError(
        "CompileOnlyService does not support devices.");
  }

  absl::StatusOr<std::unique_ptr<GlobalData>> TransferToServer(
      const LiteralSlice& literal_slice,
      const DeviceHandle* device_handle) override {
    return absl::UnimplementedError(
        "CompileOnlyService does not support device data transfers.");
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal, int64_t replica_id,
                                const DeviceHandle* device_handle) override {
    return absl::UnimplementedError(
        "CompileOnlyService does not support device data transfers.");
  }

 private:
  explicit CompileOnlyService(const ServiceOptions& options,
                              Compiler* compiler);
  CompileOnlyService(const CompileOnlyService&) = delete;
  void operator=(const CompileOnlyService&) = delete;

  // The compiler for the target platform.  This is included in place of
  // the Service::execute_backend_'s compiler, since execute_backend_ is a
  // nullptr in CompileOnlyService.
  Compiler* compiler_;  // not owned
};

}  // namespace zkx

#endif  // ZKX_SERVICE_COMPILE_ONLY_SERVICE_H_
