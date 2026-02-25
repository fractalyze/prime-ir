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

#ifndef ZKX_CLIENT_COMPILE_ONLY_CLIENT_H_
#define ZKX_CLIENT_COMPILE_ONLY_CLIENT_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/client/client.h"
#include "zkx/hlo/builder/zkx_computation.h"
#include "zkx/service/compile_only_service.h"
#include "zkx/service/compiler.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/shape.h"
#include "zkx/stream_executor/stream_executor.h"
#include "zkx/zkx.pb.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

// A ZKX Client specialization for doing ahead-of-time compilation.  This does
// not require (or attempt to instantiate) an execution-capable backend for the
// relevant platform.
class CompileOnlyClient : public Client {
 public:
  explicit CompileOnlyClient(CompileOnlyService* service)
      : Client(service), compiler_service_(service) {}

  CompileOnlyClient(const CompileOnlyClient&) = delete;
  void operator=(const CompileOnlyClient&) = delete;

  // A description of a zkx computation to compile using CompileAheadOfTime.
  struct AotZkxComputationInstance {
    const ZkxComputation* computation;  // not owned
    // Inform the compiler of the expected layout for arguments.
    std::vector<const Shape*> argument_layouts;  // not owned
    // Specifies the expected result layout.
    const Shape* result_layout;  // not owned
  };

  // Compiles a list of zkx computations for ahead-of-time execution.
  // This is intended for use in static compilation. The |options|
  // parameter describes the target for which the compiler should emit
  // code. |metadata|, if provided, is populated during compilation.
  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      absl::Span<const AotZkxComputationInstance> computations,
      const AotCompilationOptions& options,
      std::unique_ptr<AotCompilationMetadata>* metadata = nullptr);

  // Create a Hlo module config for the given program shape and arguments.
  // execution_options is optional; if not given a default is used.
  absl::StatusOr<std::unique_ptr<HloModuleConfig>> CreateModuleConfig(
      const ProgramShape& program_shape,
      absl::Span<const Shape* const> argument_shapes,
      const ExecutionOptions* execution_options);

  // Returns the size of a pointer in bytes for a given triple.
  static int64_t PointerSizeForTriple(absl::string_view triple);

 private:
  CompileOnlyService* compiler_service_;  // not owned
};

}  // namespace zkx

#endif  // ZKX_CLIENT_COMPILE_ONLY_CLIENT_H_
