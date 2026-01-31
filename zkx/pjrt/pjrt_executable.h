/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_PJRT_PJRT_EXECUTABLE_H_
#define ZKX_PJRT_PJRT_EXECUTABLE_H_

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/client/executable_build_options.h"
#include "zkx/ffi/execution_context.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/pjrt/compile_options.pb.h"
#include "zkx/pjrt/executable_metadata.pb.h"
#include "zkx/pjrt/execute_options.pb.h"
#include "zkx/pjrt/pjrt_common.h"
#include "zkx/pjrt/pjrt_layout.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/compiler.h"
#include "zkx/service/hlo_cost_analysis.h"
#include "zkx/shape.h"

namespace zkx {

// Provides configuration for implementations that support compile and execute
// spanning multiple slices. A slice is a set of devices connected by dedicated
// high speed interconnect. Connectivity between slices is typically over data
// center networks. Concrete implementations of MultiSliceConfig contain
// environment specific information to enable communication between devices on
// different slices. Passed as options during compile and execute.
// Implementations that do not support this are allowed to pass nullptr.
class MultiSliceConfig {
 public:
  virtual ~MultiSliceConfig();

  // Returns the total number of slices.
  virtual int32_t NumSlices() const = 0;

  // Returns the SliceID at this host - an integer in [0, NumSlices)
  virtual int32_t SliceId() const = 0;

  // Returns the number of devices on each slice indexed by SliceId.
  virtual absl::flat_hash_map<int32_t, int32_t> NumDevicesPerSlice() const = 0;

  // Returns a serialized proto representing MultiSliceConfig.
  virtual std::string Serialize() const = 0;
};

struct CompileOptions {
  // The layouts of the arguments that the computation should expect.
  std::optional<std::vector<Shape>> argument_layouts;

  // If true, the supplied computation expects its arguments to be wrapped in a
  // tuple and passed as a single parameter.
  bool parameter_is_tupled_arguments = false;

  // ZKX's compilation time options.
  ExecutableBuildOptions executable_build_options;

  // If true, the executable can be run on any device. May only be true if
  // !executable_build_options.has_device_assignment(), so only applies to
  // single-device executables. Beware: on GPUs, sometimes an executable
  // compiled for one device doesn't run on another.
  bool compile_portable_executable = false;

  // ZKX compilation profile version.
  int64_t profile_version = 0;

  // Set multi_slice_config to trigger compilation for DCN connected multi
  // slice operation.
  const MultiSliceConfig* multi_slice_config = nullptr;

  // Key-value string pairs, parsed in order to set miscellaneous options,
  // overriding if appropriate.
  using OptionOverride = std::variant<std::string, bool, int64_t, double>;
  using EnvironmentOptionOverrides =
      std::vector<std::pair<std::string, OptionOverride>>;
  EnvironmentOptionOverrides env_option_overrides;

  std::optional<Compiler::TargetConfig> target_config;

  // Applies env_option_overrides to executable_build_options.debug_options().
  absl::Status ApplyAllOptionOverrides();

  // Applies a single option to executable_build_options.debug_options().
  absl::Status ApplyOption(const std::string& key, const OptionOverride& value);

  absl::Status ApplyOptionFromString(
      const google::protobuf::FieldDescriptor* field, const std::string& value);

  static absl::StatusOr<
      std::vector<std::pair<std::string, CompileOptions::OptionOverride>>>
  LoadEnvOptionOverrides(
      const google::protobuf::Map<std::string, OptionOverrideProto>&
          env_option_overrides);

  // Serialize the CompileOptions into a CompileOptionsProto.
  absl::StatusOr<CompileOptionsProto> ToProto() const;

  // Deserialize the CompileOptionsProto into a CompileOptions.
  static absl::StatusOr<CompileOptions> FromProto(
      const CompileOptionsProto& proto);
};

struct LoadOptions {
  // Origin of the subslice of the target topology to run computation on.
  struct ComputationOrigin {
    int x = 0;
    int y = 0;
    int z = 0;
  };
  std::optional<ComputationOrigin> computation_origin;

  // multi_slice_config to associate with the executable during load of a multi
  // slice operation.
  const MultiSliceConfig* multi_slice_config = nullptr;
};

class ExecuteContext {
 public:
  virtual ~ExecuteContext() = default;

  ffi::ExecutionContext& ffi_context() { return ffi_context_; }
  const ffi::ExecutionContext& ffi_context() const { return ffi_context_; }

 private:
  // ZKX FFI execution context is a mechanism to attach arbitrary user data to
  // a particular call of PjRtLoadedExecutable::Execute and forward it to custom
  // calls implemented as ZKX FFI handlers.
  ffi::ExecutionContext ffi_context_;
};

struct PjRtTransferMetadata {
  // May be invalid if
  // ExecuteOptions::use_major_to_minor_data_layout_for_callbacks is true for
  // this execution.
  Shape device_shape;
};

class PjRtChunk;
class CopyToDeviceStream;

struct SendCallback {
  int64_t channel_id;
  // The callback for retrieving the send value. It will be invoked once for
  // each invocation of the corresponding Send op in the HLO program (So it can
  // be invoked multiple times if it is in a loop). Currently there is no
  // guarantee that the callback here will be invoked in the same order as their
  // corresponding HLO Send ops. The callback can also return errors to indicate
  // the execution should fail.
  //
  // IMPORTANT: the implementation might NOT signal the error to the execution,
  // and the execution will run to completion with UNDEFINED DATA returned by
  // the callback. If there is any potential control flow that depends on the
  // value of the returned data, an error return is unsafe.
  //
  // TODO(chky): Currently the callback invocation order may not be consistent
  // with the HLO send op invocation order, due to limitations in some PjRt
  // implementation. Consider making it strictly the same order as HLO program.
  std::function<absl::Status(const PjRtTransferMetadata& metadata,
                             PjRtChunk chunk, size_t total_size_in_bytes,
                             bool done)>
      callback;
};

struct RecvCallback {
  int64_t channel_id;
  // The callback for feeding the recv value. It will be invoked once for each
  // invocation of the corresponding Recv op in the HLO program (So it can be
  // invoked multiple times if it is in a loop). Currently there is no
  // guarantee that the callback here will be invoked in the same order as their
  // corresponding HLO Recv ops.
  std::function<void(const PjRtTransferMetadata& metadata,
                     std::unique_ptr<CopyToDeviceStream> stream)>
      callback;
};

struct ExecuteOptions {
  // If true, the client must pass a single PjRtBuffer which contains all of
  // the arguments as a single ZKX tuple, otherwise each argument must be
  // passed in its own PjRtBuffer. May only be true if the executable was
  // compiled with parameter_is_tupled_arguments==true.
  bool arguments_are_tupled = false;
  // If true, the computation must return a tuple, which will be destructured
  // into its elements.
  bool untuple_result = false;
  // If non-zero, identifies this execution as part of a potentially
  // multi-device launch. This can be used to detect scheduling errors, e.g. if
  // multi-host programs are launched in different orders on different hosts,
  // the launch IDs may be used by the runtime to detect the mismatch.
  int32_t launch_id = 0;
  // If non-null, an opaque context passed to an execution that may be used to
  // supply additional arguments to a derived class of PjRtExecutable. It is
  // a caller responsibility to ensure that the context is valid for the
  // duration of the execution.
  const ExecuteContext* context = nullptr;
  // If true, check that the PjRtBuffer argument shapes match the compiled
  // shapes. Otherwise, any shape with the right size on device may be passed.
  bool strict_shape_checking = true;

  // Set multi_slice_config when the computation spans multiple slices. The
  // config should match what was used during compilation to generate this
  // executable.
  const MultiSliceConfig* multi_slice_config = nullptr;

  // The send/recv callbacks for PjRt execution. The first level span is for
  // multi-device parallel execution, the second level vector contains the
  // callbacks for all send/recv ops in the executable. These callbacks can be
  // stateful and the user code is responsible for managing the states here.
  // These callbacks must outlive the execution.
  absl::Span<const std::vector<SendCallback>> send_callbacks;
  absl::Span<const std::vector<RecvCallback>> recv_callbacks;

  // If true, send callbacks are passed PjRtChunks in major-to-minor layout, and
  // recv functions should pass major-to-minor chunks to
  // CopyToDeviceStream::AddChunk.
  //
  // If false, send callbacks are passed PjRtChunks in the on-device layout
  // specified in the PjRtTransferMetadata, and recv functions should similarly
  // pass device-layout chunks to CopyToDeviceStream::AddChunk.
  bool use_major_to_minor_data_layout_for_callbacks = false;

  // The `execution_mode` decides whether the execution will be invoked in the
  // caller thread or launched to a separate thread. By default, the
  // implementation may choose either strategy or use a heuristic to decide.
  // Currently it is only applied to CPU implementations
  enum class ExecutionMode { kDefault = 0, kSynchronous, kAsynchronous };
  ExecutionMode execution_mode = ExecutionMode::kDefault;

  // If not null, measure the execution profile and store it.
  ExecutionProfile* execution_profile = nullptr;

  // A set of indices denoting the input buffers that should not be donated.
  // An input buffer may be non-donable, for example, if it is referenced more
  // than once. Since such runtime information is not available at compile time,
  // the compiler might mark the input as `may-alias`, which could lead PjRt to
  // donate the input buffer when it should not. By defining this set of
  // indices, a higher-level PjRt caller can instruct PjRtClient not to donate
  // specific input buffers.
  absl::flat_hash_set<int> non_donatable_input_indices;

  absl::StatusOr<ExecuteOptionsProto> ToProto() const;
  static absl::StatusOr<ExecuteOptions> FromProto(
      const ExecuteOptionsProto& proto);
};

// Static memory usage for a compiled program.
// The on-device memory needed to run an executable is at least
//   generated_code_size_in_bytes
//   + argument_size_in_bytes + output_size_in_bytes - alias_size_in_bytes
//   + temp_size_in_bytes.
struct CompiledMemoryStats {
  // Device default memory (e.g., HBM for GPU/TPU) usage stats.
  int64_t generated_code_size_in_bytes = 0;
  int64_t argument_size_in_bytes = 0;
  int64_t output_size_in_bytes = 0;
  // How much argument is reused for output.
  int64_t alias_size_in_bytes = 0;
  int64_t temp_size_in_bytes = 0;

  // Host memory usage stats.
  int64_t host_generated_code_size_in_bytes = 0;
  int64_t host_argument_size_in_bytes = 0;
  int64_t host_output_size_in_bytes = 0;
  int64_t host_alias_size_in_bytes = 0;
  int64_t host_temp_size_in_bytes = 0;

  std::string serialized_hlo_proto = "";
  std::string DebugString() const;

  CompiledMemoryStatsProto ToProto() const;

  static CompiledMemoryStats FromProto(const CompiledMemoryStatsProto& proto);

  void PopulateBufferStatsFromAllocations(
      absl::Span<const BufferAllocation> allocs);
};

class PjRtExecutable {
 public:
  virtual ~PjRtExecutable() = default;

  virtual int num_replicas() const = 0;

  virtual int num_partitions() const = 0;

  virtual int64_t SizeOfGeneratedCodeInBytes() const = 0;

  // Unique name for this executable, e.g., HloModule name.
  virtual std::string_view name() const = 0;

  // Return an HloModule (optimized) per partition.
  virtual absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
  GetHloModules() const = 0;

  // Returns an output Shape per program, the size should be equal to
  // `GetHloModules()`.
  virtual absl::StatusOr<std::vector<Shape>> GetOutputShapes() const;

  // Returns a list of element types for each output, the size of the outer list
  // should be equal to `GetHloModules()`.
  virtual absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
  GetOutputElementTypes() const;

  // Returns a list of dimensions for each output, the size of the outer list
  // should be equal to `GetHloModules()`.
  virtual absl::StatusOr<std::vector<std::vector<DimensionVector>>>
  GetOutputDimensions() const;

  // Returns the layout of each input parameter.
  virtual absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetParameterLayouts() const;

  // Returns the layout of each output.
  virtual absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const;

  // Returns a list of lists of memory kind strings for output. The returned
  // value is `[num_programs, num_output]`. The size of the outer list should be
  // equal to `GetHloModules()`. Under SPMD, one can use
  // `GetOutputMemoryKinds().front()`.
  virtual absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const = 0;

  // Returns a list of parameter OpSharding protos.
  virtual std::optional<std::vector<OpSharding>> GetParameterShardings() const;

  // Returns a list of output OpSharding protos.
  virtual std::optional<std::vector<OpSharding>> GetOutputShardings() const;

  // Return memory stats that allow callers to estimate device memory usage
  // when running this executable.
  virtual absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const {
    return absl::UnimplementedError(
        "Retrieving CompiledMemoryStats is not supported.");
  }

  // Returns named values for cost properties of this executable (such as
  // operations, size of input/outputs, and run time estimate). Properties may
  // differ for different platforms.
  virtual absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const = 0;

  // Serialize this executable into a string and return the value.
  virtual absl::StatusOr<std::string> SerializeExecutable() const {
    return absl::UnimplementedError("Serializing executable is not supported.");
  }

  // Return a fingerprint of this executable.
  virtual absl::StatusOr<std::string> FingerprintExecutable() const {
    return absl::UnimplementedError(
        "Fingerprinting executable is not supported.");
  }

  virtual absl::StatusOr<CompileOptions> GetCompileOptions() const {
    return absl::UnimplementedError("CompileOptions not available.");
  }
};

class PjRtExecutableUtil {
 public:
  static absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  RunHloCostAnalysis(const PjRtExecutable& executable,
                     HloCostAnalysis* hlo_cost_analysis);

  static absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  RunHloCostAnalysis(const std::vector<std::shared_ptr<HloModule>>& hlo_modules,
                     HloCostAnalysis* hlo_cost_analysis);
};

}  // namespace zkx

#endif  // ZKX_PJRT_PJRT_EXECUTABLE_H_
