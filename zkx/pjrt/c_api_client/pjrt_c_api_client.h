/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_PJRT_C_API_CLIENT_PJRT_C_API_CLIENT_H_
#define ZKX_PJRT_C_API_CLIENT_PJRT_C_API_CLIENT_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"

#include "zkx/hlo/builder/zkx_computation.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/pjrt/c/pjrt_c_api.h"
#include "zkx/pjrt/c/pjrt_c_api_helpers.h"
#include "zkx/pjrt/distributed/key_value_store_interface.h"
#include "zkx/pjrt/pjrt_client.h"
#include "zkx/pjrt/pjrt_common.h"
#include "zkx/pjrt/pjrt_compiler.h"
#include "zkx/pjrt/pjrt_device_description.h"
#include "zkx/pjrt/pjrt_executable.h"
#include "zkx/pjrt/pjrt_future.h"
#include "zkx/pjrt/pjrt_layout.h"
// clang-format off
// TODO(jeong0982): Uncomment this. Dependency: topology_description.pb
// #include "xla/pjrt/proto/topology_description.pb.h"
// clang-format on
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "zkx/service/computation_placer.h"
#include "zkx/service/hlo_cost_analysis.h"
#include "zkx/shape_util.h"
// clang-format off
// TODO(jeong0982): Uncomment this. Dependency: coordination_service.pb
// #include "xla/tsl/protobuf/coordination_service.pb.h"
// clang-format on
#include "zkx/util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

class PjRtCApiClient;

class PjRtCApiDeviceDescription : public PjRtDeviceDescription {
 public:
  PjRtCApiDeviceDescription(const PJRT_Api* c_api,
                            PJRT_DeviceDescription* device_description);

  int id() const override;

  int process_index() const override;

  std::string_view device_kind() const override;

  std::string_view DebugString() const override;

  std::string_view ToString() const override;

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override;

  absl::Span<const PjRtMemorySpaceDescription* const> memory_spaces()
      const override;

  absl::StatusOr<const PjRtMemorySpaceDescription*> default_memory_space()
      const override;

 private:
  const PJRT_Api* c_api_;
  PJRT_DeviceDescription* device_description_;  // not owned
  // Device specific attributes with corresponding values.
  absl::flat_hash_map<std::string, zkx::PjRtDeviceAttribute> attributes_;
  mutable std::vector<PjRtMemorySpaceDescription> memory_space_descriptions_;
  mutable std::vector<PjRtMemorySpaceDescription*>
      memory_space_description_pointers_;
  mutable absl::StatusOr<PjRtMemorySpaceDescription*>
      default_memory_space_description_;

  // Initializes device specific attributes.
  void InitAttributes();
  // Initialize device specific memory descriptions.
  void InitMemoryDescriptions() const;
};

class PjRtCApiMemorySpace : public PjRtMemorySpace {
 public:
  explicit PjRtCApiMemorySpace(PJRT_Memory* c_memory, PjRtCApiClient* client)
      : client_(client), c_memory_(c_memory) {}

  PjRtClient* client() const override;

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  int id() const override;

  std::string_view kind() const override;
  int kind_id() const override;

  std::string_view DebugString() const override;

  std::string_view ToString() const override;

  const PJRT_Api* pjrt_c_api() const;

  PJRT_Memory* c_memory() const { return c_memory_; }

 private:
  friend class PjRtCApiClient;

  PjRtCApiClient* client_;  // not owned
  PJRT_Memory* c_memory_;   // not owned
  std::vector<PjRtDevice*> devices_;
};

class PjRtCApiDevice : public PjRtDevice {
 public:
  explicit PjRtCApiDevice(PJRT_Device* device, PjRtCApiClient* client);

  PjRtClient* client() const override;

  bool IsAddressable() const override;

  PjRtLocalHardwareId local_hardware_id() const override;

  absl::Status TransferToInfeed(const LiteralSlice& literal) override {
    return absl::UnimplementedError(
        "PJRT C API does not support TransferToInfeed. Please report an issue "
        "at https://github.com/google/jax/issues if you need this feature.");
  }

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    return absl::UnimplementedError(
        "PJRT C API does not support TransferFromOutfeed. Please report an "
        "issue at https://github.com/google/jax/issues if you need this "
        "feature.");
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
  }

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      std::string_view kind) const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      std::string_view description) const override {
    return nullptr;
  }

  PJRT_Device* c_device() const { return device_; }

  const PjRtCApiDeviceDescription& description() const override {
    return description_;
  }

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;

  absl::StatusOr<std::intptr_t> GetStreamForExternalReadyEvents()
      const override;

 private:
  friend class PjRtCApiClient;

  PjRtCApiClient* client_ = nullptr;
  PJRT_Device* device_;  // not owned
  PjRtCApiDeviceDescription description_;
  std::vector<PjRtMemorySpace*> memory_spaces_;
};

class PjRtCApiCompiler : public PjRtCompiler {
 public:
  explicit PjRtCApiCompiler(const PJRT_Api* c_api) : c_api_(c_api) {}

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const ZkxComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  // clang-format off
  // TODO(jeong0982): Uncomment this. Dependency: PjRtTopologyDescription
  // absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>>
  // DeserializePjRtTopologyDescription(
  //     const std::string& serialized_topology) override;
  // clang-format on

 private:
  const PJRT_Api* c_api_;
};

class PjRtCApiTopologyDescription : public PjRtTopologyDescription {
 public:
  // `owned` indicates whether this PjRtCApiTopologyDescription should take
  // ownership of `c_topology`, i.e., if owned is true,
  // PJRT_TopologyDescription_Destroy will be called on `c_topology` when this
  // PjRtCApiTopologyDescription is destroyed.
  PjRtCApiTopologyDescription(const PJRT_Api* c_api,
                              PJRT_TopologyDescription* c_topology, bool owned);

  PjRtPlatformId platform_id() const override { return platform_id_; }

  std::string_view platform_name() const override { return platform_name_; }

  std::string_view platform_version() const override;

  std::optional<PjRtCompiler*> compiler() const override {
    return compiler_.get();
  }

  PJRT_TopologyDescription* c_topology() const { return c_topology_; }

  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override;

  absl::StatusOr<std::string> Serialize() const override;

  // Returns vendor specific attributes about the topology.
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override;

 private:
  std::unique_ptr<PjRtCApiCompiler> compiler_;
  const PJRT_Api* c_api_;
  // nullptr iff the PJRT_TopologyDescription isn't owned by this wrapper
  // (i.e. by the caller).
  std::unique_ptr<PJRT_TopologyDescription,
                  ::pjrt::PJRT_TopologyDescriptionDeleter>
      owned_c_topology_;
  PJRT_TopologyDescription* c_topology_;  // not owned
  // Device specific attributes with corresponding values.
  absl::flat_hash_map<std::string, zkx::PjRtDeviceAttribute> attributes_;

  const std::string platform_version_;
  const std::string platform_name_;
  const PjRtPlatformId platform_id_;

  // Initializes device specific attributes.
  void InitAttributes();
};

class PjRtCApiClient : public PjRtClient {
 public:
  PjRtCApiClient(
      const PJRT_Api* c_api, PJRT_Client* c_client,
      std::unique_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data);

  int process_index() const override;

  int device_count() const override;
  int addressable_device_count() const override;

  absl::Span<PjRtDevice* const> devices() const override;
  absl::Span<PjRtDevice* const> addressable_devices() const override;

  absl::StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_device_id) const override;

  // clang-format off
  // TODO(jeong0982): Uncomment this. Dependency: CoordinatedTaskStateInfo
  // void UpdateGlobalProcessInfo(
  //     absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) override;
  // clang-format on

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  PjRtPlatformId platform_id() const override { return platform_id_; }

  std::string_view platform_name() const override { return platform_name_; };

  std::string_view platform_version() const override;

  std::optional<PjRtPluginAttributes> plugin_attributes() const override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return absl::UnimplementedError(
        "PJRT C API does not support GetHloCostAnalysis. Please report an "
        "issue at https://github.com/google/jax/issues if you need this "
        "feature.");
  }

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  // ZKX: Changed - CompileAndLoad -> Compile (ZKX base class name)
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const ZkxComputation& computation, CompileOptions options) override;

  // ZKX: Changed - CompileAndLoad -> Compile (ZKX base class name)
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  // `PjRtCApiClient::LoadSerializedExecutable()` ignores `LoadOptions` arg
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
  LoadSerializedExecutable(std::string_view serialized,
                           std::optional<CompileOptions> options,
                           const LoadOptions& load_options) override;

  // clang-format off
  // TODO(jeong0982): Uncomment this. Dependency: PJRT_Buffer_CreateUninitialized, PJRT_Buffer_CreateAlias,
  // PJRT_Buffer_CreateError
  // CreateUninitializedBuffer, CreateAliasBuffer, CreateErrorBuffer
  // clang-format on

  absl::StatusOr<const PjRtTopologyDescription*> GetTopologyDescription()
      const override;

  // clang-format off
  // TODO(jeong0982): Uncomment this. Dependency: HostAllocator extension
  // absl::StatusOr<HostAllocator*> GetHostAllocator() const override;
  // clang-format on

  absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const ShapeSpec> shape_specs,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  absl::StatusOr<std::uintptr_t> UnsafeBufferPointer(
      PjRtBuffer* buffer) override;

  // clang-format off
  // TODO(jeong0982): Uncomment this. Dependency: CrossHostTransfers extension
  // MakeCrossHostReceiveBuffers, CrossHostSendBuffers, CrossHostReceiveBuffers
  // clang-format on

  absl::Status DmaMap(void* data, size_t size) override;

  absl::Status DmaUnmap(void* data) override;

  const PJRT_Api* pjrt_c_api() const;

  PJRT_Client* pjrt_c_client() { return c_client_.get(); }

  PjRtCApiDevice* GetCppDevice(PJRT_Device* c_device) const {
    auto it = c_to_cpp_device_map_.find(c_device);
    CHECK(it != c_to_cpp_device_map_.end());
    return it->second;
  }

  PjRtCApiMemorySpace* GetCppMemory(PJRT_Memory* c_memory) const {
    auto it = c_to_cpp_memory_map_.find(c_memory);
    CHECK(it != c_to_cpp_memory_map_.end());
    return it->second;
  }

  PjRtHostMemoryForDeviceManager* GetPjRtHostMemoryForDeviceManager()
      const override {
    return nullptr;
  }

  using CrossHostRecvNotifierFunction =
      std::function<void(PJRT_Error*, const char**, size_t*, size_t)>;

  template <typename ExtType>
  ExtType* FindExtension(PJRT_Extension_Type type) const {
    return reinterpret_cast<
        ExtType*>(  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        FindExtensionImpl(type));
  }

 private:
  void InitDevicesAndMemorySpaces();
  void InitAttributes();
  PJRT_Extension_Base* FindExtensionImpl(PJRT_Extension_Type type) const;
  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> CreatePjRtError(
      const absl::Status& error) const;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBufferInternalImpl(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      std::variant<PjRtDevice*, PjRtMemorySpace*> device_or_memory,
      const Layout* device_layout);

  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter> c_client_;
  std::unique_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data_;
  std::vector<std::unique_ptr<PjRtCApiDevice>> owned_devices_;
  std::vector<PjRtDevice*> devices_;
  std::vector<PjRtDevice*> addressable_devices_;
  absl::flat_hash_map<PJRT_Device*, PjRtCApiDevice*> c_to_cpp_device_map_;
  std::vector<std::unique_ptr<PjRtCApiMemorySpace>> owned_memory_spaces_;
  std::vector<PjRtMemorySpace*> addressable_memory_spaces_;
  absl::flat_hash_map<PJRT_Memory*, PjRtCApiMemorySpace*> c_to_cpp_memory_map_;
  // There may be an error fetching the topology desc via the C API
  // (e.g. unimplemented). Save the error during client init so we can return it
  // from GetTopologyDescription().
  absl::StatusOr<const PjRtCApiTopologyDescription> topo_desc_;
  absl::flat_hash_map<PJRT_Extension_Type, PJRT_Extension_Base*> extensions_;
  // clang-format off
  // TODO(jeong0982): Uncomment this. Dependency: HostAllocator extension
  // absl::StatusOr<std::unique_ptr<HostAllocatorInterface>> host_allocator_;
  // clang-format on

  const std::string platform_version_;
  const std::string platform_name_;
  const PjRtPlatformId platform_id_;
  absl::flat_hash_map<std::string, zkx::PjRtValueType> attributes_;
};

class PjRtCApiBuffer : public PjRtBuffer {
 public:
  PjRtCApiBuffer(PjRtCApiClient* client, PJRT_Buffer* buffer);

  PrimitiveType element_type() const override;

  absl::Span<const int64_t> dimensions() const override;

  std::shared_ptr<const PjRtLayout> layout() const override;

  // PJRT C API doesn't support tuple buffers.
  bool IsTuple() const override { return false; }

  const Shape& on_device_shape() const override;

  bool has_dynamic_dimensions() const override;

  absl::Span<const bool> is_dynamic_dimension() const override;

  absl::StatusOr<std::vector<int64_t>> logical_dimensions() override;

  absl::StatusOr<Shape> logical_on_device_shape() override;

  PjRtMemorySpace* memory_space() const override;

  PjRtDevice* device() const override;

  PjRtClient* client() const override { return client_; }

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  PjRtFuture<> ToLiteral(MutableLiteralBase* literal) override;
  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override;

  void Delete() override;

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override {
    return absl::UnimplementedError(
        "PJRT C API does not support ReleaseDeviceMemoryOwnership");
  }

  bool IsDeleted() override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override;

  // ZKX: Added - pure virtual in ZKX PjRtBuffer base class
  void CopyToRemoteDeviceScattered(
      PjRtFuture<std::vector<std::string>> serialized_descriptors,
      std::vector<RemoteSendCallback> callbacks,
      const ScatterDetails& scatter_details) override;

  PjRtFuture<> GetReadyFuture() override;

  bool IsOnCpu() const override;

  PJRT_Buffer* c_buffer() const { return buffer_.get(); }

  const PJRT_Api* pjrt_c_api() const { return client_->pjrt_c_api(); }

 private:
  // Gets the raw pointer to `readiness_event_`. If `readiness_event_` has not
  // yet been initialized, this function does so before returning the pointer.
  PJRT_Event* GetReadyEvent();

  // `MakePromiseTrackEvent` sets `readiness_promise_` up to track
  // `readiness_event_`. This is used to implement `GetReadyFuture()`.
  // `readiness_promise_` should be created before calling this function.
  void MakePromiseTrackEvent();

  PjRtCApiClient* client_;  // not owned
  std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer_;
  std::unique_ptr<PJRT_Event, ::pjrt::PJRT_EventDeleter> readiness_event_;
  // This is a shared_ptr to keep the underlying future alive even if
  // `readiness_promise` is destroyed before `readiness_event`, and the callback
  // we set on `readiness_event` modifies `readiness_promise_`.
  std::shared_ptr<PjRtFuture<>::Promise> readiness_promise_;
  // Future tied to the `readiness_promise_`.
  PjRtFuture<> readiness_future_;
  // Set and cached the first time layout() is called.
  mutable std::shared_ptr<const PjRtLayout> layout_;
  // Set and cached the first time is_dynamic_dimension() is called.
  mutable std::optional<absl::InlinedVector<bool, InlineRank()>>
      is_dynamic_dimension_;
  // Used to synchronize concurrent setting of cached values.
  mutable absl::Mutex mu_;
  // Cached result of on_device_shape();
  mutable std::optional<Shape> on_device_shape_;
};

class PjRtCApiExternalReference : public PjRtBuffer::ExternalReference {
 public:
  PjRtCApiExternalReference(PjRtCApiClient* client, PjRtCApiBuffer* buffer,
                            void* data_ptr)
      : client_(client), buffer_(buffer) {
    data_ptr_ = data_ptr;
  }
  ~PjRtCApiExternalReference() override;

  absl::Status WaitUntilBufferReadyOnStream(std::intptr_t stream) override;

 private:
  PjRtCApiClient* client_;  // not owned
  PjRtCApiBuffer* buffer_;  // not owned
};

class PjRtCApiExecutable : public PjRtExecutable {
 public:
  PjRtCApiExecutable(const PJRT_Api* c_api, PJRT_Executable* executable);

  std::string_view name() const override;
  int num_replicas() const override;
  int num_partitions() const override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const override;

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    return pjrt::GetCompiledMemoryStats(c_api_, executable_.get());
  }

  absl::StatusOr<std::vector<Shape>> GetOutputShapes() const override;

  absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
  GetOutputElementTypes() const override;

  absl::StatusOr<std::vector<std::vector<DimensionVector>>>
  GetOutputDimensions() const override;

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const override;

  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const override;

  const PJRT_Api* pjrt_c_api() const { return c_api_; }
  PJRT_Executable* c_executable() const { return executable_.get(); }

  absl::StatusOr<std::string> SerializeExecutable() const override;

  absl::StatusOr<std::string> FingerprintExecutable() const override;

  // TODO(b/438000615): Move this to PjRtLoadedExecutable.
  absl::StatusOr<std::string> GetSerializedExecutableMetadata() const;

 private:
  const PJRT_Api* c_api_;
  std::unique_ptr<PJRT_Executable, ::pjrt::PJRT_ExecutableDeleter> executable_;
};

class PjRtCApiLoadedExecutable : public PjRtLoadedExecutable {
 public:
  PjRtCApiLoadedExecutable(PjRtCApiClient* client,
                           PJRT_LoadedExecutable* executable);

  PjRtClient* client() const override { return client_; }
  std::string_view name() const override { return executable_->name(); }
  int num_replicas() const override { return executable_->num_replicas(); }
  int num_partitions() const override { return executable_->num_partitions(); }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return executable_->SizeOfGeneratedCodeInBytes();
  }

  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const override {
    return executable_->GetCostAnalysis();
  }

  const DeviceAssignment& device_assignment() const override {
    CHECK(device_assignment_ != nullptr)
        << "device_assignment_ is a nullptr. This is likely because "
           "PjRtCApiLoadedExecutable::device_assignment() was called on a "
           "portable executable, which does not have a device assignment.";
    return *device_assignment_;
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    CHECK(false)
        << "PJRT C API does not support addressable_device_logical_ids";
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return executable_->GetHloModules();
  }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    return executable_->GetCompiledMemoryStats();
  }

  absl::StatusOr<std::vector<Shape>> GetOutputShapes() const override {
    return executable_->GetOutputShapes();
  }

  absl::StatusOr<std::vector<std::vector<PrimitiveType>>>
  GetOutputElementTypes() const override {
    return executable_->GetOutputElementTypes();
  }

  absl::StatusOr<std::vector<std::vector<DimensionVector>>>
  GetOutputDimensions() const override {
    return executable_->GetOutputDimensions();
  }

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const override {
    return executable_->GetOutputLayouts();
  }

  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const override {
    return executable_->GetOutputMemoryKinds();
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  void Delete() override;
  bool IsDeleted() override;

  absl::StatusOr<std::string> SerializeExecutable() const override {
    return executable_->SerializeExecutable();
  }

  const PJRT_Api* pjrt_c_api() const { return client_->pjrt_c_api(); }
  PJRT_Executable* c_executable() const { return executable_->c_executable(); }

  PJRT_LoadedExecutable* c_loaded_executable() const {
    return loaded_executable_.get();
  }

  // std::function version of PJRT_SendCallback
  using SendCallbackFunction = std::function<PJRT_Error*(
      PJRT_Chunk*, PJRT_CallbackError*, size_t, bool)>;
  // std::function version of PJRT_RecvCallback
  using RecvCallbackFunction = std::function<void(PJRT_CopyToDeviceStream*)>;

  // Override to call FingerprintExecutable through the wrapped
  // PjRtCApiExecutable.
  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return executable_->FingerprintExecutable();
  }

 private:
  // Groups data needed to support send/recv execution callbacks.
  struct SendRecvCallbackData {
    std::vector<std::vector<PJRT_SendCallbackInfo>> c_send_callbacks;
    std::vector<PJRT_SendCallbackInfo*> c_send_callback_lists;
    std::vector<std::vector<PJRT_RecvCallbackInfo>> c_recv_callbacks;
    std::vector<PJRT_RecvCallbackInfo*> c_recv_callback_lists;
    std::vector<SendCallbackFunction> send_callback_functions;
    std::vector<RecvCallbackFunction> recv_callback_functions;
  };

  // Returns the number of outputs of the executable.
  absl::StatusOr<size_t> GetNumOutputs() const;

  // Allocates memory for the `Execute` output.
  // These functions are a little verbose, but allocating the correct amount of
  // memory on initialization (thus avoiding `resize` calls) provides a
  // significant performance optimization.
  absl::StatusOr<std::vector<std::vector<PJRT_Buffer*>>>
  InitializeOutputListsStorage(size_t outer_size) const;
  absl::StatusOr<std::vector<PJRT_Buffer**>> InitializeOutputLists(
      std::vector<std::vector<PJRT_Buffer*>>& c_output_lists_storage) const;

  // Gets common Execute_Args for use in various Execute* functions.
  // device_complete_events in the return is set if the input
  // device_complete_events has value.
  absl::StatusOr<PJRT_LoadedExecutable_Execute_Args> GetCommonExecuteArgs(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options, PJRT_ExecuteOptions& c_options,
      std::vector<std::vector<PJRT_Buffer*>>& c_argument_lists_storage,
      std::vector<PJRT_Buffer**>& c_arguments,
      std::optional<std::vector<PJRT_Event*>>& device_complete_events,
      SendRecvCallbackData& send_recv_callback_data,
      std::vector<int64_t>& non_donatable_input_indices_storage,
      std::vector<int>& task_ids_storage,
      std::vector<int64_t>& incarnation_ids_storage) const;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  ExecuteWithSingleDevice(absl::Span<PjRtBuffer* const> argument_handles,
                          PjRtDevice* device, const ExecuteOptions& options,
                          std::optional<PjRtFuture<>>& returned_future,
                          bool fill_future) const;

  PjRtCApiClient* client_;  // not owned
  std::unique_ptr<PJRT_LoadedExecutable, ::pjrt::PJRT_LoadedExecutableDeleter>
      loaded_executable_;
  std::unique_ptr<PjRtCApiExecutable> executable_;
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<const DeviceAssignment> device_assignment_;

  void InitDevices();
  void InitDeviceAssignment();
};

class CApiCopyToDeviceStream : public CopyToDeviceStream {
 public:
  CApiCopyToDeviceStream(PJRT_CopyToDeviceStream* c_stream,
                         const PJRT_Api* c_api);
  ~CApiCopyToDeviceStream() override;

  PjRtFuture<> AddChunk(PjRtChunk chunk) override;

 private:
  PJRT_CopyToDeviceStream* c_stream_;  // not owned
  const PJRT_Api* c_api_;              // not owned
};

absl::StatusOr<std::unique_ptr<PjRtClient>> GetCApiClient(
    std::string_view device_type,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {},
    std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr);

absl::StatusOr<std::unique_ptr<PjRtClient>> WrapClientAroundCApi(
    const PJRT_Api* c_api,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {},
    std::shared_ptr<KeyValueStoreInterface> kv_store = nullptr);

absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetCApiTopology(
    const PJRT_Api* c_api, std::string_view topology_name,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options);

// A variant that takes `device_type` as an input, used for plugins that are not
// registered with standard way (xla_bridge.register_plugin).
// TODO(b/322357665): Delete this method after TPU plugin changes to use the
// standard registration.
absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetCApiTopology(
    std::string_view device_type, std::string_view topology_name,
    const absl::flat_hash_map<std::string, PjRtValueType>& create_options = {});

absl::StatusOr<std::unique_ptr<PjRtCompiler>> GetCApiCompiler(
    std::string_view device_type);

absl::StatusOr<std::unique_ptr<PjRtCompiler>> GetCApiCompiler();

// clang-format off
// TODO(jeong0982): Uncomment this. Dependency: PjRtPhaseCompiler
// absl::StatusOr<std::unique_ptr<PjRtPhaseCompiler>> GetCApiPhaseCompiler(
//     std::string_view device_type);
// absl::StatusOr<std::unique_ptr<PjRtPhaseCompiler>> GetCApiPhaseCompiler();
// clang-format on

}  // namespace zkx

#endif  // ZKX_PJRT_C_API_CLIENT_PJRT_C_API_CLIENT_H_
