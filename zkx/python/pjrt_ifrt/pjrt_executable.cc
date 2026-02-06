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

#include "zkx/python/pjrt_ifrt/pjrt_executable.h"

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "zkx/pjrt/host_callback.h"
#include "zkx/pjrt/layout_mode.h"
#include "zkx/pjrt/pjrt_compiler.h"
#include "zkx/pjrt/pjrt_future.h"
#include "zkx/pjrt/utils.h"
#include "zkx/primitive_util.h"
#include "zkx/python/ifrt/basic_device_list.h"
#include "zkx/python/ifrt/memory.h"
#include "zkx/python/pjrt_ifrt/pjrt_array.h"
#include "zkx/python/pjrt_ifrt/pjrt_device.h"
#include "zkx/python/pjrt_ifrt/pjrt_dtype.h"
#include "zkx/python/pjrt_ifrt/pjrt_memory.h"
#include "zkx/python/pjrt_ifrt/zkx_sharding.h"
#include "zkx/service/hlo.pb.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/status_macros.h"

namespace zkx::ifrt {
namespace {

// Returns the output element types of the first module in a
// `PjRtLoadedExecutable`.
absl::StatusOr<std::vector<zkx::PrimitiveType>>
GetFirstModuleOutputElementTypes(
    zkx::PjRtLoadedExecutable* pjrt_loaded_executable) {
  auto element_types = pjrt_loaded_executable->GetOutputElementTypes();
  TF_RETURN_IF_ERROR(element_types.status());
  if (element_types->empty()) {
    return absl::InvalidArgumentError("No output element types found");
  }
  return element_types->front();
}

// Returns the output dimensions of the first module in a
// `PjRtLoadedExecutable`.
absl::StatusOr<std::vector<zkx::DimensionVector>>
GetFirstModuleOutputDimensions(
    zkx::PjRtLoadedExecutable* pjrt_loaded_executable) {
  auto dimensions = pjrt_loaded_executable->GetOutputDimensions();
  TF_RETURN_IF_ERROR(dimensions.status());
  if (dimensions->empty()) {
    return absl::InvalidArgumentError("No output dimensions found");
  }
  return dimensions->front();
}

// Returns the output shardings of the first module in a
// `PjRtLoadedExecutable`.
absl::StatusOr<std::optional<zkx::HloSharding>> GetFirstModuleOutputSharding(
    zkx::PjRtLoadedExecutable* pjrt_loaded_executable,
    const zkx::Shape& shape) {
  auto output_shardings = pjrt_loaded_executable->GetOutputShardings();
  std::optional<zkx::HloSharding> result_hlo_sharding;
  if (output_shardings.has_value()) {
    std::vector<zkx::HloSharding> hlo_shardings;
    hlo_shardings.reserve(output_shardings->size());
    for (const auto& sharding : *output_shardings) {
      TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                          zkx::HloSharding::FromProto(sharding));
      hlo_shardings.push_back(hlo_sharding);
    }
    if (shape.IsTuple()) {
      return zkx::HloSharding::Tuple(shape, hlo_shardings);
    } else {
      return hlo_shardings.front();
    }
  }
  return std::nullopt;
}

// Returns the flattened output memory_kinds of the first module in a
// `PjRtLoadedExecutable`.
// `UnimplementedError` will be converted into `std::nullopt`.
absl::StatusOr<std::optional<std::vector<std::string_view>>>
GetFirstModuleOutputMemoryKinds(
    zkx::PjRtLoadedExecutable* pjrt_loaded_executable) {
  auto output_memory_kinds = pjrt_loaded_executable->GetOutputMemoryKinds();
  // Gracefully handle an unimplemented error.
  if (absl::IsUnimplemented(output_memory_kinds.status())) {
    return std::nullopt;
  }
  TF_RETURN_IF_ERROR(output_memory_kinds.status());
  // Expect `zkx::PjRtExecutable::GetOutputMemoryKinds()` to return at least
  // one module's output memory_kinds if it returns any non-error result.
  if (output_memory_kinds->empty()) {
    return absl::FailedPreconditionError("No output memory kinds found");
  }
  return std::move(output_memory_kinds)->front();
}

// Returns the flattened output layouts of the first module in a
// `PjRtLoadedExecutable`.
// `UnimplementedError` will be converted into a vector of `nullptr`.
absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
GetFirstModuleOutputLayouts(
    zkx::PjRtLoadedExecutable* pjrt_loaded_executable,
    absl::Span<const zkx::LayoutMode> output_layout_modes) {
  absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
      executable_output_layouts = pjrt_loaded_executable->GetOutputLayouts();
  // An unimplemented error is converted into all-default layouts.
  if (absl::IsUnimplemented(executable_output_layouts.status())) {
    return std::vector<std::shared_ptr<const zkx::PjRtLayout>>(
        /*size=*/output_layout_modes.size(), /*value=*/nullptr);
  }
  TF_RETURN_IF_ERROR(executable_output_layouts.status());
  std::vector<std::shared_ptr<const zkx::PjRtLayout>> output_layouts;
  if (executable_output_layouts->size() != output_layout_modes.size()) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Output memory kinds and output layout modes have different sizes: %d "
        "vs. %d",
        executable_output_layouts->size(), output_layout_modes.size()));
  }
  output_layouts.reserve(executable_output_layouts->size());
  for (int i = 0; i < executable_output_layouts->size(); ++i) {
    if (output_layout_modes[i].mode == zkx::LayoutMode::Mode::kDefault) {
      output_layouts.push_back(nullptr);
    } else {
      output_layouts.push_back(std::move((*executable_output_layouts)[i]));
    }
  }
  return output_layouts;
}

struct ShapePartialInfo {
  std::vector<zkx::PrimitiveType> element_types;
  std::vector<zkx::DimensionVector> dimensions;
};

absl::StatusOr<ShapePartialInfo> CreateShapePartialInfo(
    absl::Span<const zkx::Shape> shapes) {
  ShapePartialInfo partial_info;
  partial_info.element_types.reserve(shapes.size());
  partial_info.dimensions.reserve(shapes.size());
  for (const auto& shape : shapes) {
    if (shape.IsTuple()) {
      return absl::FailedPreconditionError(
          "Tupled shape is not supported in `CreateShapePartialInfo`.");
    }
    partial_info.element_types.push_back(shape.element_type());
    partial_info.dimensions.push_back(
        zkx::ShapeUtil::CreateDimensionVectorFromShape(shape));
  }

  return partial_info;
}

// Special `zkx::GetLayoutModes()` implementation for obtaining layout modes
// from `hlo_module` without serializing it into proto.

const char* kDelimiter = ";";

absl::StatusOr<std::vector<LayoutMode>> GetLayoutModesFromFrontendAttr(
    std::string_view attr) {
  // SkipEmpty() needed to avoid returning the empty string when attr is empty.
  std::vector<std::string> str_modes =
      absl::StrSplit(attr, kDelimiter, absl::SkipEmpty());
  std::vector<LayoutMode> result;
  for (const std::string& str_mode : str_modes) {
    TF_ASSIGN_OR_RETURN(LayoutMode mode, LayoutMode::FromString(str_mode));
    result.emplace_back(std::move(mode));
  }
  return result;
}

absl::StatusOr<std::vector<LayoutMode>> GetLayoutModes(
    const HloModule& hlo_module, std::string_view frontend_attr_name,
    size_t num_values) {
  const auto& frontend_attrs = hlo_module.frontend_attributes().map();
  auto iter = frontend_attrs.find(frontend_attr_name);
  if (iter == frontend_attrs.end()) {
    // Return all default layouts if frontend attr isn't present.
    return std::vector<LayoutMode>(num_values);
  }
  return GetLayoutModesFromFrontendAttr(iter->second);
}

}  // namespace

char PjRtCompatibleExecutable::ID = 0;
char PjRtCompatibleLoadedExecutable::ID = 0;
char PjRtExecutable::ID = 0;
char PjRtLoadedExecutable::ID = 0;

absl::StatusOr<ExecutableRef> PjRtExecutable::Create(
    std::shared_ptr<zkx::PjRtExecutable> pjrt_executable) {
  return ExecutableRef(new PjRtExecutable(std::move(pjrt_executable)));
}

absl::StatusOr<std::optional<std::string>> PjRtExecutable::Fingerprint() const {
  DCHECK(this);
  return pjrt_executable_->FingerprintExecutable();
}

absl::StatusOr<std::string> PjRtExecutable::Serialize() const {
  DCHECK(this);
  return pjrt_executable_->SerializeExecutable();
}

absl::StatusOr<LoadedExecutableRef> PjRtLoadedExecutable::Create(
    PjRtClient* client,
    std::shared_ptr<zkx::PjRtLoadedExecutable> pjrt_loaded_executable,
    std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
    DeviceListRef executable_devices) {
  // TODO(hyeontaek): Use a full shape and a sharding rather than a per-shard
  // shape.
  VLOG(3) << "PjRtLoadedExecutable::Create";
  VLOG(3) << "Using per-shard shape";
  TF_ASSIGN_OR_RETURN(
      auto result_element_types,
      GetFirstModuleOutputElementTypes(pjrt_loaded_executable.get()));
  TF_ASSIGN_OR_RETURN(
      auto result_dimensions,
      GetFirstModuleOutputDimensions(pjrt_loaded_executable.get()));
  TF_ASSIGN_OR_RETURN(
      auto result_memory_kinds,
      GetFirstModuleOutputMemoryKinds(pjrt_loaded_executable.get()));
  // Obtaining output layout modes and output layouts directly from
  // `PjRtLoadedExecutable` may fail because the currently PjRt implementations
  // often fetch and serialize the optimized HLO. For now, we gracefully
  // handle it by omitting output layouts at creation time and using output
  // `PjRtBuffer`'s concrete layouts.
  // TODO(hyeontaek): Add a way to obtain output layout modes and
  // `PjRtLoadedExecutable::GetOutputLayouts()` without causing the optimized
  // HLO to be serialized and fetched.
  std::optional<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
      output_layouts;
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> hlo_modules =
      pjrt_loaded_executable->GetHloModules();
  if (hlo_modules.ok()) {
    if (hlo_modules->empty()) {
      return absl::FailedPreconditionError("Requires at least one HloModule.");
    }
    absl::StatusOr<std::vector<zkx::LayoutMode>> output_layout_modes =
        GetLayoutModes(*hlo_modules->front(), "out_layout_modes",
                       result_element_types.size());
    if (output_layout_modes.ok()) {
      absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
          first_module_output_layouts = GetFirstModuleOutputLayouts(
              pjrt_loaded_executable.get(), *output_layout_modes);
      if (first_module_output_layouts.ok()) {
        output_layouts = *std::move(first_module_output_layouts);
      }
    }
  }
  return CreateInternal(client, std::move(pjrt_loaded_executable),
                        result_element_types, result_dimensions,
                        /*result_hlo_sharding=*/std::nullopt,
                        result_memory_kinds, output_layouts,
                        loaded_host_callbacks, std::move(executable_devices));
}

namespace {

absl::StatusOr<std::vector<zkx::Shape>> ResultShapesOfModule(
    mlir::ModuleOp module) {
  auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    return absl::InvalidArgumentError("MLIR module has no main function");
  }
  auto type = main.getFunctionType();
  std::vector<zkx::Shape> result_shapes;
  result_shapes.reserve(type.getNumResults());
  for (unsigned i = 0; i < type.getNumResults(); ++i) {
    auto result_type = type.getResult(i);
    result_shapes.push_back(zkx::TypeToShape(result_type));
  }
  return result_shapes;
}

}  // namespace

absl::StatusOr<LoadedExecutableRef> PjRtLoadedExecutable::Create(
    PjRtClient* client, mlir::ModuleOp module,
    zkx::CompileOptions compile_options,
    std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
    DeviceListRef executable_devices) {
  VLOG(3) << "PjRtLoadedExecutable::Create";
  if (VLOG_IS_ON(3)) {
    module.dump();
  }
  VLOG(3) << compile_options.ToProto()->DebugString();
  const auto& build_options = compile_options.executable_build_options;
  const bool auto_spmd_partitioning =
      build_options.use_spmd_partitioning() &&
      build_options.num_partitions() > 1 &&
      (build_options.use_auto_spmd_partitioning() ||
       build_options.any_allow_spmd_sharding_propagation_to_parameters() ||
       build_options.any_allow_spmd_sharding_propagation_to_output());

  // We have to do process the MLIR before the compile call, since the latter
  // will use the MLIR as scratch space, or possibly even deallocate it.
  TF_ASSIGN_OR_RETURN(const std::vector<zkx::Shape> result_shapes,
                      ResultShapesOfModule(module));
  absl::StatusOr<std::vector<zkx::LayoutMode>> output_layout_modes =
      GetOutputLayoutModes(module);

  // TODO(chokboole): Should rename Compile to CompileAndLoad
  TF_ASSIGN_OR_RETURN(auto pjrt_loaded_executable,
                      client->pjrt_client()->Compile(
                          std::move(module), std::move(compile_options)));

  if (auto_spmd_partitioning) {
    // TODO(hyeontaek): Use a full shape and a sharding rather than a per-shard
    // shape.
    VLOG(3) << "Using per-shard shape";
    TF_ASSIGN_OR_RETURN(
        auto result_element_types,
        GetFirstModuleOutputElementTypes(pjrt_loaded_executable.get()));
    TF_ASSIGN_OR_RETURN(
        auto result_dimensions,
        GetFirstModuleOutputDimensions(pjrt_loaded_executable.get()));
    TF_ASSIGN_OR_RETURN(
        auto result_memory_kinds,
        GetFirstModuleOutputMemoryKinds(pjrt_loaded_executable.get()));
    // Obtaining output layout modes and output layouts directly from
    // `PjRtLoadedExecutable` may fail because the currently PjRt
    // implementations often fetch and serialize the optimized HLO. For now, we
    // gracefully handle it by omitting output layouts at creation time and
    // using output `PjRtBuffer`'s concrete layouts.
    // TODO(hyeontaek): Add a way to obtain output layout modes and
    // `PjRtLoadedExecutable::GetOutputLayouts()` without causing the optimized
    // HLO to be serialized and fetched.
    std::optional<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
        output_layouts;
    if (output_layout_modes.ok()) {
      absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
          first_module_output_layouts = GetFirstModuleOutputLayouts(
              pjrt_loaded_executable.get(), *output_layout_modes);
      if (first_module_output_layouts.ok()) {
        output_layouts = *std::move(first_module_output_layouts);
      }
    }
    return CreateInternal(client, std::move(pjrt_loaded_executable),
                          result_element_types, result_dimensions,
                          /*result_hlo_sharding=*/std::nullopt,
                          result_memory_kinds, output_layouts,
                          std::move(loaded_host_callbacks),
                          std::move(executable_devices));
  } else {
    VLOG(3) << "Using full shape";
    // TODO(yueshengys): Consider getting element types and dimensions directly
    // from module.
    bool tuple_output = result_shapes.size() != 1;
    zkx::Shape result_shape;
    std::vector<zkx::Shape> output_shapes;
    if (tuple_output) {
      result_shape = zkx::ShapeUtil::MakeTupleShape(result_shapes);
      output_shapes = std::move(result_shapes);
    } else {
      result_shape = result_shapes.front();
      output_shapes = result_shape.IsTuple()
                          ? result_shape.tuple_shapes()
                          : std::vector<zkx::Shape>{result_shape};
    }
    TF_ASSIGN_OR_RETURN(auto shape_partial_info,
                        CreateShapePartialInfo(output_shapes));
    TF_ASSIGN_OR_RETURN(auto result_hlo_sharding,
                        GetFirstModuleOutputSharding(
                            pjrt_loaded_executable.get(), result_shape));
    TF_ASSIGN_OR_RETURN(
        auto result_memory_kinds,
        GetFirstModuleOutputMemoryKinds(pjrt_loaded_executable.get()));
    // Obtaining output layout modes and output layouts directly from
    // `PjRtLoadedExecutable` may fail because the currently PjRt
    // implementations often fetch and serialize the optimized HLO. For now, we
    // gracefully handle it by omitting output layouts at creation time and
    // using output `PjRtBuffer`'s concrete layouts.
    // TODO(hyeontaek): Add a way to obtain output layout modes and
    // `PjRtLoadedExecutable::GetOutputLayouts()` without causing the optimized
    // HLO to be serialized and fetched.
    std::optional<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
        output_layouts;
    if (output_layout_modes.ok()) {
      absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
          first_module_output_layouts = GetFirstModuleOutputLayouts(
              pjrt_loaded_executable.get(), *output_layout_modes);
      if (first_module_output_layouts.ok()) {
        output_layouts = *std::move(first_module_output_layouts);
      }
    }
    return CreateInternal(
        client, std::move(pjrt_loaded_executable),
        shape_partial_info.element_types, shape_partial_info.dimensions,
        result_hlo_sharding, result_memory_kinds, output_layouts,
        std::move(loaded_host_callbacks), std::move(executable_devices));
  }
}

absl::StatusOr<LoadedExecutableRef> PjRtLoadedExecutable::CreateInternal(
    PjRtClient* client,
    std::shared_ptr<zkx::PjRtLoadedExecutable> pjrt_loaded_executable,
    absl::Span<const zkx::PrimitiveType> result_element_types,
    absl::Span<const zkx::DimensionVector> result_dimensions,
    const std::optional<zkx::HloSharding>& result_hlo_sharding,
    const std::optional<std::vector<std::string_view>>& result_memory_kinds,
    const std::optional<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>&
        output_layouts,
    std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
    DeviceListRef executable_devices) {
  // For jit(pmap(...)), the device assignment (passed as `executable_devices`)
  // may contain a single device while the PjRt executable has multiple
  // addressable devices. We check for this condition and replace
  // `executable_devices` with the executable's addressable devices if
  // necessary.
  if (pjrt_loaded_executable->num_replicas() > 1 &&
      executable_devices->devices().size() == 1) {
    if (pjrt_loaded_executable->addressable_devices().size() > 1) {
      BasicDeviceList::Devices ds;
      ds.reserve(pjrt_loaded_executable->addressable_devices().size());
      for (zkx::PjRtDevice* device :
           pjrt_loaded_executable->addressable_devices()) {
        TF_ASSIGN_OR_RETURN(Device * ifrt_device,
                            client->LookupPjRtDevice(device));
        ds.push_back(ifrt_device);
      }
      executable_devices = BasicDeviceList::Create(std::move(ds));
    } else if (pjrt_loaded_executable->addressable_devices().size() == 1) {
      TF_ASSIGN_OR_RETURN(
          Device * ifrt_device,
          client->LookupPjRtDevice(
              pjrt_loaded_executable->addressable_devices().front()));
      if (ifrt_device != executable_devices->devices().front()) {
        return absl::FailedPreconditionError(
            "Addressable device does not match sharding device");
      }
    }
  }
  if (executable_devices->devices().size() <
      pjrt_loaded_executable->addressable_devices().size()) {
    return absl::FailedPreconditionError(
        "Sharding devices must be at least as many as addressable devices");
  }
  std::vector<DType> output_dtypes;
  std::vector<Shape> output_shapes;
  std::vector<ShardingRef> output_shardings;

  auto append_arg = [&](const zkx::PrimitiveType& element_type,
                        const zkx::DimensionVector& dimensions,
                        const zkx::HloSharding* sharding,
                        MemoryKind memory_kind) -> absl::Status {
    TF_ASSIGN_OR_RETURN(auto dtype, ToDType(element_type));
    output_dtypes.push_back(dtype);
    output_shapes.push_back(Shape(dimensions));

    CHECK(zkx::primitive_util::IsArrayType(element_type));

    if (sharding != nullptr) {
      output_shardings.push_back(ifrt::HloSharding::Create(
          executable_devices, memory_kind, *sharding));
    } else {
      // Assume a traditional replication computation where tile shapes are
      // the same as global shapes.
      const zkx::DimensionVector& tile_shape_dimensions = dimensions;
      output_shardings.push_back(ifrt::ConcreteEvenSharding::Create(
          executable_devices, memory_kind,
          /*shape=*/ifrt::Shape(dimensions),
          /*shard_shape=*/ifrt::Shape(tile_shape_dimensions)));
    }
    return absl::OkStatus();
  };
  auto append_token = [&](MemoryKind memory_kind) {
    output_dtypes.push_back(DType(DType::kToken));
    output_shapes.push_back(Shape({}));
    output_shardings.push_back(
        ifrt::ConcreteEvenSharding::Create(executable_devices, memory_kind,
                                           /*shape=*/ifrt::Shape({}),
                                           /*shard_shape=*/ifrt::Shape({})));
  };
  auto check_output_sharding_condition =
      [](absl::Span<const zkx::PrimitiveType> element_types,
         const zkx::HloSharding& sharding) {
        if (sharding.IsTuple()) {
          // Check that the HLO sharding of the result has the same number of
          // elements as the output tuple shape. If the output is an empty tuple
          // then the output sharding will have a single element for the tuple
          // as a special case, so we will have to allow that by checking this
          // condition specifically.
          return element_types.size() == sharding.tuple_elements().size() ||
                 (element_types.empty() &&
                  sharding.tuple_elements().size() == 1);
        }
        return element_types.size() == 1;
      };

  if (result_memory_kinds.has_value() &&
      result_memory_kinds->size() != result_element_types.size()) {
    return absl::FailedPreconditionError(
        "Output memory kinds are inconsistent with the output shape");
  }
  if (result_hlo_sharding.has_value() &&
      !check_output_sharding_condition(result_element_types,
                                       *result_hlo_sharding)) {
    return absl::FailedPreconditionError(
        "Output sharding is inconsistent with the output shape");
  }

  CHECK_EQ(result_element_types.size(), result_dimensions.size());
  output_dtypes.reserve(result_element_types.size());
  output_shapes.reserve(result_element_types.size());
  output_shardings.reserve(result_element_types.size());
  for (int i = 0; i < result_element_types.size(); ++i) {
    const auto& element_type = result_element_types[i];
    MemoryKind element_memory_kind;
    if (result_memory_kinds.has_value()) {
      element_memory_kind = MemoryKind((*result_memory_kinds)[i]);
    }
    if (zkx::primitive_util::IsArrayType(element_type)) {
      const zkx::HloSharding* element_hlo_sharding = nullptr;
      if (result_hlo_sharding.has_value()) {
        element_hlo_sharding = result_hlo_sharding->IsTuple()
                                   ? &result_hlo_sharding->tuple_elements()[i]
                                   : &*result_hlo_sharding;
        if (element_hlo_sharding->IsTuple()) {
          return absl::FailedPreconditionError(
              "Nested-tupled output sharding is not supported");
        }
      }
      TF_RETURN_IF_ERROR(append_arg(element_type, result_dimensions[i],
                                    element_hlo_sharding, element_memory_kind));
    } else if (element_type == TOKEN) {
      append_token(element_memory_kind);
    } else {
      return absl::FailedPreconditionError(
          "The element type is not a supported type (array, token)");
    }
  }

  std::vector<PjRtHostSendAndRecvLoadedHostCallback*>
      host_send_and_recv_callbacks;
  host_send_and_recv_callbacks.reserve(loaded_host_callbacks.size());
  // Gather all `PjRtLoadedHostCallback` separately, as each execution will
  // register `PjRtLoadedHostCallback` for host send and recv. All host
  // callbacks will be referenced by the executable and any pending execution to
  // guarantee the liveliness of host callbacks during executions.
  for (auto& loaded_host_callback : loaded_host_callbacks) {
    auto* host_send_and_recv_callback =
        llvm::dyn_cast<PjRtHostSendAndRecvLoadedHostCallback>(
            loaded_host_callback.get());
    if (host_send_and_recv_callback != nullptr) {
      host_send_and_recv_callbacks.push_back(host_send_and_recv_callback);
    }
  }

  std::vector<Device*> addressable_devices;
  addressable_devices.reserve(
      pjrt_loaded_executable->addressable_devices().size());
  for (zkx::PjRtDevice* device :
       pjrt_loaded_executable->addressable_devices()) {
    TF_ASSIGN_OR_RETURN(Device * ifrt_device, client->LookupPjRtDevice(device));
    addressable_devices.push_back(ifrt_device);
  }

  return LoadedExecutableRef(new PjRtLoadedExecutable(
      client, std::move(pjrt_loaded_executable), std::move(executable_devices),
      std::move(addressable_devices), std::move(loaded_host_callbacks),
      std::move(host_send_and_recv_callbacks), std::move(output_dtypes),
      std::move(output_shapes), std::move(output_shardings),
      std::move(output_layouts)));
}

PjRtLoadedExecutable::PjRtLoadedExecutable(
    PjRtClient* client,
    std::shared_ptr<zkx::PjRtLoadedExecutable> pjrt_loaded_executable,
    DeviceListRef devices, std::vector<Device*> addressable_devices,
    std::vector<tsl::RCReference<LoadedHostCallback>> all_loaded_host_callbacks,
    std::vector<PjRtHostSendAndRecvLoadedHostCallback*>
        host_send_recv_callbacks,
    std::vector<DType> output_dtypes, std::vector<Shape> output_shapes,
    std::vector<ShardingRef> output_shardings,
    std::optional<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
        output_layouts)
    : client_(client),
      pjrt_loaded_executable_(std::move(pjrt_loaded_executable)),
      devices_(std::move(devices)),
      addressable_devices_(std::move(addressable_devices)),
      all_loaded_host_callbacks_(
          std::make_shared<std::vector<tsl::RCReference<LoadedHostCallback>>>(
              std::move(all_loaded_host_callbacks))),
      host_send_recv_callbacks_(std::move(host_send_recv_callbacks)),
      output_dtypes_(std::move(output_dtypes)),
      output_shapes_(std::move(output_shapes)),
      output_shardings_(std::move(output_shardings)),
      output_layouts_(std::move(output_layouts)),
      user_context_(UserContextScope::current()) {}

PjRtLoadedExecutable::~PjRtLoadedExecutable() = default;

absl::StatusOr<PjRtLoadedExecutable::ExecuteResult>
PjRtLoadedExecutable::Execute(absl::Span<ArrayRef> args,
                              const ExecuteOptions& options,
                              std::optional<DeviceListRef> devices) {
  DCHECK(this);
  // TODO(hyeontaek): Check input sharding consistency.

  // Convert an Array vector into 2-level PjRtBuffer vectors, optionally copying
  // to new devices.
  std::vector<std::vector<PjRtBuffer*>> argument_handles;
  std::vector<std::unique_ptr<PjRtBuffer>> owned_buffers;

  int num_computations;
  const bool portable_execution = devices.has_value();
  PjRtCompatibleDevice* portable_execution_device = nullptr;
  if (portable_execution) {
    if ((*devices)->size() != 1) {
      return absl::InvalidArgumentError(
          "Only single-shard portable execution is supported");
    }
    num_computations = 1;
    portable_execution_device =
        static_cast<PjRtDevice*>((*devices)->devices().front());
  } else {
    if (devices_->devices().empty()) {
      return absl::InvalidArgumentError(
          "No devices provided for portable executable");
    }
    num_computations = addressable_devices_.size();
  }

  argument_handles.resize(num_computations);
  for (int i = 0; i < num_computations; ++i) {
    argument_handles[i].reserve(args.size());
  }
  for (int i = 0; i < args.size(); ++i) {
    auto* pjrt_array =
        llvm::dyn_cast_or_null<PjRtCompatibleArray>(args[i].get());
    if (!pjrt_array) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Only PjRtCompatibleArray is supported, but argument %d is %s", i,
          pjrt_array->DebugString()));
    }
    int j = 0;
    // TODO(hyeontaek): Check pjrt_array->pjrt_buffers().size() ==
    // num_computations
    for (const auto& pjrt_buffer : pjrt_array->pjrt_buffers()) {
      argument_handles[j].push_back(pjrt_buffer.get());
      ++j;
    }
  }

  zkx::ExecuteOptions opts;
  opts.untuple_result = true;
  opts.launch_id = options.launch_id;
  opts.use_major_to_minor_data_layout_for_callbacks = true;
  opts.non_donatable_input_indices = options.non_donatable_input_indices;

  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: ExecutionOptions::execution_stream_id
  // clang-format on
  // opts.execution_stream_id = options.execution_stream_id;
  // TODO(chokobole): Uncomment this. Dependency: ExecutionOptions::incarnations
  // absl::StatusOr<absl::flat_hash_map<int, IncarnationId>> incarnations =
  //     client()->Incarnations();
  // if (incarnations.ok()) {
  //   opts.incarnations = *std::move(incarnations);
  // } else {
  //   VLOG(3) << "Unable to get incarnations: " << incarnations.status();
  // }

  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: ExecutionOptions::call_location
  // clang-format on
  // if (options.custom_options.has_value()) {
  //   const auto& attributes = options.custom_options->map();
  //   // Check if the custom options contain a call location key.
  //   if (auto it =
  //           attributes.find(PjRtCompatibleLoadedExecutable::kCallLocation);
  //       it != attributes.end()) {
  //     const AttributeMap::Value& value = it->second;
  //     if (const auto* call_location =
  //             std::get_if<AttributeMap::StringValue>(&value)) {
  //       opts.call_location = call_location->value;
  //     }
  //   }
  // }

  auto context = std::make_unique<zkx::ExecuteContext>();
  auto platform_id = pjrt_loaded_executable_->client()->platform_id();
  // TODO(chokobole): Uncomment this. Dependency: FfiLoadedHostCallbacks
  // auto ffi_callbacks = std::make_unique<zkx::FfiLoadedHostCallbacks>();
  auto callbacks = std::make_unique<std::vector<void*>>();
  // Forward callbacks via FFI's ExecutionContext for CPU/GPU platforms only.
  if (platform_id == CpuId() || platform_id == CudaId() ||
      platform_id == RocmId() || platform_id == SyclId()) {
    for (const auto& loaded_host_callback : *all_loaded_host_callbacks_) {
      auto* ffi_loaded_host_callback =
          llvm::dyn_cast<PjRtFfiLoadedHostCallback>(loaded_host_callback.get());
      if (ffi_loaded_host_callback != nullptr) {
        void* callback = ffi_loaded_host_callback->callable();
        callbacks->push_back(callback);
      }
    }
    // NOTE(dsuo): For now, check that either all or none of the host callbacks
    // are FFI callbacks. Otherwise, we have an error.
    // TODO(b/406585850): Improve how we determine when loaded host callbacks
    // are forwarded to ffi::ExecutionContext.
    if (!callbacks->empty() &&
        callbacks->size() != all_loaded_host_callbacks_->size()) {
      return absl::InvalidArgumentError(
          "ifrt::LoadedHostCallbacks must either be all "
          "ifrt::PjRtFfiLoadedHostCallback or none.");
    }
    // TODO(chokobole): Uncomment this. Dependency: FfiLoadedHostCallbacks
    // ffi_callbacks->callbacks = callbacks->data();
    // ffi_callbacks->num_callbacks = callbacks->size();
    // ffi::TypeRegistry::TypeId type_id(FfiLoadedHostCallbacks::id.type_id);
    // CHECK_OK(context->ffi_context().Insert(type_id, ffi_callbacks.get()));
    opts.context = context.get();
  }

  // When using host callbacks on CPU, we need to use synchronous dispatch to
  // avoid deadlocks with reentrant callbacks. Note that this option only
  // affects the CPU runtime.
  if (!all_loaded_host_callbacks_->empty()) {
    opts.execution_mode = zkx::ExecuteOptions::ExecutionMode::kSynchronous;
  }

  std::unique_ptr<HostCallbackStates> host_callback_states;
  if (!host_send_recv_callbacks_.empty()) {
    host_callback_states = std::make_unique<HostCallbackStates>();
    for (int i = 0; i < num_computations; ++i) {
      auto& contexts = host_callback_states->contexts.emplace_back();
      auto& send_callbacks =
          host_callback_states->send_callbacks.emplace_back();
      auto& recv_callbacks =
          host_callback_states->recv_callbacks.emplace_back();

      for (const auto& host_send_recv_callback : host_send_recv_callbacks_) {
        contexts.push_back(CreateHostCallbackStateAndAppendSendRecvCallbacks(
            host_send_recv_callback->host_callback(),
            /*host_memory_for_device_manager=*/nullptr, send_callbacks,
            recv_callbacks, opts.use_major_to_minor_data_layout_for_callbacks));
      }
    }
    opts.send_callbacks = host_callback_states->send_callbacks;
    opts.recv_callbacks = host_callback_states->recv_callbacks;
  }

  // Execute the computation.
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> pjrt_outputs;
  zkx::ifrt::Future<> status;
  if (portable_execution) {
    std::optional<PjRtFuture<>> returned_pjrt_future;
    TF_RET_CHECK(portable_execution_device->IsAddressable());
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<PjRtBuffer>> single_device_pjrt_results,
        pjrt_loaded_executable_->ExecutePortable(
            argument_handles.front(), portable_execution_device->pjrt_device(),
            opts, returned_pjrt_future,
            /*fill_future=*/true));

    pjrt_outputs.push_back(std::move(single_device_pjrt_results));
    status = *std::move(returned_pjrt_future);
  } else {
    std::optional<std::vector<PjRtFuture<>>> returned_pjrt_futures;
    returned_pjrt_futures.emplace();

    TF_ASSIGN_OR_RETURN(
        pjrt_outputs, pjrt_loaded_executable_->Execute(argument_handles, opts,
                                                       returned_pjrt_futures));

    status = JoinFutures(absl::MakeSpan(*returned_pjrt_futures));
  }

  if (!all_loaded_host_callbacks_->empty()) {
    // For host callbacks to work, returned futures must be supported so that we
    // can use the futures to extend the lifetime of the host callbacks until
    // the execution finishes.
    status.OnReady(
        [all_loaded_host_callbacks = all_loaded_host_callbacks_,
         host_callback_states = std::move(host_callback_states),
         // clang-format off
         // TODO(chokobole): Uncomment this. Dependency: ExecuteContext
         // clang-format on
         // context = std::move(context),
         // clang-format off
         // TODO(chokobole): Uncomment this. Dependency: FfiLoadedHostCallbacks
         // clang-format on
         //  ffi_callbacks = std::move(ffi_callbacks),
         callbacks = std::move(callbacks)](absl::Status) mutable {
          all_loaded_host_callbacks.reset();
        });
  }

  // Convert 2-level PjRtBuffer vectors into an Array vector.
  std::vector<ArrayRef> outputs;
  // TODO(hyeontaek): Check output dtype/shape consistency with the actual
  // output.
  if (pjrt_outputs.size() != num_computations) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Unexpected number of computations in outputs: %d vs. %d",
        pjrt_outputs.size(), num_computations));
  }
  const int num_outputs = pjrt_outputs.empty() ? output_dtypes_.size()
                                               : pjrt_outputs.front().size();
  if (num_outputs != output_dtypes_.size()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Unexpected number of outputs: %d vs. %d", num_outputs,
                        output_dtypes_.size()));
  }
  outputs.reserve(num_outputs);
  // Single-device Shardings for portable execution. Outputs with the same
  // memory_kind shares the same Sharding object.
  absl::flat_hash_map<MemoryKind, ShardingRef> single_device_shardings;

  std::vector<std::shared_ptr<const zkx::PjRtLayout>> layouts;
  layouts.reserve(num_outputs);
  if (output_layouts_.has_value()) {
    // TODO(hyeontaek): Once we can get `output_layouts_` reliably, only keep
    // this path.
    layouts = *output_layouts_;
  } else if (!pjrt_outputs.empty()) {
    for (int i = 0; i < num_outputs; ++i) {
      auto layout = output_dtypes_[i].kind() == zkx::ifrt::DType::kToken
                        ? std::make_shared<zkx::PjRtLayout>(zkx::Layout())
                        : pjrt_outputs.front()[i]->layout();
      layouts.push_back(std::move(layout));
    }
  } else {
    auto maybe_layouts = GetOutputLayouts();
    if (absl::IsUnimplemented(maybe_layouts.status())) {
      for (int i = 0; i < num_outputs; ++i) {
        std::shared_ptr<const zkx::PjRtLayout> layout;
        if (output_dtypes_[i].kind() == zkx::ifrt::DType::kToken) {
          layout = std::make_shared<zkx::PjRtLayout>(zkx::Layout());
        } else {
          TF_ASSIGN_OR_RETURN(layout,
                              client_->GetDefaultPjRtLayout(
                                  output_dtypes_[i], output_shapes_[i].dims(),
                                  devices_->devices().front(),
                                  output_shardings_[i]->memory_kind()));
        }
        layouts.push_back(std::move(layout));
      }
    } else {
      TF_RETURN_IF_ERROR(maybe_layouts.status());
      layouts = *std::move(maybe_layouts);
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    PjRtArray::PjRtBuffers buffers;
    buffers.reserve(num_computations);
    const MemoryKind dst_memory_kind = output_shardings_[i]->memory_kind();
    const MemoryKind canonical_dst_memory_kind = CanonicalizeMemoryKind(
        dst_memory_kind, output_shardings_[i]->devices()->devices().front());

    for (int j = 0; j < num_computations; ++j) {
      if (j > 0) {
        if (auto memory_kind =
                MakeMemoryKindFromPjRtBuffer(pjrt_outputs[j][i].get());
            canonical_dst_memory_kind !=
            CanonicalizeMemoryKindWithPjRtDevice(
                memory_kind, pjrt_outputs[j][i]->device())) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "Memory kind mismatch. Got sharding with memory kind '%v' and "
              "buffer with memory_kind '%v'",
              dst_memory_kind, memory_kind));
        }
      }
      buffers.push_back(
          std::shared_ptr<PjRtBuffer>(pjrt_outputs[j][i].release()));
    }
    std::optional<ShardingRef> sharding;
    if (portable_execution) {
      if (auto it = single_device_shardings.find(dst_memory_kind);
          it == single_device_shardings.end()) {
        sharding = single_device_shardings
                       .insert({dst_memory_kind, SingleDeviceSharding::Create(
                                                     portable_execution_device,
                                                     dst_memory_kind)})
                       .first->second;
      } else {
        sharding = it->second;
      }
    } else {
      sharding = output_shardings_[i];
    }
    outputs.push_back(*PjRtArray::Create(
        client_, output_dtypes_[i], output_shapes_[i], *std::move(sharding),
        std::move(buffers), std::move(layouts[i])));
  }

  ExecuteResult result;
  if (options.fill_status) {
    result.status = status;
  }
  result.outputs = std::move(outputs);
  return result;
}

absl::StatusOr<std::optional<std::string>> PjRtLoadedExecutable::Fingerprint()
    const {
  DCHECK(this);
  absl::StatusOr<std::string> fingerprint =
      pjrt_loaded_executable_->FingerprintExecutable();
  if (fingerprint.ok()) {
    return {fingerprint.value()};
  } else if (fingerprint.status().code() == absl::StatusCode::kUnimplemented) {
    // Return nullopt in case of unimplemented error.
    return std::nullopt;
  } else {
    return fingerprint.status();
  }
}

absl::StatusOr<std::string> PjRtLoadedExecutable::Serialize() const {
  DCHECK(this);
  return pjrt_loaded_executable_->SerializeExecutable();
}

}  // namespace zkx::ifrt
