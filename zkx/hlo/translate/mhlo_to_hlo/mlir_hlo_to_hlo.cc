/* Copyright 2019 The OpenXLA Authors.
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

#include "zkx/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_field.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

#include "stablehlo/dialect/Base.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/comparison_util.h"
#include "zkx/debug_options_flags.h"
#include "zkx/hlo/builder/zkx_computation.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "zkx/hlo/translate/mhlo_to_hlo/literal_exporter.h"
#include "zkx/hlo/translate/mhlo_to_hlo/location_exporter.h"
#include "zkx/hlo/translate/mhlo_to_hlo/module_attributes_exporter.h"
#include "zkx/hlo/translate/mhlo_to_hlo/stack_frame_index_builder.h"
#include "zkx/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "zkx/layout.h"
#include "zkx/layout_util.h"
#include "zkx/mlir/mlir_utils.h"
#include "zkx/mlir/utils/error_util.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "zkx/mlir_hlo/mhlo/transforms/passes.h"
#include "zkx/service/gpu/backend_configs.pb.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/zkx_data.pb.h"

namespace {

// Boolean attribute.
constexpr char kJaxBufferDonor[] = "jax.buffer_donor";

// BitcastOp lowering strings.
constexpr char kResultLayout[] = "result_layout";
constexpr char kSourceLayout[] = "source_layout";

// CustomCallOp lowering strings.
constexpr char kBackendConfig[] = "backend_config";

// MHLO attributes. Module level attributes require namespacing.
constexpr char kMhloFrontendAttributes[] = "mhlo.frontend_attributes";
constexpr char kMhloInputOutputAlias[] = "mhlo.input_output_alias";
constexpr char kMhloIsDynamic[] = "mhlo.is_dynamic";
constexpr char kMhloParameterReplication[] = "mhlo.parameter_replication";
constexpr char kMhloReplication[] = "mhlo.is_same_data_across_replicas";
constexpr char kMhloSharding[] = "mhlo.sharding";
constexpr char kMhloSpmdOutputSharding[] = "mhlo.spmd_output_sharding";
constexpr char kMhloSpmdParametersShardings[] =
    "mhlo.spmd_parameters_shardings";
constexpr char kMhloUseAutoSpmdPartitioning[] =
    "mhlo.use_auto_spmd_partitioning";
constexpr char kMhloZkxEntryComputationParameterLayouts[] =
    "mhlo.zkx_entry_computation_parameter_layouts";
constexpr char kMhloZkxEntryComputationParameterTiles[] =
    "mhlo.zkx_entry_computation_parameter_tiles";
constexpr char kMhloZkxEntryComputationResultLayout[] =
    "mhlo.zkx_entry_computation_result_layout";
constexpr char kMhloZkxEntryComputationResultTiles[] =
    "mhlo.zkx_entry_computation_result_tiles";

// Miscellaneous string literals.
constexpr char kArgEmptyTuple[] = "arg_empty_tuple";
constexpr char kArgPrefix[] = "Arg_";
constexpr char kArgTuple[] = "arg_tuple";
constexpr char kDefaultLayoutAttrName[] = "zkx_shape";
constexpr char kExecutionThread[] = "execution_thread";
// Array attribute. Same shape as infeed result, but contains a
// minor_to_major array for every tensor.
constexpr char kMain[] = "main";
constexpr char kRegionPrefix[] = "region_";
constexpr char kTfAliasingOutput[] = "tf.aliasing_output";

// Passes through everything except for unique_ptr, on which it calls get().
// This exists to allow the generated code to call ZKX functions that take a raw
// pointer.
template <typename T>
T Unwrap(T t) {
  return t;
}

template <typename T>
T* Unwrap(const std::unique_ptr<T>& t) {
  return t.get();
}

mlir::LogicalResult GetZkxOp(
    mlir::Value val, const llvm::DenseMap<mlir::Value, zkx::ZkxOp>& val_map,
    zkx::ZkxOp* result, mlir::Operation* op) {
  auto iter = val_map.find(val);
  if (iter == val_map.end()) {
    return op->emitOpError(
        "requires all operands to be defined in the parent region for export");
  }
  *result = iter->second;
  return mlir::success();
}

bool IsBoundedOrStatic(mlir::Type ty) {
  auto ranked_ty = mlir::dyn_cast_or_null<mlir::RankedTensorType>(ty);
  if (!ranked_ty) return false;

  if (ranked_ty.hasStaticShape()) return true;

  auto encoding = mlir::dyn_cast_or_null<mlir::mhlo::TypeExtensionsAttr>(
      ranked_ty.getEncoding());
  if (!encoding || encoding.getBounds().empty()) return false;

  int64_t rank = ranked_ty.getRank();
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (ranked_ty.isDynamicDim(dim) &&
        encoding.getBounds()[dim] == mlir::ShapedType::kDynamic)
      return false;
  }
  return true;
}

uint32_t Convertuint32_t(uint32_t i) { return i; }
uint64_t Convertuint64_t(uint64_t i) { return i; }

std::vector<int64_t> ConvertDenseIntAttr(mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

std::vector<int64_t> ConvertDenseIntAttr(
    std::optional<mlir::DenseIntElementsAttr> attr) {
  if (!attr) return {};
  return ConvertDenseIntAttr(*attr);
}

// Converts the broadcast_dimensions attribute into a vector of dimension
// numbers (empty if the attribute is absent).
std::vector<int64_t> Convert_broadcast_dimensions(
    std::optional<mlir::DenseIntElementsAttr> broadcast_dimensions) {
  if (!broadcast_dimensions.has_value()) return {};

  return ConvertDenseIntAttr(*broadcast_dimensions);
}

zkx::Layout ExtractLayout(mlir::Operation* op, int rank,
                          llvm::StringRef attr_name = kDefaultLayoutAttrName) {
  if (auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(attr_name)) {
    llvm::SmallVector<int64_t, 4> minor_to_major;
    DCHECK_EQ(rank, attr.size());
    minor_to_major.reserve(attr.size());
    for (const llvm::APInt& i : attr) {
      minor_to_major.push_back(i.getZExtValue());
    }
    return zkx::LayoutUtil::MakeLayout(minor_to_major);
  }
  return zkx::LayoutUtil::MakeDescendingLayout(rank);
}

// Returns a failure or a valid ZKX shape corresponding to the given op's
// results.
mlir::FailureOr<zkx::Shape> ExtractZkxShape(mlir::Operation* op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kDefaultLayoutAttrName)) {
    return *zkx::ParseShape(
        std::string_view(attr.getValue().data(), attr.getValue().size()));
  } else {
    std::vector<zkx::Shape> subshapes;
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      subshapes.push_back(zkx::TypeToShape(result.getType()));
      if (subshapes.back().element_type() == zkx::PRIMITIVE_TYPE_INVALID) {
        return op->emitError()
               << "result #" << index << " type is not supported";
      }
    }
    if (subshapes.size() > 1) {
      return zkx::ShapeUtil::MakeTupleShape(subshapes);
    }
    return subshapes[0];
  }
}

#define I64_ELEMENTS_ATTR_TO_VECTOR(attribute)               \
  std::vector<int64_t> Convert_##attribute(                  \
      std::optional<mlir::DenseIntElementsAttr> attribute) { \
    return ConvertDenseIntAttr(attribute);                   \
  }

I64_ELEMENTS_ATTR_TO_VECTOR(broadcast_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(permutation);
I64_ELEMENTS_ATTR_TO_VECTOR(start_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(limit_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(strides);
I64_ELEMENTS_ATTR_TO_VECTOR(slice_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(dimensions);

#undef I64_ELEMENTS_ATTR_TO_VECTOR

// Converts the comparison_direction string attribute into the ZKX enum. The
// string is assumed to correspond to exactly one of the allowed strings
// representing the enum. This should have been checked in the op verify method.
zkx::ComparisonDirection Convert_comparison_direction(
    llvm::StringRef comparison_direction_string) {
  return zkx::StringToComparisonDirection(comparison_direction_string.str())
      .value();
}

zkx::ScatterDimensionNumbers Convert_scatter_dimension_numbers(
    mlir::mhlo::ScatterDimensionNumbersAttr input) {
  zkx::ScatterDimensionNumbers output;

  auto update_window_dims = input.getUpdateWindowDims();
  std::copy(update_window_dims.begin(), update_window_dims.end(),
            google::protobuf::RepeatedFieldBackInserter(
                output.mutable_update_window_dims()));

  auto inserted_window_dims = input.getInsertedWindowDims();
  std::copy(inserted_window_dims.begin(), inserted_window_dims.end(),
            google::protobuf::RepeatedFieldBackInserter(
                output.mutable_inserted_window_dims()));

  auto input_batching_dims = input.getInputBatchingDims();
  std::copy(input_batching_dims.begin(), input_batching_dims.end(),
            google::protobuf::RepeatedFieldBackInserter(
                output.mutable_input_batching_dims()));

  auto scatter_indices_batching_dims = input.getScatterIndicesBatchingDims();
  std::copy(scatter_indices_batching_dims.begin(),
            scatter_indices_batching_dims.end(),
            google::protobuf::RepeatedFieldBackInserter(
                output.mutable_scatter_indices_batching_dims()));

  auto scatter_dims_to_operand_dims = input.getScatterDimsToOperandDims();
  std::copy(scatter_dims_to_operand_dims.begin(),
            scatter_dims_to_operand_dims.end(),
            google::protobuf::RepeatedFieldBackInserter(
                output.mutable_scatter_dims_to_operand_dims()));

  output.set_index_vector_dim(input.getIndexVectorDim());
  return output;
}

// Returns an OpSharding proto from the "sharding" attribute of the op. If the
// op doesn't have a sharding attribute or the sharding attribute is invalid,
// returns std::nullopt.
std::optional<zkx::OpSharding> CreateOpShardingFromAttribute(
    mlir::Operation* op) {
  auto shardingAttr = op->getAttrOfType<mlir::StringAttr>(kMhloSharding);
  if (!shardingAttr) return std::nullopt;
  return zkx::ConvertSharding(shardingAttr.getValue());
}

// Returns a FrontendAttributes proto from the "frontend_attributes" attribute
// of the op. An empty FrontendAttributes proto is returned if an op does not
// have frontend attributes.
void CreateFrontendAttributes(mlir::ArrayRef<mlir::NamedAttribute> named_attrs,
                              zkx::FrontendAttributes& frontend_attributes) {
  for (const auto& attr : named_attrs)
    if (auto value_str_attr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
      frontend_attributes.mutable_map()->insert(
          {attr.getName().str(), value_str_attr.getValue().str()});
}

// Returns a FrontendAttributes proto from the "frontend_attributes" attribute
// of the op. An empty FrontendAttributes proto is returned if an op does not
// have frontend attributes.
void CreateFrontendAttributes(
    const mlir::DictionaryAttr& frontend_attributes_dict,
    zkx::FrontendAttributes& frontend_attributes) {
  CreateFrontendAttributes(frontend_attributes_dict.getValue(),
                           frontend_attributes);
}

zkx::FrontendAttributes CreateZkxFrontendAttributesFromOp(mlir::Operation* op) {
  zkx::FrontendAttributes frontend_attributes;
  auto frontend_attributes_dict =
      op->getAttrOfType<mlir::DictionaryAttr>(kMhloFrontendAttributes);
  if (!frontend_attributes_dict) return frontend_attributes;
  CreateFrontendAttributes(frontend_attributes_dict, frontend_attributes);
  return frontend_attributes;
}

void ExtractFrontendAttributesFromFunction(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<std::optional<zkx::FrontendAttributes>>* fe_attrs) {
  fe_attrs->resize(function.getNumArguments(), std::nullopt);
  for (int i = 0, end = function.getNumArguments(); i < end; ++i)
    if (auto fe_attr = function.getArgAttrOfType<mlir::DictionaryAttr>(
            i, kMhloFrontendAttributes)) {
      zkx::FrontendAttributes frontend_attributes;
      CreateFrontendAttributes(fe_attr, frontend_attributes);
      (*fe_attrs)[i] = frontend_attributes;
    }
}

bool SomeOptionalShardingsAreSet(
    llvm::ArrayRef<std::optional<zkx::OpSharding>> shardings) {
  return llvm::any_of(shardings,
                      [](const std::optional<zkx::OpSharding>& sharding) {
                        return sharding.has_value();
                      });
}

// Extracts argument and result shardings from function.
void ExtractShardingsFromFunction(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<std::optional<zkx::OpSharding>>* arg_shardings,
    llvm::SmallVectorImpl<std::optional<zkx::OpSharding>>* ret_shardings) {
  arg_shardings->resize(function.getNumArguments(),
                        std::optional<zkx::OpSharding>());
  for (int i = 0, end = function.getNumArguments(); i < end; ++i)
    if (auto sharding =
            function.getArgAttrOfType<mlir::StringAttr>(i, kMhloSharding))
      (*arg_shardings)[i] = zkx::ConvertSharding(sharding.getValue());

  ret_shardings->resize(function.getNumResults(),
                        std::optional<zkx::OpSharding>());
  for (int i = 0, end = function.getNumResults(); i < end; ++i)
    if (auto sharding =
            function.getResultAttrOfType<mlir::StringAttr>(i, kMhloSharding))
      (*ret_shardings)[i] = zkx::ConvertSharding(sharding.getValue());
}

// Creates a tuple sharding with the given shardings if at least one is present.
//
// Adds replicated shardings for any missing tuple shardings.
std::optional<zkx::OpSharding> CreateTupleSharding(
    llvm::ArrayRef<std::optional<zkx::OpSharding>> tuple_shardings) {
  if (tuple_shardings.empty() ||
      !SomeOptionalShardingsAreSet(tuple_shardings)) {
    return std::nullopt;
  }
  zkx::OpSharding sharding;
  sharding.set_type(zkx::OpSharding::TUPLE);
  for (const std::optional<zkx::OpSharding>& tuple_sharding : tuple_shardings) {
    if (tuple_sharding) {
      *sharding.add_tuple_shardings() = *tuple_sharding;
    } else {
      zkx::OpSharding fallback_sharding;
      fallback_sharding.set_type(zkx::OpSharding::REPLICATED);
      *sharding.add_tuple_shardings() = fallback_sharding;
    }
  }

  return sharding;
}

// If `ops` has a single element, returns that element. Otherwise, returns
// a tuple instruction with `ops` and attaches a tuple sharding from
// `shardings`.
zkx::ZkxOp CreateTupleIfMultipleOps(
    zkx::ZkxBuilder* builder, llvm::ArrayRef<zkx::ZkxOp> ops,
    llvm::ArrayRef<std::optional<zkx::OpSharding>> shardings) {
  if (ops.size() == 1) {
    return ops[0];
  }
  zkx::ZkxScopedShardingAssignment scoped_sharding(
      builder, CreateTupleSharding(shardings));
  return Tuple(builder, ops);
}

// Returns the flattened result shardings of the given `op_sharding`, i.e.,
// either:
// - an empty vector if `op_sharding` is `std::nullopt`.
// - the tuple shardings in `op_sharding` if it has type TUPLE.
// - otherwise, returns a vector of size `num_results` filled with
//   `op_sharding`.
llvm::SmallVector<std::optional<zkx::OpSharding>> GetResultShardings(
    std::optional<zkx::OpSharding> op_sharding, int64_t num_results) {
  if (!op_sharding) {
    return {};
  }
  llvm::SmallVector<std::optional<zkx::OpSharding>> res_shardings;
  res_shardings.reserve(num_results);
  if (op_sharding->type() == zkx::OpSharding::TUPLE) {
    assert(op_sharding->tuple_shardings_size() == num_results);
    res_shardings.assign(op_sharding->tuple_shardings().begin(),
                         op_sharding->tuple_shardings().end());
  } else {
    res_shardings.append(num_results, op_sharding);
  }
  return res_shardings;
}

// Returns the OpSharding of each op in `zkx_ops`, or std::nullopt if the op
// doesn't have a sharding.
llvm::SmallVector<std::optional<zkx::OpSharding>> GetZkxOpShardings(
    llvm::ArrayRef<zkx::ZkxOp> zkx_ops) {
  llvm::SmallVector<std::optional<zkx::OpSharding>> shardings;
  shardings.reserve(zkx_ops.size());
  for (const zkx::ZkxOp& zkx_op : zkx_ops) {
    auto sharding = zkx_op.builder()->GetOpSharding(zkx_op);
    assert(sharding.ok() && "can't find ZkxOp for argument");
    shardings.push_back(*sharding);
  }
  return shardings;
}

}  // namespace

namespace mlir {
namespace {

class ConvertToHloModule {
 public:
  using ValueLoweringMap = llvm::DenseMap<Value, zkx::ZkxOp>;
  using FunctionLoweringMap = llvm::DenseMap<func::FuncOp, zkx::ZkxComputation>;

  // If use_tuple_args is true, then the entry function's arguments are
  // converted to a tuple and passed as a single parameter.
  // Similarly, if return tuple is true, then the entry function's return values
  // are converted to a tuple even when there is only a single return value.
  // Multiple return values are always converted to a tuple and returned as a
  // single value.
  explicit ConvertToHloModule(ModuleOp module, zkx::ZkxBuilder& module_builder,
                              MlirToHloConversionOptions options)
      : module_(module), module_builder_(module_builder), options_(options) {}

  // Perform the lowering to ZKX. This function returns failure if an error was
  // encountered.
  //
  // TODO(hinsu): Check for dynamic shapes and exit instead of crashing.
  LogicalResult Run() {
    auto main = module_.lookupSymbol<func::FuncOp>(kMain);
    if (!main)
      return module_.emitError(
          "conversion requires module with `main` function");

    for (auto func : module_.getOps<func::FuncOp>()) {
      if (func.empty()) continue;
      if (failed(RunOnFunction(func))) return failure();
    }
    return success();
  }

  // Lower a specific function to HLO.
  LogicalResult RunOnFunction(func::FuncOp f);

  zkx::HloModuleProto ConsumeMainProto() {
    auto main = module_.lookupSymbol<func::FuncOp>(kMain);
    // This is an invariant check as Run returns failure if there is no main
    // function and so the main proto shouldn't be consumed in that case.
    CHECK(main) << "requires module to have main function";  // Crash Ok.
    return lowered_computation_[main].proto();
  }

  // Lower a `Region` to a `ZkxComputation`
  LogicalResult LowerRegionAsComputation(
      Region* region, zkx::ZkxComputation* func,
      llvm::ArrayRef<Value> implicit_operands = {},
      llvm::ArrayRef<Value> implicit_results = {},
      bool ensure_single_arg = false,
      llvm::ArrayRef<std::optional<zkx::OpSharding>> arg_shardings = {},
      llvm::ArrayRef<std::optional<zkx::OpSharding>> ret_shardings = {});

  // Lower a single `Block` to a `ZkxComputation`
  LogicalResult LowerBasicBlockAsFunction(
      Block* block, zkx::ZkxBuilder* builder, bool is_entry_function,
      bool ensure_single_arg,
      const std::vector<bool>& entry_args_same_across_replicas,
      llvm::ArrayRef<std::optional<zkx::OpSharding>> arg_shardings,
      llvm::ArrayRef<std::optional<zkx::OpSharding>> ret_shardings,
      llvm::ArrayRef<std::optional<zkx::FrontendAttributes>> fe_attrs,
      zkx::ZkxComputation* result, llvm::ArrayRef<Value> implicit_operands = {},
      llvm::ArrayRef<Value> implicit_results = {});

  // Lower cast to HLO cast instruction
  LogicalResult LowerCast(Operation* inst,
                          const MlirToHloConversionOptions& options,
                          ConvertToHloModule::ValueLoweringMap* value_lowering);

  // Lower constant to HLO constant instruction
  LogicalResult LowerConstant(
      Operation* inst, zkx::ZkxBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      ElementsAttr const_attr);

  // Lower function call to HLO call instruction
  LogicalResult LowerFunctionCall(
      func::CallOp call_op, zkx::ZkxBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering);

  // Lower MHLO and Func return instructions to HLO return instruction
  LogicalResult LowerReturn(
      Operation* inst, bool is_entry_function,
      llvm::ArrayRef<std::optional<zkx::OpSharding>> ret_shardings,
      llvm::ArrayRef<Value> implicit_results, zkx::ZkxBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      zkx::ZkxOp* return_value, const MlirToHloConversionOptions& options);

  // Extract shape from instruction and propagate layouts to the ZKX op.
  LogicalResult PropagateLayouts(const MlirToHloConversionOptions& options,
                                 Operation* inst, zkx::ZkxOp zkx_op);

  // Look up a symbol with the specified name, returning null if no such name
  // exists.
  func::FuncOp LookUpSymbol(FlatSymbolRefAttr symbol) {
    return module_.lookupSymbol<func::FuncOp>(symbol);
  }

  // Get Reference to lowered ZKX computation for a function.
  zkx::ZkxComputation& GetLoweredComputation(func::FuncOp func) {
    return lowered_computation_[func];
  }

  LogicalResult Lower(
      Operation* inst, bool is_entry_function,
      llvm::ArrayRef<std::optional<zkx::OpSharding>> ret_shardings,
      llvm::ArrayRef<Value> implicit_results, zkx::ZkxBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      zkx::ZkxOp* return_value);

  const MlirToHloConversionOptions& GetOptions() const { return options_; }

  zkx::StackFrameIndexProto BuildStackFramesIndexProto() {
    return stack_frame_indexes_builder_.Build();
  }

 private:
  LogicalResult SetEntryTupleShapesAndLeafReplication(
      Block* block, const std::vector<bool>& entry_args_same_across_replicas,
      llvm::SmallVectorImpl<zkx::Shape>* arg_shapes,
      std::vector<bool>* leaf_replication);

  LogicalResult SetEntryTupleShardings(
      Block* block, zkx::ZkxBuilder* builder,
      llvm::ArrayRef<std::optional<zkx::OpSharding>> arg_shardings,
      llvm::SmallVectorImpl<zkx::Shape>* arg_shapes);

  // The module being lowered.
  ModuleOp module_;

  // The top-level ZkxBuilder.
  zkx::ZkxBuilder& module_builder_;

  // Common stack frame index builder.
  StackFrameIndexBuilder stack_frame_indexes_builder_;

  // Map between function and lowered computation.
  FunctionLoweringMap lowered_computation_;

  // Unique suffix to give to the name of the next lowered region.
  size_t region_id_ = 0;

  // Conversion options
  MlirToHloConversionOptions options_;
};

}  // namespace
}  // namespace mlir

namespace {

struct OpLoweringContext {
  llvm::DenseMap<mlir::Value, zkx::ZkxOp>* values;
  mlir::ConvertToHloModule* converter;
  zkx::ZkxBuilder* builder;
  mlir::StackFrameIndexBuilder* frame_index_builder;
};

mlir::LogicalResult GetTuple(mlir::Operation* op,
                             mlir::Operation::operand_range values,
                             OpLoweringContext ctx,
                             llvm::SmallVectorImpl<zkx::ZkxOp>& results) {
  results.reserve(values.size());
  for (mlir::Value value : values) {
    if (failed(GetZkxOp(value, *ctx.values, &results.emplace_back(), op)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult GetZkxOps(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> values,
                              OpLoweringContext ctx,
                              llvm::SmallVectorImpl<zkx::ZkxOp>& results) {
  results.reserve(values.size());
  for (mlir::Value value : values) {
    if (failed(GetZkxOp(value, *ctx.values, &results.emplace_back(), op)))
      return mlir::failure();
  }
  return mlir::success();
}

void BuildGetTupleElementsForTupleResults(
    mlir::Operation* op, zkx::ZkxOp tuple, zkx::ZkxBuilder* builder,
    llvm::DenseMap<mlir::Value, zkx::ZkxOp>& values,
    unsigned num_implicit_results = 0) {
  const std::optional<zkx::OpSharding>& sharding = builder->sharding();
  if (sharding.has_value()) {
    bool is_tuple_sharding = sharding->type() == zkx::OpSharding::TUPLE;
    assert(!is_tuple_sharding || (op->getNumResults() + num_implicit_results ==
                                  sharding->tuple_shardings_size()));
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      // If `sharding` is not a tuple sharding, then every `get-tuple-element`
      // gets the same sharding.
      zkx::ZkxScopedShardingAssignment scoped_sharding(
          builder,
          is_tuple_sharding ? sharding->tuple_shardings(index) : sharding);
      values[result] = zkx::GetTupleElement(tuple, index);
    }
  } else {
    zkx::ZkxScopedShardingAssignment scoped_sharding(builder, std::nullopt);
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      values[result] = zkx::GetTupleElement(tuple, index);
    }
  }
}

void BuildGetTupleElementsForTupleResults(mlir::Operation* op, zkx::ZkxOp tuple,
                                          OpLoweringContext ctx,
                                          unsigned num_implicit_results = 0) {
  BuildGetTupleElementsForTupleResults(op, tuple, ctx.builder, *ctx.values,
                                       num_implicit_results);
}

}  // namespace

namespace mlir::mhlo {
namespace {

LogicalResult ExportZkxOp(DynamicBroadcastInDimOp op, OpLoweringContext ctx) {
  // This op has no expression in the legacy export format.
  return failure();
}

LogicalResult ExportZkxOp(DynamicIotaOp op, OpLoweringContext ctx) {
  // TODO(b/264240901): Implement MHLO export for DynamicIotaOp.
  return failure();
}

LogicalResult ExportZkxOp(DynamicPadOp op, OpLoweringContext ctx) {
  // TODO(b/264240901): Implement MHLO export for DynamicPadOp.
  return failure();
}

LogicalResult ExportZkxOp(DynamicReshapeOp op, OpLoweringContext ctx) {
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType) return op->emitOpError() << "expected ranked result";
  auto resultBounds = hlo::encodingToBounds(resultType.getEncoding());
  if (resultBounds.empty())
    return op->emitOpError() << "expected bounded result";
  auto shapeType = dyn_cast<RankedTensorType>(op.getOutputShape().getType());
  if (!shapeType || !shapeType.getElementType().isInteger(32))
    return op->emitOpError() << "expected output shape to be tensor<Nxi32>";

  auto& value_map = *ctx.values;
  zkx::ZkxOp operand;
  zkx::ZkxOp outputShape;
  if (failed(GetZkxOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetZkxOp(op.getOutputShape(), value_map, &outputShape, op)))
    return failure();

  SmallVector<zkx::ZkxOp> dimSizes;
  SmallVector<int64_t> newSizeBounds;
  std::vector<bool> dimsAreDynamic;
  for (auto i = 0; i < resultType.getRank(); ++i) {
    auto runtimeSizeX1 = zkx::Slice(outputShape, {i}, {i + 1}, {1});
    dimSizes.push_back(zkx::Reshape(runtimeSizeX1, {}));

    auto dimSize = resultType.getDimSize(i);
    auto dimBound = resultBounds[i];
    if (!hlo::isStaticDimSize(dimSize) && !hlo::isStaticDimSize(dimBound))
      return op->emitOpError() << "unbounded dynamism is not supported";
    newSizeBounds.push_back(hlo::isStaticDimSize(dimSize) ? dimSize : dimBound);
    dimsAreDynamic.push_back(!hlo::isStaticDimSize(dimSize));
  }
  value_map[op] =
      zkx::DynamicReshape(operand, dimSizes, newSizeBounds, dimsAreDynamic);
  return success();
}

LogicalResult ExportZkxOp(RealDynamicSliceOp op, OpLoweringContext ctx) {
  // TODO(b/264240901): Implement MHLO export for RealDynamicSliceOp.
  return failure();
}

LogicalResult ExportZkxOp(BitcastConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxOp operand;
  if (failed(GetZkxOp(op.getOperand(), value_map, &operand, op)))
    return failure();

  value_map[op] = zkx::BitcastConvertType(
      operand, zkx::mlir_utils::MlirTypeToPrimitiveTypeWithSign(
                   getElementTypeOrSelf(op.getType())));
  return success();
}

LogicalResult ExportZkxOp(BroadcastInDimOp op, OpLoweringContext ctx) {
  auto type = dyn_cast<RankedTensorType>(op.getType());
  if (!type) return failure();
  auto& value_map = *ctx.values;
  zkx::ZkxOp operand;
  if (failed(GetZkxOp(op.getOperand(), value_map, &operand, op)))
    return failure();

  // Use TypeToShape to handle bounded dynamism.
  // HLO expects broadcast sizes to use the bound's value, not kDynamic.
  zkx::Shape shape = zkx::TypeToShape(type);
  value_map[op] =
      BroadcastInDim(operand, shape.dimensions(),
                     Convert_broadcast_dimensions(op.getBroadcastDimensions()));
  return success();
}

LogicalResult ExportZkxOp(IfOp op, OpLoweringContext ctx) {
  zkx::ZkxComputation true_branch;
  zkx::ZkxComputation false_branch;
  auto& value_map = *ctx.values;

  // mhlo.IfOp does not have any operands or blocks arguments. The computation
  // inside the region-blocks use implicit captures of values defined above.
  // In order to create the zkx parameters for functions corresponding to
  // IfOp regions, we need to infer the a region-block's arguments, using all
  // the values used in the region but defined above. Note that in case there
  // are zero implicit capture for a region, we use an empty tuple as the zkx
  // parameter.
  //
  // Note that the implicit values used in true and false branch regions might
  // be different and, as a result, the zkx parameters for the corresponding
  // regions could have different shapes.
  llvm::SetVector<Value> implicit_true_operand_set, implicit_false_operand_set;
  getUsedValuesDefinedAbove(op.getTrueBranch(), op.getTrueBranch(),
                            implicit_true_operand_set);
  getUsedValuesDefinedAbove(op.getFalseBranch(), op.getFalseBranch(),
                            implicit_false_operand_set);

  llvm::SmallVector<Value> implicit_true_operands =
      implicit_true_operand_set.takeVector();
  llvm::SmallVector<Value> implicit_false_operands =
      implicit_false_operand_set.takeVector();

  llvm::SmallVector<std::optional<zkx::OpSharding>> ret_shardings =
      GetResultShardings(ctx.builder->sharding(), op->getNumResults());

  llvm::SmallVector<zkx::ZkxOp> true_args;
  if (failed(GetZkxOps(op, implicit_true_operands, ctx, true_args)))
    return failure();

  llvm::SmallVector<zkx::ZkxOp> false_args;
  if (failed(GetZkxOps(op, implicit_false_operands, ctx, false_args)))
    return failure();

  llvm::SmallVector<std::optional<zkx::OpSharding>> true_arg_shardings,
      false_arg_shardings;
  if (!ret_shardings.empty()) {
    // We only add arg shardings if there are result shardings, otherwise it
    // means sharding propagation hasn't been done yet.
    true_arg_shardings = GetZkxOpShardings(true_args);
    false_arg_shardings = GetZkxOpShardings(false_args);
  }

  // Create zkx parameters for functions corresponding to ifOp regions using the
  // implicit captures operands. Also export the instructions within those
  // regions.
  if (failed(ctx.converter->LowerRegionAsComputation(
          &op.getTrueBranch(), &true_branch, implicit_true_operands,
          /*implicit_results=*/{}, /*ensure_single_arg=*/true,
          true_arg_shardings, ret_shardings)) ||
      failed(ctx.converter->LowerRegionAsComputation(
          &op.getFalseBranch(), &false_branch, implicit_false_operands,
          /*implicit_results=*/{}, /*ensure_single_arg=*/true,
          false_arg_shardings, ret_shardings))) {
    return failure();
  }

  // Create the Zkx pred argument.
  zkx::ZkxOp pred;
  if (failed(GetZkxOp(op.getPred(), value_map, &pred, op))) return failure();

  // Create the true branch Zkx argument.
  zkx::ZkxOp true_arg =
      CreateTupleIfMultipleOps(ctx.builder, true_args, true_arg_shardings);

  // Create the false branch Zkx argument.
  zkx::ZkxOp false_arg =
      CreateTupleIfMultipleOps(ctx.builder, false_args, false_arg_shardings);

  // Create ZKX Conditional op.
  auto ifop =
      zkx::Conditional(pred, true_arg, true_branch, false_arg, false_branch);

  // mhlo.IfOp have multiple returns, untuple all the results of ZKX's.
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = ifop;
  } else {
    BuildGetTupleElementsForTupleResults(op, ifop, ctx);
  }

  return success();
}

LogicalResult ExportZkxOp(CaseOp op, OpLoweringContext ctx) {
  llvm::DenseMap<Value, zkx::ZkxOp>& value_map = *ctx.values;
  // OperandRange operands = op.branch_operands();
  MutableArrayRef<Region> branches = op.getBranches();
  llvm::SmallVector<zkx::ZkxOp, 4> branch_operands(branches.size());
  std::vector<zkx::ZkxComputation> computations(branches.size());
  std::vector<zkx::ZkxComputation*> computations_p(branches.size());

  // mhlo.CaseOp does not have any operands or blocks arguments. The computation
  // inside the region-blocks use implicit captures of values defined above.
  // In order to create the zkx parameters for functions corresponding to
  // CaseOp regions, we need to infer the a region-block's arguments, using all
  // the values used in the region but defined above. Note that in case there
  // are zero implicit captures for a region, we use an empty tuple as the zkx
  // parameter.
  //
  // Note that the implicit values used in the regions might
  // be different and, as a result, the zkx parameters for the corresponding
  // regions could have different shapes.
  for (unsigned i = 0; i < branches.size(); ++i) {
    llvm::SetVector<Value> implicit_operand_set;
    getUsedValuesDefinedAbove(branches[i], branches[i], implicit_operand_set);
    llvm::SmallVector<Value> implicit_operands =
        implicit_operand_set.takeVector();

    llvm::SmallVector<std::optional<zkx::OpSharding>> ret_shardings =
        GetResultShardings(ctx.builder->sharding(), op->getNumResults());

    // Create the branches[i]'s Zkx argument.
    llvm::SmallVector<zkx::ZkxOp> args;
    if (failed(GetZkxOps(op, implicit_operands, ctx, args))) return failure();

    llvm::SmallVector<std::optional<zkx::OpSharding>> arg_shardings;
    if (!ret_shardings.empty()) {
      // We only add arg shardings if there are result shardings, otherwise it
      // means sharding propagation hasn't been done yet.
      arg_shardings = GetZkxOpShardings(args);
    }

    branch_operands[i] =
        CreateTupleIfMultipleOps(ctx.builder, args, arg_shardings);

    // Create zkx parameters for functions corresponding to region branches[i]
    // using the implicit captures operands. Also export the instructions within
    // that region.
    computations_p[i] = &computations[i];
    if (failed(ctx.converter->LowerRegionAsComputation(
            &branches[i], computations_p[i], implicit_operands,
            /*implicit_results=*/{}, /*ensure_single_arg=*/true, arg_shardings,
            ret_shardings)))
      return failure();
  }

  zkx::ZkxOp index;
  if (failed(GetZkxOp(op.getIndex(), value_map, &index, op))) return failure();

  zkx::ZkxOp caseop = zkx::Conditional(index, computations_p, branch_operands);

  // mhlo.CaseOp have multiple returns, untuple all the results of ZKX's.
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = caseop;
  } else {
    BuildGetTupleElementsForTupleResults(op, caseop, ctx);
  }
  return success();
}

// Specialize CompareOp export to set broadcast_dimensions argument.
LogicalResult ExportZkxOp(CompareOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxOp lhs, rhs;
  if (failed(GetZkxOp(op.getLhs(), value_map, &lhs, op))) return failure();
  if (failed(GetZkxOp(op.getRhs(), value_map, &rhs, op))) return failure();
  auto dir = Convert_comparison_direction(
      stringifyComparisonDirection(op.getComparisonDirection()));

  zkx::ZkxOp zkx_result = zkx::Compare(lhs, rhs, dir);
  value_map[op] = zkx_result;
  return success();
}

LogicalResult ExportZkxOp(ConstantOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportZkxOp(ConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxOp operand;
  if (failed(GetZkxOp(op.getOperand(), value_map, &operand, op)))
    return failure();

  value_map[op] = zkx::ConvertElementType(
      operand, zkx::mlir_utils::MlirTypeToPrimitiveTypeWithSign(
                   getElementTypeOrSelf(op.getType())));
  return success();
}

LogicalResult ExportZkxOp(IotaOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = zkx::Iota(ctx.builder, zkx::TypeToShape(op.getType()),
                            op.getIotaDimension());
  return success();
}

LogicalResult ExportZkxOp(MapOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComputation(),
                                                     &computation))) {
    return failure();
  }
  llvm::SmallVector<zkx::ZkxOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  value_map[op] = zkx::Map(ctx.builder, operands, computation,
                           Convert_dimensions(op.getDimensions()));
  return success();
}

LogicalResult ExportZkxOp(PadOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::PaddingConfig padding_config;
  auto edge_padding_low = ConvertDenseIntAttr(op.getEdgePaddingLow());
  auto edge_padding_high = ConvertDenseIntAttr(op.getEdgePaddingHigh());
  // TODO(chokobole): Do we need this? Dependency: interior_padding
  // auto interior_padding = ConvertDenseIntAttr(op.getInteriorPadding());
  for (int64_t i = 0, end = edge_padding_low.size(); i < end; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(edge_padding_low[i]);
    dims->set_edge_padding_high(edge_padding_high[i]);
    // TODO(chokobole): Do we need this? Dependency: interior_padding
    // dims->set_interior_padding(interior_padding[i]);
  }
  zkx::ZkxOp operand, padding_value;
  if (failed(GetZkxOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetZkxOp(op.getPaddingValue(), value_map, &padding_value, op)))
    return failure();

  value_map[op] = zkx::Pad(operand, padding_value, padding_config);
  return success();
}

LogicalResult ExportZkxOp(ReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getBody(), &body))) {
    return failure();
  }
  llvm::SmallVector<zkx::ZkxOp> operands, init_values;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands)) ||
      failed(GetTuple(op, op.getInitValues(), ctx, init_values))) {
    return failure();
  }
  zkx::ZkxOp result =
      zkx::Reduce(ctx.builder, operands, init_values, body,
                  Convert_broadcast_dimensions(op.getDimensions()));
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    BuildGetTupleElementsForTupleResults(op, result, ctx);
  }
  return success();
}

LogicalResult ExportZkxOp(ReduceWindowOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getBody(), &body))) {
    return failure();
  }
  llvm::SmallVector<zkx::ZkxOp> operands, init_values;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands)) ||
      failed(GetTuple(op, op.getInitValues(), ctx, init_values))) {
    return failure();
  }

  // Construct Window proto from MHLO attributes.
  zkx::Window window;
  auto window_dimensions = ConvertDenseIntAttr(op.getWindowDimensions());
  auto window_strides = ConvertDenseIntAttr(op.getWindowStrides());
  auto base_dilations = ConvertDenseIntAttr(op.getBaseDilations());
  auto window_dilations = ConvertDenseIntAttr(op.getWindowDilations());
  auto padding = zkx::ConvertNx2Attribute(op.getPadding());
  if (!padding.ok()) {
    return op.emitError() << "failed to convert padding attribute";
  }

  for (int64_t i = 0, e = window_dimensions.size(); i < e; ++i) {
    auto* dim = window.add_dimensions();
    dim->set_size(window_dimensions[i]);
    dim->set_stride(i < static_cast<int64_t>(window_strides.size())
                        ? window_strides[i]
                        : 1);
    dim->set_base_dilation(i < static_cast<int64_t>(base_dilations.size())
                               ? base_dilations[i]
                               : 1);
    dim->set_window_dilation(i < static_cast<int64_t>(window_dilations.size())
                                 ? window_dilations[i]
                                 : 1);
    if (!padding->empty()) {
      dim->set_padding_low((*padding)[i].first);
      dim->set_padding_high((*padding)[i].second);
    }
  }

  zkx::ZkxOp result =
      zkx::ReduceWindow(ctx.builder, operands, init_values, body, window);
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    BuildGetTupleElementsForTupleResults(op, result, ctx);
  }
  return success();
}

LogicalResult ExportZkxOp(ReshapeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxOp operand;
  if (failed(GetZkxOp(op.getOperand(), value_map, &operand, op)))
    return failure();

  value_map[op] =
      zkx::Reshape(operand, zkx::TypeToShape(op.getType()).dimensions());
  return success();
}

LogicalResult ExportZkxOp(ReturnOp op, OpLoweringContext ctx) {
  // Failure on purpose because `mhlo::ReturnOp` will be handled by
  // special purpose logic in `ConvertToHloModule::Lower`.
  return failure();
}

LogicalResult ExportZkxOp(ScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxComputation update_computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getUpdateComputation(),
                                                     &update_computation))) {
    return failure();
  }
  zkx::ScatterDimensionNumbers dimension_numbers =
      Convert_scatter_dimension_numbers(op.getScatterDimensionNumbers());

  llvm::SmallVector<zkx::ZkxOp> operands;
  llvm::SmallVector<zkx::ZkxOp> updates;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  if (failed(GetTuple(op, op.getUpdates(), ctx, updates))) return failure();

  zkx::ZkxOp scatter_indices;
  if (failed(GetZkxOp(op.getScatterIndices(), value_map, &scatter_indices, op)))
    return failure();

  auto scatter_op = zkx::Scatter(
      operands, scatter_indices, updates, update_computation, dimension_numbers,
      op.getIndicesAreSorted(), op.getUniqueIndices());
  if (op->getNumResults() == 1) {
    value_map[op.getResult(0)] = scatter_op;
    return success();
  }

  // mhlo.ScatterOp supports multiple returns, untuple all the results of ZKX's.
  BuildGetTupleElementsForTupleResults(op, scatter_op, ctx);

  return success();
}

// TODO(b/298671312): The semantics of zkx::SetDimensionSize have changed so
// that it always returns a dynamic shape. The old semantics are still
// available through zkx::RemoveDynamicDimension, so to avoid changing MHLO
// semantics we explicitly check for that case here. However, we should
// consider adding a RemoveDynamicDimensionOp to HLO and MHLO.
LogicalResult ExportZkxOp(SetDimensionSizeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  zkx::ZkxOp array;
  if (failed(GetZkxOp(op.getOperand(), value_map, &array, op)))
    return failure();
  auto dimension = Convertuint64_t(op.getDimension());
  auto shape_or = ctx.builder->GetShapePtr(array);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }
  zkx::ZkxOp zkx_result;
  if (auto constant =
          llvm::dyn_cast_or_null<ConstantOp>(op.getSize().getDefiningOp());
      constant != nullptr) {
    auto value = constant.getValue();
    auto values = value.getValues<IntegerAttr>();
    if ((*values.begin()).getValue().getSExtValue() ==
        shape_or.value()->dimensions(dimension)) {
      zkx_result = zkx::RemoveDynamicDimension(array, dimension);
    }
  }
  if (!zkx_result.valid()) {
    zkx::ZkxOp dynamic_size;
    if (failed(GetZkxOp(op.getSize(), value_map, &dynamic_size, op)))
      return failure();
    zkx_result = zkx::SetDimensionSize(array, dynamic_size, dimension);
  }
  value_map[result] = zkx_result;
  return success();
}

LogicalResult ExportZkxOp(SortOp op, OpLoweringContext ctx) {
  zkx::ZkxComputation comparator;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComparator(),
                                                     &comparator)))
    return failure();

  llvm::SmallVector<zkx::ZkxOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  auto sorted =
      zkx::Sort(operands, comparator, op.getDimension(), op.getIsStable());

  auto& value_map = *ctx.values;
  auto shape_or = sorted.builder()->GetShape(sorted);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }

  zkx::Shape& shape = shape_or.value();
  if (!shape.IsTuple()) {
    value_map[op.getResult(0)] = sorted;
    return success();
  }

  // MLIR's sort supports multiple returns, untuple all the results of ZKX's.
  BuildGetTupleElementsForTupleResults(op, sorted, ctx);
  return success();
}

LogicalResult ExportZkxOp(SubtractOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  zkx::ZkxOp lhs;
  if (failed(GetZkxOp(*op.getODSOperands(0).begin(), value_map, &lhs, op)))
    return failure();
  zkx::ZkxOp rhs;
  if (failed(GetZkxOp(*op.getODSOperands(1).begin(), value_map, &rhs, op)))
    return failure();
  auto zkx_result = zkx::Sub(Unwrap(lhs), Unwrap(rhs));
  value_map[result] = zkx_result;
  return success();
}

LogicalResult ExportZkxOp(WhileOp op, OpLoweringContext ctx) {
  zkx::ZkxComputation condition;
  zkx::ZkxComputation body;

  // If the results of the while op have a sharding, we use those shardings for
  // the corresponding arguments and return shardings in the body and condition.
  llvm::SmallVector<std::optional<zkx::OpSharding>> res_shardings =
      GetResultShardings(ctx.builder->sharding(), op->getNumResults());

  // mhlo.WhileOp has operands and corresponding blocks arguments, but the
  // computation inside its region-blocks can also use implicit captures of
  // values defined above.
  // In order to create the zkx parameters for functions corresponding to
  // WhileOp regions, we need to infer the implicit region-block's arguments,
  // using all the values used in the region but defined above.
  //
  // Note that the body and cond regions of WhileOp share the same block
  // arguments, so we collect the implicit values for both in a single set.
  llvm::SetVector<Value> implicit_operand_set;
  getUsedValuesDefinedAbove(op->getRegions(), implicit_operand_set);
  llvm::SmallVector<Value> implicit_operands =
      implicit_operand_set.takeVector();

  llvm::SmallVector<zkx::ZkxOp> implicit_args;
  if (failed(GetZkxOps(op, implicit_operands, ctx, implicit_args)))
    return failure();

  // We need to append the shardings of the implicit values to the result
  // shardings, since the HLO While will have those implicit values as
  // additional operands and results.
  llvm::SmallVector<std::optional<zkx::OpSharding>> implicit_shardings;
  if (!implicit_args.empty() && !res_shardings.empty()) {
    // We only add implicit arg shardings if there are result shardings,
    // otherwise it means sharding propagation hasn't been done yet.
    implicit_shardings = GetZkxOpShardings(implicit_args);

    res_shardings.append(implicit_shardings.begin(), implicit_shardings.end());
    if (std::optional<zkx::OpSharding> new_sharding =
            CreateTupleSharding(res_shardings)) {
      ctx.builder->SetSharding(*new_sharding);
    }
  }

  // The body of the While needs to return the same number of values as its
  // arguments, as they are carried over to the next iteration. Thus, we pass
  // the `implicit_operands` as `implicit_results`, to carry them over as is.
  if (failed(ctx.converter->LowerRegionAsComputation(
          &op.getBody(), &body, implicit_operands,
          /*implicit_results=*/implicit_operands,
          /*ensure_single_arg=*/true, /*arg_shardings=*/res_shardings,
          /*ret_shardings=*/res_shardings)) ||
      failed(ctx.converter->LowerRegionAsComputation(
          &op.getCond(), &condition, implicit_operands,
          /*implicit_results=*/{},
          /*ensure_single_arg=*/true, /*arg_shardings=*/res_shardings))) {
    return failure();
  }

  // In case MHLO's whileOp has multiple operands, create zkx::Tuple, using
  // those operands, to be used as sole operand of zkx::While.
  llvm::SmallVector<zkx::ZkxOp> operands;
  if (failed(GetTuple(op, op.getOperands(), ctx, operands))) return failure();
  operands.append(implicit_args.begin(), implicit_args.end());

  zkx::ZkxOp operand = operands[0];
  if (operands.size() > 1) operand = Tuple(ctx.builder, operands);

  zkx::ZkxOp whileop = zkx::While(condition, body, operand);

  auto& value_map = *ctx.values;
  auto shape_or = whileop.builder()->GetShape(whileop);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }

  zkx::Shape& shape = shape_or.value();
  if (!shape.IsTuple()) {
    value_map[op.getResult(0)] = whileop;
    return success();
  }

  // mhlo.WhileOp supports multiple returns, untuple all the results of ZKX's.
  BuildGetTupleElementsForTupleResults(
      op, whileop, ctx, /*num_implicit_results=*/implicit_args.size());

  return success();
}

LogicalResult ExportZkxOp(BitcastOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  zkx::ZkxOp operand;
  if (failed(GetZkxOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  zkx::ZkxOp bitcast = zkx::internal::ZkxBuilderFriend::BuildBitcast(
      ctx.builder, operand, zkx::TypeToShape(op.getType()));
  value_map[op] = bitcast;
  if (ctx.converter->GetOptions().propagate_bitcast_layouts_to_backend_config) {
    // Encode the source and result layout of the bitcast into the ZKX HLO
    // backend config as a protobuf. Note that this is a temporary solution
    // which will go away once ZKX:GPU stops falling back to ZKX HLO Elemental
    // IR emitters.
    zkx::HloInstructionProto* bitcast_proto =
        zkx::internal::ZkxBuilderFriend::GetInstruction(bitcast);
    zkx::HloInstructionProto* operand_proto =
        zkx::internal::ZkxBuilderFriend::GetInstruction(operand);
    zkx::LayoutProto result_layout =
        ExtractLayout(op, bitcast_proto->shape().dimensions_size(),
                      kResultLayout)
            .ToProto();
    zkx::LayoutProto source_layout =
        ExtractLayout(op, operand_proto->shape().dimensions_size(),
                      kSourceLayout)
            .ToProto();
    zkx::gpu::BitcastBackendConfig bitcast_config;
    *bitcast_config.mutable_source_layout() = source_layout;
    *bitcast_config.mutable_result_layout() = result_layout;
    *bitcast_proto->mutable_backend_config() =
        bitcast_config.SerializeAsString();
  }
  return success();
}

}  // namespace
}  // namespace mlir::mhlo

#include "zkx/hlo/translate/mhlo_to_hlo/operator_writers.inc"

namespace mlir {
namespace {

// MHLO and ZKX HLO disagree on the meaning of addition of `pred` / `i1`, so
// there has to be a special case somewhere to account for the difference. To
// get the expected behavior of an `AddOp` on `i1`, we have to use `xor`. Since
// the majority of the conversion is generated code, we just sidestep it here
// for this single case, and inline the code to emit an `xor`.
LogicalResult ExportZkxOperatorWrapped(Operation* inst, OpLoweringContext ctx) {
  auto op = dyn_cast<mhlo::AddOp>(inst);
  if (op && cast<TensorType>(op.getResult().getType())
                .getElementType()
                .isSignlessInteger(1)) {
    auto& value_map = *ctx.values;
    auto result = op.getResult();
    zkx::ZkxOp zkx_arg_0;
    if (failed(GetZkxOp(op.getLhs(), value_map, &zkx_arg_0, op)))
      return failure();
    zkx::ZkxOp zkx_arg_1;
    if (failed(GetZkxOp(op.getRhs(), value_map, &zkx_arg_1, op)))
      return failure();
    auto zkx_result = zkx::Xor(Unwrap(zkx_arg_0), Unwrap(zkx_arg_1));
    value_map[result] = zkx_result;
    return success();
  }

  return ExportZkxOperator(inst, ctx);
}

LogicalResult ConvertToHloModule::PropagateLayouts(
    const MlirToHloConversionOptions& options, Operation* inst,
    zkx::ZkxOp zkx_op) {
  // See MlirToHloConversionOptions for more about layouts.
  if (options.propagate_layouts) {
    auto* shape = zkx::internal::ZkxBuilderFriend::GetInstruction(zkx_op)
                      ->mutable_shape();
    // TODO(kramm): merge this with ConvertLayout.
    FailureOr<zkx::Shape> mlir_shape_or = ExtractZkxShape(inst);
    if (failed(mlir_shape_or)) return failure();
    *shape = mlir_shape_or->ToProto();
  }

  return success();
}

LogicalResult ConvertToHloModule::LowerCast(
    Operation* inst, const MlirToHloConversionOptions& options,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  auto cast_op = cast<tensor::CastOp>(inst);
  Value operand = cast_op.getOperand();
  auto ty = dyn_cast<ShapedType>(operand.getType());
  // If this was a cast from a static or bounded tensors, then it is a noop
  // for export to HLO and we can use the operand.
  if (!ty || !IsBoundedOrStatic(ty)) {
    inst->emitOpError()
        << "requires static or bounded operand for HLO translation";
    return failure();
  }

  zkx::ZkxOp zkx_operand;
  auto& value_map = *value_lowering;
  if (failed(GetZkxOp(operand, value_map, &zkx_operand, cast_op)))
    return failure();
  value_map[cast_op.getResult()] = zkx_operand;
  if (failed(PropagateLayouts(options, inst, zkx_operand))) {
    return failure();
  }
  return success();
}

LogicalResult ConvertToHloModule::LowerConstant(
    Operation* inst, zkx::ZkxBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    ElementsAttr const_attr) {
  if (!isa<ShapedType>(inst->getResult(0).getType())) {
    return inst->emitError(
        "expected shaped type during constant mhlo -> hlo translation");
  }

  FailureOr<zkx::Shape> shape_or = ExtractZkxShape(inst);
  if (failed(shape_or)) return failure();

  auto literal_or =
      mhlo::CreateLiteralFromAttribute(const_attr, shape_or->layout());
  if (!literal_or.ok()) return inst->emitError(literal_or.status().ToString());
  auto converted_literal_or = literal_or->Convert(shape_or->element_type());
  if (!converted_literal_or.ok())
    return inst->emitError(converted_literal_or.status().ToString());

  zkx::ZkxScopedShardingAssignment scoped_sharding(
      builder, CreateOpShardingFromAttribute(inst));
  auto constant = zkx::ConstantLiteral(builder, converted_literal_or.value());
  auto& value_map = *value_lowering;
  value_map[inst->getResult(0)] = constant;

  return success();
}

LogicalResult ConvertToHloModule::LowerReturn(
    Operation* inst, bool is_entry_function,
    llvm::ArrayRef<std::optional<zkx::OpSharding>> ret_shardings,
    llvm::ArrayRef<Value> implicit_results, zkx::ZkxBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    zkx::ZkxOp* return_value, const MlirToHloConversionOptions& options) {
  // Construct the return value for the function. If there is a single value
  // returned, then return it directly, else create a tuple and return.
  unsigned num_return_values = inst->getNumOperands() + implicit_results.size();
  std::optional<zkx::OpSharding> ret_tuple_sharding =
      CreateTupleSharding(ret_shardings);
  auto& value_map = *value_lowering;
  if ((options_.return_tuple && is_entry_function) || num_return_values != 1) {
    std::vector<zkx::ZkxOp> returns;
    returns.reserve(num_return_values);
    // NOTE: we can't use operand_range in llvm::concat.
    for (Value ret : inst->getOperands()) {
      zkx::ZkxOp& operand = returns.emplace_back();
      if (failed(GetZkxOp(ret, value_map, &operand, inst))) return failure();
    }
    for (Value ret : implicit_results) {
      zkx::ZkxOp& operand = returns.emplace_back();
      if (failed(GetZkxOp(ret, value_map, &operand, inst))) return failure();
    }
    if (is_entry_function && ret_tuple_sharding) {
      assert(implicit_results.empty() &&
             "entry functions shouldn't have implicit results");
      for (OpOperand& ret : inst->getOpOperands()) {
        unsigned index = ret.getOperandNumber();

        zkx::Shape return_shape = zkx::TypeToShape(ret.get().getType());
        absl::StatusOr<zkx::ZkxOp> reshape =
            ReshapeWithCorrectRepresentationAndSharding(
                builder, returns[index], return_shape,
                options_.layout_preference_fn, options_.shape_representation_fn,
                ret_shardings[index],
                /*fast_mem=*/false);
        if (!reshape.ok())
          return inst->emitError() << reshape.status().message();
        returns[index] = reshape.value();
      }
    }

    zkx::ZkxScopedShardingAssignment scoped_sharding(builder,
                                                     ret_tuple_sharding);
    *return_value = zkx::Tuple(builder, returns);
    return success();
  }

  if (num_return_values == 1) {
    Value ret = implicit_results.empty() ? inst->getOperand(0)
                                         : implicit_results.front();
    zkx::ZkxOp operand;
    if (failed(GetZkxOp(ret, value_map, &operand, inst))) return failure();

    if (ret_tuple_sharding) {
      auto tuple = Tuple(builder, {operand});
      builder->SetSharding(*ret_shardings[0]);
      *return_value = GetTupleElement(tuple, 0);
      builder->ClearSharding();
    } else {
      *return_value = operand;
    }
  }

  return success();
}

LogicalResult ConvertToHloModule::Lower(
    Operation* inst, bool is_entry_function,
    llvm::ArrayRef<std::optional<zkx::OpSharding>> ret_shardings,
    llvm::ArrayRef<Value> implicit_results, zkx::ZkxBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    zkx::ZkxOp* return_value) {
  // Explicitly fail for ops that are not supported for export.
  if (inst->getDialect() !=
          inst->getContext()->getLoadedDialect<mhlo::MhloDialect>() &&
      !isa<func::ConstantOp, arith::ConstantOp, func::CallOp, tensor::CastOp,
           func::ReturnOp>(inst)) {
    inst->emitOpError("unsupported op for export to ZKX");
    return failure();
  }

  *return_value = zkx::ZkxOp();

  if (succeeded(ExportZkxOperatorWrapped(
          inst,
          {value_lowering, this, builder, &stack_frame_indexes_builder_}))) {
    if (inst->getNumResults() == 1) {
      auto iter = value_lowering->find(inst->getResult(0));
      if (iter == value_lowering->end()) {
        inst->emitOpError(
            "inst has a result, but it's not found in value_lowering");
        return failure();
      }
      if (failed(PropagateLayouts(options_, inst, iter->second))) {
        return failure();
      }
    }
    // For infeed ops stemming back to InfeedDequeueTuple, respect the
    // layout attribute, and create the corresponding layout in hlo.
    // TODO(chokobole): Uncomment this. Dependency: mhlo::InfeedOp
    // if (isa<mhlo::InfeedOp>(inst)) {
    //   return LowerInfeed(inst, builder, value_lowering);
    // }
    return success();
  }

  if (auto call_op = dyn_cast<func::CallOp>(inst)) {
    return LowerFunctionCall(call_op, builder, value_lowering);
  }

  if (isa<tensor::CastOp>(inst)) {
    return LowerCast(inst, options_, value_lowering);
  }

  // TODO(chokobole): Uncomment this. Dependency: mhlo::CompositeOp
  // if (auto composite_op = dyn_cast<mhlo::CompositeOp>(inst)) {
  //   return LowerCompositeCall(inst, &module_builder_, builder,
  //   value_lowering,
  //                             return_value);
  // }

  ElementsAttr const_attr;
  if (matchPattern(inst, m_Constant(&const_attr))) {
    return LowerConstant(inst, builder, value_lowering, const_attr);
  }

  if (isa<mhlo::ReturnOp, func::ReturnOp>(inst)) {
    return LowerReturn(inst, is_entry_function, ret_shardings, implicit_results,
                       builder, value_lowering, return_value, options_);
  }

  inst->emitOpError() << "can't be translated to ZKX HLO";
  return failure();
}

LogicalResult ConvertToHloModule::LowerFunctionCall(
    func::CallOp call_op, zkx::ZkxBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  zkx::ZkxScopedShardingAssignment scoped_sharding(
      builder, CreateOpShardingFromAttribute(call_op));
  auto& value_map = *value_lowering;
  func::FuncOp callee = module_.lookupSymbol<func::FuncOp>(call_op.getCallee());
  if (failed(RunOnFunction(callee))) return failure();
  std::vector<zkx::ZkxOp> operands;
  for (auto operand : call_op.getOperands()) {
    zkx::ZkxOp zkx_operand;
    if (failed(GetZkxOp(operand, value_map, &zkx_operand, call_op)))
      return failure();
    operands.push_back(zkx_operand);
  }
  // Each call to zkx::Call would insert a copy of the computation to
  // the HLO. Thus each callsite would have a unique callee in the
  // exported HLO. HLO syntactically does not require all calls to have unique
  // callees, but eventually before lowering call graph is "flattened" to
  // make that true. This is done before lowering because buffer assignment
  // needs this invariant.

  // Remove the backend_config from the frontend attributes.
  zkx::FrontendAttributes fe_attrs = CreateZkxFrontendAttributesFromOp(call_op);
  std::string backend_config = "";
  auto fe_attrs_map = fe_attrs.mutable_map();
  if (fe_attrs_map->contains(kBackendConfig)) {
    backend_config = fe_attrs_map->at(kBackendConfig);
    fe_attrs_map->erase(kBackendConfig);
  }
  zkx::ZkxScopedFrontendAttributesAssignment assignment(builder, fe_attrs);
  zkx::ZkxOp call_result =
      zkx::Call(builder, lowered_computation_[callee], operands);
  zkx::HloInstructionProto* call_instruction =
      zkx::internal::ZkxBuilderFriend::GetInstruction(call_result);
  // `call_op` with `backend_config` can appear when round-tripping a program
  // that has already run some ZKX host communication passes.
  call_instruction->set_backend_config(backend_config);
  // Use GetTupleElement for multiple outputs
  unsigned num_results = call_op.getNumResults();
  if (num_results > 1) {
    BuildGetTupleElementsForTupleResults(call_op, call_result, builder,
                                         value_map);
  } else if (num_results == 1) {
    value_map[call_op.getResult(0)] = call_result;
  }
  return success();
}

LogicalResult ConvertToHloModule::RunOnFunction(func::FuncOp f) {
  if (lowered_computation_.count(f)) return success();
  if (!llvm::hasSingleElement(f)) {
    return f.emitError("only single block Function supported");
  }

  // Create a sub-builder if this is not the main function.
  std::unique_ptr<zkx::ZkxBuilder> builder_up;
  bool entry_function = f.getName() == kMain;
  if (!entry_function)
    builder_up = module_builder_.CreateSubBuilder(f.getName().str());
  auto& builder = entry_function ? module_builder_ : *builder_up;

  zkx::ZkxComputation computation;
  std::vector<bool> entry_args_same_across_replicas;
  llvm::SmallVector<std::optional<zkx::OpSharding>, 4> arg_shardings;
  llvm::SmallVector<std::optional<zkx::OpSharding>, 4> ret_shardings;
  llvm::SmallVector<std::optional<zkx::FrontendAttributes>, 4> arg_fe_attrs;
  if (entry_function) {
    bool any_arg_replicated = false;
    entry_args_same_across_replicas.reserve(f.getNumArguments());
    for (int64_t i = 0; i < f.getNumArguments(); ++i) {
      auto attr = f.getArgAttrOfType<BoolAttr>(i, kMhloReplication);
      entry_args_same_across_replicas.push_back(attr != nullptr &&
                                                attr.getValue());
      any_arg_replicated |= entry_args_same_across_replicas.back();
      // Pass the alias info to the builder so that it will build the alias info
      // into the resulting HloModule.
      auto buffer_donor = f.getArgAttrOfType<BoolAttr>(i, kJaxBufferDonor);
      if (buffer_donor) {
        if (options_.use_tuple_args) {
          builder.AddBufferDonor(/*param_number=*/0, /*param_index=*/{i});
        } else {
          builder.AddBufferDonor(/*param_number=*/i, /*param_index=*/{});
        }
      }
      auto aliasing_output =
          f.getArgAttrOfType<IntegerAttr>(i, kTfAliasingOutput);
      if (!aliasing_output) continue;
      zkx::ShapeIndex output_index;
      if ((options_.return_tuple && entry_function) || f.getNumResults() != 1) {
        output_index = {aliasing_output.getInt()};
      } else {
        if (aliasing_output.getInt() != 0) {
          return f.emitError(
              "Aliasing output must be 0 if only one output exists");
        }
        output_index = {};
      }
      if (options_.use_tuple_args) {
        builder.SetUpAlias(output_index, /*param_number=*/0,
                           /*param_index=*/{i});
      } else {
        builder.SetUpAlias(output_index, /*param_number=*/i,
                           /*param_index=*/{});
      }
    }
    // Do not populate this field when nothing is replicated, since empty field
    // means no replication. This avoids the need for unrelated tests to handle
    // this field.
    if (!any_arg_replicated) entry_args_same_across_replicas.clear();
    ExtractFrontendAttributesFromFunction(f, &arg_fe_attrs);
  }
  ExtractShardingsFromFunction(f, &arg_shardings, &ret_shardings);
  if (failed(LowerBasicBlockAsFunction(&f.front(), &builder, entry_function,
                                       false, entry_args_same_across_replicas,
                                       arg_shardings, ret_shardings,
                                       arg_fe_attrs, &computation))) {
    return failure();
  }
  if (auto execution_thread = f->getAttrOfType<StringAttr>(kExecutionThread)) {
    computation.mutable_proto()->mutable_computations(0)->set_execution_thread(
        execution_thread.str());
  }
  for (int i = 0; i < f.getNumArguments(); ++i) {
    if (auto pr = f.getArgAttrOfType<ArrayAttr>(i, kMhloParameterReplication)) {
      for (auto b : pr.getValue())
        for (auto& instr : *computation.mutable_proto()
                                ->mutable_computations(0)
                                ->mutable_instructions())
          if (instr.parameter_number() == i)
            instr.mutable_parameter_replication()
                ->add_replicated_at_leaf_buffers(cast<BoolAttr>(b).getValue());
    }
  }
  lowered_computation_[f] = std::move(computation);
  return success();
}

LogicalResult ConvertToHloModule::SetEntryTupleShapesAndLeafReplication(
    Block* block, const std::vector<bool>& entry_args_same_across_replicas,
    llvm::SmallVectorImpl<zkx::Shape>* arg_shapes,
    std::vector<bool>* leaf_replication) {
  arg_shapes->reserve(block->getNumArguments());
  leaf_replication->reserve(block->getNumArguments());
  for (BlockArgument& arg : block->getArguments()) {
    arg_shapes->push_back(zkx::TypeToShape(arg.getType()));
    zkx::Shape& arg_shape = arg_shapes->back();
    auto layout_preference_status =
        options_.layout_preference_fn ? options_.layout_preference_fn(arg_shape)
                                      : ZkxLayoutPreference::kNoPreference;
    if (!layout_preference_status.ok())
      return block->getParentOp()->emitError()
             << layout_preference_status.status().message();

    auto arg_shape_status = options_.shape_representation_fn
                                ? options_.shape_representation_fn(
                                      arg_shape, /*use_fast_memory=*/false,
                                      layout_preference_status.value())
                                : arg_shape;
    if (!arg_shape_status.ok())
      return block->getParentOp()->emitError()
             << arg_shape_status.status().message();

    arg_shape = std::move(arg_shape_status.value());

    if (entry_args_same_across_replicas.empty()) continue;
    for (int i = 0, e = zkx::ShapeUtil::GetLeafCount(arg_shape); i < e; ++i)
      leaf_replication->push_back(
          entry_args_same_across_replicas[arg.getArgNumber()]);
  }

  return success();
}

LogicalResult ConvertToHloModule::SetEntryTupleShardings(
    Block* block, zkx::ZkxBuilder* builder,
    llvm::ArrayRef<std::optional<zkx::OpSharding>> arg_shardings,
    llvm::SmallVectorImpl<zkx::Shape>* arg_shapes) {
  if (!arg_shardings.empty() && SomeOptionalShardingsAreSet(arg_shardings)) {
    zkx::OpSharding sharding;
    sharding.set_type(zkx::OpSharding::TUPLE);
    for (const auto& arg_sharding : llvm::enumerate(arg_shardings)) {
      if (arg_sharding.value().has_value()) {
        auto hlo_sharding = zkx::HloSharding::FromProto(*arg_sharding.value());
        if (!hlo_sharding.ok())
          return block->getParentOp()->emitError()
                 << hlo_sharding.status().message();

        auto status = RewriteLayoutWithShardedShape(
            hlo_sharding.value(), /*use_fast_memory=*/false,
            options_.layout_preference_fn, options_.shape_representation_fn,
            &(*arg_shapes)[arg_sharding.index()]);
        if (!status.ok())
          return block->getParentOp()->emitError() << status.message();

        *sharding.add_tuple_shardings() = *arg_sharding.value();
      } else {
        zkx::OpSharding fallback_sharding;
        fallback_sharding.set_type(zkx::OpSharding::REPLICATED);
        *sharding.add_tuple_shardings() = fallback_sharding;
      }
    }

    builder->SetSharding(sharding);
  }

  return success();
}

// Creates an `OpMetadata` with the debug name from the `value`'s
// `Location`.
zkx::OpMetadata GetOpNameMetadataFromLocation(Value value) {
  zkx::OpMetadata m;
  m.set_op_name(mhlo::GetDebugNameFromLocation(value.getLoc()));
  return m;
}

LogicalResult ConvertToHloModule::LowerBasicBlockAsFunction(
    Block* block, zkx::ZkxBuilder* builder, bool is_entry_function,
    bool ensure_single_arg,
    const std::vector<bool>& entry_args_same_across_replicas,
    llvm::ArrayRef<std::optional<zkx::OpSharding>> arg_shardings,
    llvm::ArrayRef<std::optional<zkx::OpSharding>> ret_shardings,
    llvm::ArrayRef<std::optional<zkx::FrontendAttributes>> fe_attrs,
    zkx::ZkxComputation* result, llvm::ArrayRef<Value> implicit_operands,
    llvm::ArrayRef<Value> implicit_results) {
  // Mapping from the Value to lowered ZkxOp.
  ValueLoweringMap lowering;

  // If using tuples as input, then there is only one input parameter that is a
  // tuple.
  if (is_entry_function && options_.use_tuple_args) {
    llvm::SmallVector<zkx::Shape, 4> arg_shapes;
    std::vector<bool> leaf_replication;
    if (failed(SetEntryTupleShapesAndLeafReplication(
            block, entry_args_same_across_replicas, &arg_shapes,
            &leaf_replication)))
      return failure();

    if (failed(
            SetEntryTupleShardings(block, builder, arg_shardings, &arg_shapes)))
      return failure();

    zkx::Shape input_shape = zkx::ShapeUtil::MakeTupleShape(arg_shapes);
    // TODO(bartchr): we are saving location information on single params
    // but not tuple params. Do the same for tuple params. To do so, either
    // fuse all the `Location`s or join the operation name strings with
    // ";" (which is essentially the same).
    auto tuple =
        zkx::Parameter(builder, 0, input_shape, kArgTuple, leaf_replication);
    builder->ClearSharding();

    for (BlockArgument& arg : block->getArguments()) {
      zkx::ZkxScopedShardingAssignment scoped_sharding(
          builder, arg_shardings.empty() ? std::nullopt
                                         : arg_shardings[arg.getArgNumber()]);
      lowering[arg] = zkx::GetTupleElement(tuple, arg.getArgNumber());
    }
  } else {
    if (ensure_single_arg) {
      // Applicable for mhlo.IfOp or mhlo.CaseOp or mhlo.WhileOp.
      llvm::SmallVector<zkx::Shape, 4> arg_shapes;

      // Lowering supports mix of block args and implicit operands
      // Block args must be added before implicit capture operands

      auto args_size = block->getNumArguments() + implicit_operands.size();

      arg_shapes.reserve(args_size);
      for (BlockArgument& arg : block->getArguments())
        arg_shapes.push_back(zkx::TypeToShape(arg.getType()));
      for (Value implicit_operand : implicit_operands)
        arg_shapes.push_back(zkx::TypeToShape(implicit_operand.getType()));

      if (args_size > 1) {
        zkx::ZkxScopedShardingAssignment scoped_sharding(
            builder, arg_shardings.empty()
                         ? std::nullopt
                         : CreateTupleSharding(arg_shardings));
        // TODO(bartchr): we are saving location information on single params
        // but not tuple params. Do the same for tuple params. To do so, either
        // fuse all the `Location`s or join the operation name strings
        // with ";" (which is essentially the same).
        auto tuple = zkx::Parameter(
            builder, 0, zkx::ShapeUtil::MakeTupleShape(arg_shapes), kArgTuple);

        for (BlockArgument& arg : block->getArguments()) {
          auto num = arg.getArgNumber();
          zkx::ZkxScopedShardingAssignment scoped_sharding(
              builder,
              arg_shardings.empty() ? std::nullopt : arg_shardings[num]);
          lowering[arg] = zkx::GetTupleElement(tuple, num);
        }
        for (auto [implicit_index, implicit_operand] :
             llvm::enumerate(implicit_operands)) {
          int64_t arg_index = block->getNumArguments() + implicit_index;
          zkx::ZkxScopedShardingAssignment scoped_sharding(
              builder,
              arg_shardings.empty() ? std::nullopt : arg_shardings[arg_index]);
          lowering[implicit_operand] = zkx::GetTupleElement(tuple, arg_index);
        }
      } else if (args_size == 1) {
        // Save the location information as a name. For example JAX will set the
        // name of the function argument. Want to preserve these for debugging.
        zkx::ZkxScopedShardingAssignment scoped_sharding(
            builder,
            arg_shardings.empty() ? std::nullopt : arg_shardings.front());
        Value arg = implicit_operands.empty() ? block->getArgument(0)
                                              : implicit_operands.front();
        zkx::ZkxScopedOpMetadataAssignment op_metadata(
            builder, GetOpNameMetadataFromLocation(arg));
        lowering[arg] = zkx::Parameter(builder, 0, arg_shapes[0], kArgPrefix);
      } else {
        // Applicable only for IfOp or CaseOp. No implicit operands implies no
        // zkx parameters. In this case, we create an empty tuple as the
        // block-parameter.
        zkx::Parameter(builder, 0, zkx::ShapeUtil::MakeTupleShape(arg_shapes),
                       kArgEmptyTuple);
      }
    } else {
      for (BlockArgument& arg : block->getArguments()) {
        auto num = arg.getArgNumber();
        zkx::Shape shape = zkx::TypeToShape(arg.getType());
        zkx::ZkxScopedShardingAssignment scoped_sharding(
            builder, arg_shardings.empty() ? std::nullopt : arg_shardings[num]);
        if (!fe_attrs.empty() && fe_attrs[num]) {
          // Populates frontend attributes for parameters only for the entry
          // functions with no tuple args.
          builder->SetFrontendAttributes(*fe_attrs[num]);
        }
        // Save the location information as a name. For example JAX will set the
        // name of the function argument of these. Want to preserve these for
        // debugging.
        zkx::ZkxScopedOpMetadataAssignment op_metadata(
            builder, GetOpNameMetadataFromLocation(arg));
        if (entry_args_same_across_replicas.empty()) {
          lowering[arg] = zkx::Parameter(builder, num, shape,
                                         absl::StrCat(kArgPrefix, num));
        } else {
          lowering[arg] = zkx::Parameter(
              builder, num, shape, absl::StrCat(kArgPrefix, num),
              std::vector<bool>(entry_args_same_across_replicas[num],
                                zkx::ShapeUtil::GetLeafCount(shape)));
        }
        builder->ClearFrontendAttributes();
      }
    }
  }

  zkx::ZkxOp return_value;
  for (auto& inst : *block)
    if (failed(Lower(&inst, is_entry_function, ret_shardings, implicit_results,
                     builder, &lowering, &return_value)))
      return failure();

  // Build the ZkxComputation and check for failures.
  auto computation_or =
      return_value.valid() ? builder->Build(return_value) : builder->Build();
  if (!computation_or.ok()) {
    block->back().emitError() << computation_or.status().message();
    return failure();
  }
  *result = std::move(computation_or.value());
  return success();
}

LogicalResult ConvertToHloModule::LowerRegionAsComputation(
    Region* region, zkx::ZkxComputation* func,
    llvm::ArrayRef<Value> implicit_operands,
    llvm::ArrayRef<Value> implicit_results, bool ensure_single_arg,
    llvm::ArrayRef<std::optional<zkx::OpSharding>> arg_shardings,
    llvm::ArrayRef<std::optional<zkx::OpSharding>> ret_shardings) {
  std::unique_ptr<zkx::ZkxBuilder> builder = module_builder_.CreateSubBuilder(
      absl::StrCat(kRegionPrefix, region_id_++));
  return LowerBasicBlockAsFunction(
      &region->front(), builder.get(),
      /*is_entry_function=*/false,
      /*ensure_single_arg*/ ensure_single_arg,
      /*entry_args_same_across_replicas=*/{}, arg_shardings, ret_shardings,
      /*fe_attrs=*/{}, func, implicit_operands, implicit_results);
}

// Runs the PrepareForExport pass on the ModuleOp.
absl::Status PrepareForExport(ModuleOp module) {
  bool hasShapeOps = false;
  module.walk([&](Operation* op) {
    hasShapeOps |= isa<shape::ShapeDialect>(op->getDialect());
    return hasShapeOps ? WalkResult::interrupt() : WalkResult::advance();
  });
  PassManager pm(module.getContext());
  pm.addNestedPass<func::FuncOp>(mhlo::createPrepareForExportPass());
  if (hasShapeOps) {
    // Experimental support for exporting dynamic MHLO programs to HLO.
    // Only bounded dynamism is planned to be supported; unbounded dynamism
    // is out of scope for now.
    pm.addNestedPass<func::FuncOp>(mhlo::createSymbolicShapeOptimizationPass());
    pm.addNestedPass<func::FuncOp>(mhlo::createShapeLegalizeToHloPass());
  }

  BaseScopedDiagnosticHandler handler(module.getContext());

  (void)pm.run(module);
  absl::Status s = handler.ConsumeStatus();
  if (!s.ok()) {
    s = absl::Status(
        s.code(),
        absl::StrCat("Unable to prepare for ZKX export: ", s.message()));
  }
  return s;
}

}  // namespace

absl::Status ConvertMlirHloToHlo(ModuleOp module, zkx::HloProto* hlo_proto,
                                 MlirToHloConversionOptions options) {
  // To support the ongoing migration of ZKX's compiler interface from MHLO
  // to StableHLO, we've inserted this fallback to provide support for backends
  // which are converting incoming ModuleOps directly to HLO.
  // zkx::MlirToZkxComputation is a better API for this purpose because it
  // supports not just MHLO, but also CHLO and StableHLO, but we will
  // temporarily support StableHLO to MHLO lowering here as well to ensure
  // a smooth migration.
  PassManager pm(module->getContext());
  pm.addPass(mhlo::createStablehloLegalizeToHloPass());
  if (failed(pm.run(module))) {
    return absl::InternalError("Unable to convert StableHLO to MHLO");
  }

  TF_RETURN_IF_ERROR(PrepareForExport(module));
  BaseScopedDiagnosticHandler diag_handler(module.getContext());
  zkx::ZkxBuilder module_builder(kMain);
  ConvertToHloModule converter(module, module_builder, options);
  if (failed(converter.Run())) return diag_handler.ConsumeStatus();
  zkx::HloModuleProto hlo_module = converter.ConsumeMainProto();
  StringRef module_name = module.getName() ? *module.getName() : kMain;
  hlo_module.set_name(module_name.str());
  // TODO(chokobole): Uncomment this. Dependency:
  // Convert_cross_program_prefetches if (auto cross_program_prefetches =
  //         module->getAttrOfType<ArrayAttr>(kMhloCrossProgramPrefetches)) {
  //   for (const auto& prefetch :
  //        Convert_cross_program_prefetches(cross_program_prefetches)) {
  //     *hlo_module.add_cross_program_prefetches() = std::move(prefetch);
  //   }
  // }
  if (auto is_dynamic = module->getAttrOfType<BoolAttr>(kMhloIsDynamic)) {
    hlo_module.set_is_dynamic(is_dynamic.getValue());
  }
  if (auto frontend_attributes =
          module->getAttrOfType<DictionaryAttr>(kMhloFrontendAttributes)) {
    CreateFrontendAttributes(frontend_attributes,
                             *hlo_module.mutable_frontend_attributes());
  }
  if (auto use_auto_spmd_partitioning =
          module->getAttrOfType<BoolAttr>(kMhloUseAutoSpmdPartitioning)) {
    hlo_module.set_use_auto_spmd_partitioning(
        use_auto_spmd_partitioning.getValue());
  }
  if (auto spmd_output_sharding =
          module->getAttrOfType<StringAttr>(kMhloSpmdOutputSharding)) {
    *hlo_module.mutable_spmd_output_sharding() =
        *zkx::ConvertSharding(spmd_output_sharding.getValue());
  }
  if (auto input_output_alias =
          module->getAttrOfType<ArrayAttr>(kMhloInputOutputAlias)) {
    if (std::optional<zkx::HloInputOutputAliasProto> input_output_alias_proto =
            zkx::ConvertInputOutputAlias(input_output_alias.getValue())) {
      *hlo_module.mutable_input_output_alias() = *input_output_alias_proto;
    }
  }
  if (auto spmd_parameters_sharding =
          module->getAttrOfType<ArrayAttr>(kMhloSpmdParametersShardings)) {
    for (const auto& sharding : spmd_parameters_sharding.getValue()) {
      *hlo_module.add_spmd_parameters_shardings() =
          *zkx::ConvertSharding(cast<StringAttr>(sharding).getValue());
    }
  }
  if (auto zkx_entry_computation_parameter_layout =
          module->getAttrOfType<ArrayAttr>(
              kMhloZkxEntryComputationParameterLayouts)) {
    auto status = mhlo::ExportModuleEntryComputationParameterLayouts(
        zkx_entry_computation_parameter_layout, hlo_module);
    if (!status.ok()) return status;
  }
  if (auto zkx_entry_computation_parameter_tiles =
          module->getAttrOfType<ArrayAttr>(
              kMhloZkxEntryComputationParameterTiles)) {
    auto status = mhlo::ExportModuleEntryComputationParameterTiles(
        zkx_entry_computation_parameter_tiles, hlo_module);
    if (!status.ok()) return status;
  }
  if (auto zkx_entry_computation_result_layout =
          module->getAttrOfType<ArrayAttr>(
              kMhloZkxEntryComputationResultLayout)) {
    auto status = mhlo::ExportModuleEntryComputationResultLayout(
        zkx_entry_computation_result_layout, hlo_module);
    if (!status.ok()) return status;
  }
  if (auto zkx_entry_computation_result_tiles =
          module->getAttrOfType<ArrayAttr>(
              kMhloZkxEntryComputationResultTiles)) {
    auto status = mhlo::ExportModuleEntryComputationResultTiles(
        zkx_entry_computation_result_tiles, hlo_module);
    if (!status.ok()) return status;
  }

  zkx::StackFrameIndexProto stack_frame_index =
      converter.BuildStackFramesIndexProto();
  hlo_module.mutable_stack_frame_index()->Swap(&stack_frame_index);
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<zkx::HloModule>> ConvertMlirHloToHloModule(
    ModuleOp module, MlirToHloConversionOptions options) {
  zkx::HloProto hlo_proto;
  TF_RETURN_IF_ERROR(ConvertMlirHloToHlo(module, &hlo_proto, options));

  // Create default config.
  const zkx::HloModuleProto& module_proto = hlo_proto.hlo_module();
  TF_ASSIGN_OR_RETURN(zkx::HloModuleConfig config,
                      zkx::HloModule::CreateModuleConfigFromProto(
                          module_proto, zkx::GetDebugOptionsFromFlags()));

  // Modify config with values stored in MLIR module attributes
  mhlo::ExportHloModuleConfig(config, module);

  return zkx::HloModule::CreateFromProto(module_proto, config);
}

absl::Status BuildHloFromMlirHlo(Block& block, zkx::ZkxBuilder& builder,
                                 llvm::ArrayRef<zkx::ZkxOp> zkx_params,
                                 std::vector<zkx::ZkxOp>& returns,
                                 MlirToHloConversionOptions options) {
  auto module = block.getParentOp()->getParentOfType<ModuleOp>();
  TF_RETURN_IF_ERROR(PrepareForExport(module));
  // No tuple support in Builder converter API.
  options.return_tuple = false;
  options.use_tuple_args = false;
  ConvertToHloModule converter(module, builder, options);

  ConvertToHloModule::ValueLoweringMap lowering;
  // zkx_params should only include non-constant parameters the block arguments
  // correspond to.
  if (zkx_params.size() != block.getArguments().size())
    return absl::InternalError(absl::StrCat(
        "zkx_params size (", zkx_params.size(), ") != block arguments size (",
        block.getArguments().size(), ")"));
  for (BlockArgument& arg : block.getArguments()) {
    auto num = arg.getArgNumber();
    lowering[arg] = zkx_params[num];
  }

  BaseScopedDiagnosticHandler diag_handler(module.getContext());
  for (auto& inst : block) {
    if (isa<mhlo::ReturnOp, func::ReturnOp>(inst)) {
      returns.resize(inst.getNumOperands());
      for (OpOperand& ret : inst.getOpOperands()) {
        unsigned index = ret.getOperandNumber();
        zkx::ZkxOp operand;
        if (failed(GetZkxOp(ret.get(), lowering, &operand, &inst)))
          return diag_handler.ConsumeStatus();
        returns[index] = operand;
      }
    } else {
      zkx::ZkxOp return_value;
      if (failed(converter.Lower(&inst, /*is_entry_function=*/true,
                                 /*ret_shardings=*/{},
                                 /*implicit_results=*/{}, &builder, &lowering,
                                 &return_value)))
        return diag_handler.ConsumeStatus();
    }
  }

  return absl::OkStatus();
}

absl::Status ConvertMlirHloToHlo(ModuleOp module, zkx::HloProto* hlo_proto,
                                 bool use_tuple_args, bool return_tuple,
                                 MlirToHloConversionOptions options) {
  options.use_tuple_args = use_tuple_args;
  options.return_tuple = return_tuple;
  return ConvertMlirHloToHlo(module, hlo_proto, options);
}

}  // namespace mlir
