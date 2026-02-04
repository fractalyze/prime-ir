/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/backends/cpu/codegen/cpu_kernel_emitter.h"

#include <functional>

#include "absl/functional/bind_front.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/Linker/Linker.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "zk_dtypes/include/field/root_of_unity.h"

#ifdef ZKX_HAS_OPENMP
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#endif

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
#include "prime_ir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"
#include "prime_ir/Dialect/Poly/IR/PolyDialect.h"
#include "prime_ir/Dialect/Poly/IR/PolyOps.h"
#include "prime_ir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "xla/tsl/platform/cpu_info.h"
#include "xla/tsl/platform/statusor.h"
#include "zk_dtypes/include/all_types.h"
#include "zkx/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "zkx/base/bits.h"
#include "zkx/base/logging.h"
#include "zkx/codegen/llvm_ir_kernel_source.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/layout_util.h"
#include "zkx/literal.h"
#include "zkx/mlir/codegen_utils.h"
#include "zkx/mlir/mlir_utils.h"
#include "zkx/primitive_util.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/shape_util.h"

namespace zkx::cpu {

namespace {

mlir::RankedTensorType MakeMlirTensorTypeWithoutLayout(
    const Shape& shape, mlir::MLIRContext* context) {
  return mlir_utils::ShapeToMlirTensorType(
      ShapeUtil::MakeShape(shape.element_type(), shape.dimensions()), context);
}

template <typename T>
absl::StatusOr<mlir::prime_ir::field::RootOfUnityAttr> GetRootOfUnityAttr(
    mlir::MLIRContext* mlir_context, int64_t fft_length) {
  using UnderlyingType = typename T::UnderlyingType;

  TF_ASSIGN_OR_RETURN(T root_of_unity,
                      zk_dtypes::GetRootOfUnity<T>(fft_length));
  return mlir::prime_ir::field::RootOfUnityAttr::get(
      mlir_context,
      /*type=*/mlir_utils::GetMlirPrimeFieldType<T>(mlir_context),
      /*root=*/
      mlir_utils::GetMlirIntegerAttr(mlir_context, root_of_unity.value()),
      /*degree=*/
      mlir_utils::GetMlirIntegerAttr(mlir_context, UnderlyingType(fft_length)));
}

absl::StatusOr<mlir::Value> CreateZeroPoint(EmitterLocOpBuilder& b,
                                            const Shape& shape) {
  switch (shape.element_type()) {
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2) \
  case enum:                                            \
    return mlir_utils::CreateMlirEcPointConstant(b, cpp_type::Zero());
    ZK_DTYPES_PUBLIC_EC_POINT_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid primitive type: %s",
          primitive_util::LowercasePrimitiveTypeName(shape.element_type())));
  }
}

mlir::Value CreateIntegerMaximum(EmitterLocOpBuilder& b, mlir::Value lhs,
                                 mlir::Value rhs, bool is_signed) {
  if (is_signed) {
    return b.create<mlir::arith::MaxSIOp>(lhs, rhs);
  } else {
    return b.create<mlir::arith::MaxUIOp>(lhs, rhs);
  }
}

mlir::Value CreateIntegerMinimum(EmitterLocOpBuilder& b, mlir::Value lhs,
                                 mlir::Value rhs, bool is_signed) {
  if (is_signed) {
    return b.create<mlir::arith::MinSIOp>(lhs, rhs);
  } else {
    return b.create<mlir::arith::MinUIOp>(lhs, rhs);
  }
}

mlir::Value CreateFieldCompare(EmitterLocOpBuilder& b,
                               ComparisonDirection direction, mlir::Value lhs,
                               mlir::Value rhs) {
  return b.create<mlir::prime_ir::field::CmpOp>(
      mlir_utils::CreateMlirArithCmpIPredicate(direction, false), lhs, rhs);
}

mlir::Value CreateFieldMaximum(EmitterLocOpBuilder& b, mlir::Value lhs,
                               mlir::Value rhs) {
  auto ge = CreateFieldCompare(b, ComparisonDirection::kGe, lhs, rhs);
  return b.create<mlir::arith::SelectOp>(ge, lhs, rhs);
}

mlir::Value CreateFieldMinimum(EmitterLocOpBuilder& b, mlir::Value lhs,
                               mlir::Value rhs) {
  auto le = CreateFieldCompare(b, ComparisonDirection::kLe, lhs, rhs);
  return b.create<mlir::arith::SelectOp>(le, lhs, rhs);
}

void LoadMlirDialects(mlir::MLIRContext* mlir_context) {
  mlir_context->loadDialect<
      // clang-format off
      mlir::affine::AffineDialect,
      mlir::arith::ArithDialect,
      mlir::bufferization::BufferizationDialect,
      mlir::cf::ControlFlowDialect,
      mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect,
      mlir::LLVM::LLVMDialect,
      mlir::math::MathDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::sparse_tensor::SparseTensorDialect,
      mlir::prime_ir::elliptic_curve::EllipticCurveDialect,
      mlir::prime_ir::field::FieldDialect,
      mlir::prime_ir::mod_arith::ModArithDialect,
      mlir::prime_ir::poly::PolyDialect,
      mlir::prime_ir::tensor_ext::TensorExtDialect
      // clang-format on
      >();
}

void OneShotBufferize(mlir::OpPassManager& pm) {
  // NOTE: One-shot bufferize does not deallocate buffers. This is done by the
  // ownership-based buffer deallocation pass.
  // https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  pm.addPass(mlir::memref::createExpandReallocPass());
  pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertBufferizationToMemRefPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void AddPasses(mlir::PassManager& pm, CpuKernelEmitter::PassFlag& flag) {
  if (VLOG_IS_ON(4)) {
    pm.getContext()->disableMultithreading();
    auto print_before = [](mlir::Pass*, mlir::Operation*) { return true; };
    auto print_after = [](mlir::Pass*, mlir::Operation*) { return true; };
    pm.enableIRPrinting(print_before, print_after, /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/true);
  }
  if (flag.enable_sparsification_and_bufferization) {
    VLOG(2) << "add pass: -sparsification-and-bufferization";
    flag.enable_expand_strided_metadata = true;
    mlir::bufferization::OneShotBufferizationOptions bufferization_options;
    bufferization_options.bufferizeFunctionBoundaries = true;
    mlir::SparsificationOptions sparsification_options;
    pm.addPass(mlir::createSparsificationAndBufferizationPass(
        bufferization_options, sparsification_options,
        /*createSparseDeallocs=*/false,
        /*enableRuntimeLibrary=*/false,
        /*enableBufferInitialization=*/false,
        /*vectorLength=*/0,
        /*enableVLAVectorization=*/false,
        /*enableSIMDIndex32=*/false,
        /*enableGPULibgen=*/false,
        /*sparseEmitStrategy=*/mlir::SparseEmitStrategy::kFunctional,
        /*parallelizationStrategy=*/
        mlir::SparseParallelizationStrategy::kDenseOuterLoop));
  }

  auto maybe_add_elementwise_to_linalg =
      [](mlir::PassManager& pm, CpuKernelEmitter::PassFlag& flag) -> void {
    if (flag.enable_elementwise_to_linalg) {
      VLOG(2) << "add pass: -convert-elementwise-to-linalg "
                 "-linalg-fuse-elementwise-ops";
      flag.enable_linalg_to_parallel_loops = true;
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::createConvertElementwiseToLinalgPass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::createLinalgElementwiseOpFusionPass());
    }
  };

  if (flag.enable_poly_to_field) {
    VLOG(2) << "add pass: -poly-to-field";
    flag.enable_field_to_arith = true;
    pm.addPass(mlir::prime_ir::poly::createPolyToField());
  }

  maybe_add_elementwise_to_linalg(pm, flag);

  if (flag.enable_elliptic_curve_to_field) {
    VLOG(2) << "add pass: -elliptic-curve-to-field";
    flag.enable_field_to_arith = true;
    pm.addPass(mlir::prime_ir::elliptic_curve::createEllipticCurveToField());
  }

  maybe_add_elementwise_to_linalg(pm, flag);

  if (flag.enable_field_to_arith) {
    VLOG(2) << "add pass: -field-to-mod-arith -mod-arith-to-arith";
    pm.addPass(mlir::prime_ir::field::createFieldToModArith());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::prime_ir::mod_arith::createModArithToArith());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (flag.enable_tensor_ext_to_tensor) {
    VLOG(2) << "add pass: -tensor-ext-to-tensor";
    pm.addPass(mlir::prime_ir::tensor_ext::createTensorExtToTensor());
  }

  if (flag.enable_tensor_to_linalg) {
    VLOG(2) << "add pass: -convert-tensor-to-linalg";
    flag.enable_linalg_to_parallel_loops = true;
    pm.addPass(mlir::createConvertTensorToLinalgPass());
  }

  if (flag.enable_one_shot_bufferize) {
    VLOG(2) << "add pass: -one-shot-bufferize";
    OneShotBufferize(pm);
  }
  if (flag.enable_buffer_results_to_out_params) {
    VLOG(2) << "add pass: -buffer-results-to-out-params=hoist-static-allocs";
    mlir::bufferization::BufferResultsToOutParamsPassOptions out_params_options;
    out_params_options.hoistStaticAllocs = true;
    pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass(
        out_params_options));
  }

  if (flag.enable_linalg_to_parallel_loops) {
    VLOG(2) << "add pass: -convert-linalg-to-parallel-loops";
    flag.enable_scf_to_cf = true;
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertLinalgToParallelLoopsPass());
  }

  if (flag.enable_lower_affine) {
    VLOG(2) << "add pass: -lower-affine";
    flag.enable_scf_to_cf = true;
    pm.addPass(mlir::createLowerAffinePass());
  }

  if (flag.enable_omp) {
#ifdef ZKX_HAS_OPENMP
    VLOG(2) << "add pass: -convert-scf-to-openmp";
    // NOTE(batzor): This pass introduces memref.alloca ops that have to be
    // lowered before the -convert-scf-to-cf pass.
    flag.enable_finalize_memref_to_llvm = true;
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
#else
    VLOG(2) << "ZKX is not built with OpenMP. Skipping OpenMP pass...";
#endif
  }

  if (flag.enable_expand_strided_metadata) {
    VLOG(2) << "add pass: -expand-strided-metadata";
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::memref::createExpandStridedMetadataPass());

    VLOG(2) << "add pass: -lower-affine";
    flag.enable_scf_to_cf = true;
    pm.addPass(mlir::createLowerAffinePass());
  }

  if (flag.enable_finalize_memref_to_llvm) {
    VLOG(2) << "add pass: -finalize-memref-to-llvm";
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  }

  if (flag.enable_scf_to_cf) {
    VLOG(2) << "add pass: -convert-scf-to-cf";
    pm.addPass(mlir::createSCFToControlFlowPass());
  }
  if (flag.enable_elliptic_curve_to_llvm) {
    VLOG(2) << "add pass: -convert-ec-to-llvm";
    pm.addPass(mlir::prime_ir::elliptic_curve::createEllipticCurveToLLVM());
  }
  if (flag.enable_ext_field_to_llvm) {
    VLOG(2) << "add pass: -convert-ext-field-to-llvm";
    pm.addPass(mlir::prime_ir::field::createExtFieldToLLVM());
  }
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

std::unique_ptr<llvm::Module> TranslateMLIRToLLVM(
    mlir::ModuleOp module, llvm::LLVMContext* llvm_context) {
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, *llvm_context);
  CHECK(llvm_module) << "Failed to translate module to LLVM IR.";
  VLOG(2) << "LLVM module after translation";
  ZKX_VLOG_LINES(2, llvm_ir::DumpToString(llvm_module.get()));
  return llvm_module;
}

std::unique_ptr<llvm::Module> CreateLLVMModule(
    mlir::MLIRContext* mlir_context, mlir::ModuleOp module,
    llvm::LLVMContext* llvm_context, CpuKernelEmitter::PassFlag& pass_flag) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
#ifdef ZKX_HAS_OPENMP
  mlir::registerOpenMPDialectTranslation(registry);
#endif
  mlir::registerAllExtensions(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  if (pass_flag.enable_tensor_to_linalg) {
    mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  }
  if (pass_flag.enable_one_shot_bufferize) {
    mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(
        registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  }
  if (pass_flag.enable_elliptic_curve_to_llvm) {
    mlir::prime_ir::elliptic_curve::registerConvertEllipticCurveToLLVMInterface(
        registry);
  }
  if (pass_flag.enable_ext_field_to_llvm) {
    mlir::prime_ir::field::registerConvertExtFieldToLLVMInterface(registry);
  }
  module->getContext()->appendDialectRegistry(registry);

  VLOG(2) << "MLIR before optimizations";
  ZKX_VLOG_LINES(2, llvm_ir::DumpToString(module));
  mlir::PassManager pm(mlir_context);
  AddPasses(pm, pass_flag);

  CHECK(mlir::succeeded(pm.run(module)));

  VLOG(2) << "MLIR after optimizations";
  ZKX_VLOG_LINES(2, llvm_ir::DumpToString(module));

  return TranslateMLIRToLLVM(module, llvm_context);
}

void Postprocess(std::unique_ptr<llvm::Module> llvm_module,
                 llvm::Module* new_llvm_module,
                 KernelApiIrBuilder::KernelPrototype& kernel_prototype,
                 std::string_view name) {
  llvm::Linker linker(*new_llvm_module);
  bool failed = linker.linkInModule(std::move(llvm_module));
  CHECK(!failed) << "Linking failed";

  llvm::Function* impl_fn_c_iface =
      new_llvm_module->getFunction(absl::StrCat("_mlir_ciface_", name));
  CHECK(impl_fn_c_iface);

  llvm::Instruction* instr =
      kernel_prototype.function->getEntryBlock().getTerminator();
  llvm::IRBuilder<> b(instr);
  llvm::SmallVector<llvm::Value*> args;
  for (const auto& ir_array : kernel_prototype.arguments) {
    args.push_back(ir_array);
  }
  args.push_back(kernel_prototype.results[0]);
  b.CreateCall(impl_fn_c_iface, args);
}

}  // namespace

CpuKernelEmitter::CpuKernelEmitter(mlir::MLIRContext* context,
                                   const HloInstruction* instr,
                                   const BufferAssignment* buffer_assignment)
    : mlir_context_(context),
      instr_(instr),
      buffer_assignment_(buffer_assignment) {}

absl::StatusOr<llvm::SmallVector<mlir::Type>>
CpuKernelEmitter::MakeFuncArguments() const {
  llvm::SmallVector<mlir::Type> args;
  args.reserve(instr_->operand_count() + 1);
  for (int64_t i = 0; i < instr_->operand_count(); ++i) {
    const HloInstruction* operand = instr_->operand(i);
    const Shape& shape = operand->shape();
    if (LayoutUtil::IsSparseArray(shape)) {
      args.push_back(mlir_utils::ShapeToMlirMemRefType(
          ShapeUtil::MakeShape(U8, {ShapeUtil::SparseArrayDataSize(shape)}),
          mlir_context_));
    } else {
      if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
        args.push_back(mlir_utils::ShapeToMlirMemRefType(
            ShapeUtil::ChangeElementType(shape, U8), mlir_context_));
      } else {
        args.push_back(mlir_utils::ShapeToMlirMemRefType(shape, mlir_context_));
      }
    }
  }
  return std::move(args);
}

absl::StatusOr<llvm::SmallVector<mlir::Type>>
CpuKernelEmitter::MakeFuncReturnTypes() const {
  if (LayoutUtil::IsSparseArray(instr_->shape())) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unhandled sparse array layout: %s", instr_->ToString()));
  }
  llvm::SmallVector<mlir::Type> ret;
  if (primitive_util::IsSubByteNonPredType(instr_->shape().element_type())) {
    ret.push_back(mlir_utils::ShapeToMlirMemRefType(
        ShapeUtil::ChangeElementType(instr_->shape(), U8), mlir_context_));
  } else {
    ret.push_back(
        mlir_utils::ShapeToMlirMemRefType(instr_->shape(), mlir_context_));
  }
  return std::move(ret);
}

mlir::Value CpuKernelEmitter::EmitCSROperand(EmitterLocOpBuilder& b,
                                             mlir::Block* entry_block,
                                             int64_t i,
                                             const Shape& shape) const {
  int64_t num_rows = shape.dimensions(0);
  int64_t num_nonzeros = shape.layout().num_nonzeros();

  int64_t offset0 = 0;
  int64_t offset1 = offset0 + (num_rows + 1) * sizeof(uint32_t);
  int64_t offset2 = offset1 + num_nonzeros * sizeof(uint32_t);

  auto offset0_op = b.create<mlir::arith::ConstantIndexOp>(offset0);
  auto offset1_op = b.create<mlir::arith::ConstantIndexOp>(offset1);
  auto offset2_op = b.create<mlir::arith::ConstantIndexOp>(offset2);

  Shape row_ptrs_shape = ShapeUtil::MakeShape(U32, {num_rows + 1});
  Shape col_indices_shape = ShapeUtil::MakeShape(U32, {num_nonzeros});
  Shape values_array_shape =
      ShapeUtil::MakeShape(shape.element_type(), {num_nonzeros});

  auto row_ptrs_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(row_ptrs_shape, mlir_context_),
      entry_block->getArgument(i), offset0_op, mlir::ValueRange{});
  auto col_indices_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(col_indices_shape, mlir_context_),
      entry_block->getArgument(i), offset1_op, mlir::ValueRange{});
  auto values_array_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(values_array_shape, mlir_context_),
      entry_block->getArgument(i), offset2_op, mlir::ValueRange{});

  auto row_ptrs = b.create<mlir::bufferization::ToTensorOp>(
      MakeMlirTensorTypeWithoutLayout(row_ptrs_shape, mlir_context_),
      row_ptrs_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto col_indices = b.create<mlir::bufferization::ToTensorOp>(
      MakeMlirTensorTypeWithoutLayout(col_indices_shape, mlir_context_),
      col_indices_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto values_array = b.create<mlir::bufferization::ToTensorOp>(
      MakeMlirTensorTypeWithoutLayout(values_array_shape, mlir_context_),
      values_array_memref,
      /*restrict=*/true,
      /*writable=*/false);

  return b.create<mlir::sparse_tensor::AssembleOp>(
      mlir_utils::ShapeToMlirTensorType(shape, mlir_context_),
      mlir::ValueRange{row_ptrs, col_indices}, values_array);
}

mlir::Value CpuKernelEmitter::EmitCSCOperand(EmitterLocOpBuilder& b,
                                             mlir::Block* entry_block,
                                             int64_t i,
                                             const Shape& shape) const {
  int64_t num_cols = shape.dimensions(0);
  int64_t num_nonzeros = shape.layout().num_nonzeros();

  int64_t offset0 = 0;
  int64_t offset1 = offset0 + (num_cols + 1) * sizeof(uint32_t);
  int64_t offset2 = offset1 + num_nonzeros * sizeof(uint32_t);

  auto offset0_op = b.create<mlir::arith::ConstantIndexOp>(offset0);
  auto offset1_op = b.create<mlir::arith::ConstantIndexOp>(offset1);
  auto offset2_op = b.create<mlir::arith::ConstantIndexOp>(offset2);

  Shape col_ptrs_shape = ShapeUtil::MakeShape(U32, {num_cols + 1});
  Shape row_indices_shape = ShapeUtil::MakeShape(U32, {num_nonzeros});
  Shape values_array_shape =
      ShapeUtil::MakeShape(shape.element_type(), {num_nonzeros});

  auto col_ptrs_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(col_ptrs_shape, mlir_context_),
      entry_block->getArgument(i), offset0_op, mlir::ValueRange{});
  auto row_indices_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(row_indices_shape, mlir_context_),
      entry_block->getArgument(i), offset1_op, mlir::ValueRange{});
  auto values_array_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(values_array_shape, mlir_context_),
      entry_block->getArgument(i), offset2_op, mlir::ValueRange{});

  auto col_ptrs = b.create<mlir::bufferization::ToTensorOp>(
      MakeMlirTensorTypeWithoutLayout(col_ptrs_shape, mlir_context_),
      col_ptrs_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto row_indices = b.create<mlir::bufferization::ToTensorOp>(
      MakeMlirTensorTypeWithoutLayout(row_indices_shape, mlir_context_),
      row_indices_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto values_array = b.create<mlir::bufferization::ToTensorOp>(
      MakeMlirTensorTypeWithoutLayout(values_array_shape, mlir_context_),
      values_array_memref,
      /*restrict=*/true,
      /*writable=*/false);

  return b.create<mlir::sparse_tensor::AssembleOp>(
      mlir_utils::ShapeToMlirTensorType(shape, mlir_context_),
      mlir::ValueRange{col_ptrs, row_indices}, values_array);
}

mlir::Value CpuKernelEmitter::EmitCOOOperand(EmitterLocOpBuilder& b,
                                             mlir::Block* entry_block,
                                             int64_t i,
                                             const Shape& shape) const {
  int64_t num_nonzeros = shape.layout().num_nonzeros();
  CHECK_LE(num_nonzeros, std::numeric_limits<int32_t>::max());

  auto positions_type = mlir::RankedTensorType::get({2}, b.getI32Type());
  auto positions_attr = mlir::DenseElementsAttr::get(
      positions_type,
      llvm::ArrayRef<int32_t>{0, static_cast<int32_t>(num_nonzeros)});
  auto positions =
      b.create<mlir::arith::ConstantOp>(positions_type, positions_attr);

  int64_t offset0 = 0;
  int64_t offset1 = offset0 + num_nonzeros * sizeof(uint32_t) * 2;

  auto offset0_op = b.create<mlir::arith::ConstantIndexOp>(offset0);
  auto offset1_op = b.create<mlir::arith::ConstantIndexOp>(offset1);

  Shape indices_shape = ShapeUtil::MakeShape(U32, {num_nonzeros, 2});
  Shape values_array_shape =
      ShapeUtil::MakeShape(shape.element_type(), {num_nonzeros});

  auto indices_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(indices_shape, mlir_context_),
      entry_block->getArgument(i), offset0_op, mlir::ValueRange{});
  auto values_array_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(values_array_shape, mlir_context_),
      entry_block->getArgument(i), offset1_op, mlir::ValueRange{});

  auto indices = b.create<mlir::bufferization::ToTensorOp>(
      MakeMlirTensorTypeWithoutLayout(indices_shape, mlir_context_),
      indices_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto values_array = b.create<mlir::bufferization::ToTensorOp>(
      MakeMlirTensorTypeWithoutLayout(values_array_shape, mlir_context_),
      values_array_memref,
      /*restrict=*/true,
      /*writable=*/false);

  return b.create<mlir::sparse_tensor::AssembleOp>(
      mlir_utils::ShapeToMlirTensorType(shape, mlir_context_),
      mlir::ValueRange{positions, indices}, values_array);
}

absl::StatusOr<absl::flat_hash_map<const HloInstruction*, mlir::Value>>
CpuKernelEmitter::EmitOperands(EmitterLocOpBuilder& b,
                               mlir::Block* entry_block) const {
  absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
  for (int64_t i = 0; i < instr_->operand_count(); ++i) {
    const HloInstruction* operand = instr_->operand(i);
    const Shape& shape = operand->shape();
    if (LayoutUtil::IsSparseArray(shape)) {
      pass_flag_.enable_sparsification_and_bufferization = true;

      if (LayoutUtil::IsCSRArray(shape)) {
        values[operand] = EmitCSROperand(b, entry_block, i, shape);
      } else if (LayoutUtil::IsCSCArray(shape)) {
        values[operand] = EmitCSCOperand(b, entry_block, i, shape);
      } else if (LayoutUtil::IsCOOArray(shape)) {
        values[operand] = EmitCOOOperand(b, entry_block, i, shape);
      } else {
        return absl::UnimplementedError(absl::StrFormat(
            "Unhandled sparse array layout: %s", operand->ToString()));
      }
    } else {
      if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
        auto load =
            b.create<mlir::memref::LoadOp>(entry_block->getArgument(i), {});
        if (ShapeUtil::IsScalar(shape)) {
          auto bit_width = primitive_util::BitWidth(shape.element_type());
          if (bit_width == 2 || bit_width == 4) {
            auto shr = b.create<mlir::arith::ShRUIOp>(
                load, b.create<mlir::arith::ConstantOp>(
                          b.getIntegerAttr(b.getI8Type(), 8 - bit_width)));
            values[operand] = b.create<mlir::arith::TruncIOp>(
                mlir_utils::PrimitiveTypeToMlirType(shape.element_type(),
                                                    b.getContext()),
                shr);
            continue;
          } else {
            return absl::InternalError(absl::StrFormat(
                "Unhandled sub byte non pred type: %s", operand->ToString()));
          }
        } else {
          return absl::UnimplementedError(absl::StrFormat(
              "tensor input with sub byte non pred type is not supported: %s",
              operand->ToString()));
        }
      }

      pass_flag_.enable_one_shot_bufferize = true;

      values[operand] = b.create<mlir::bufferization::ToTensorOp>(
          MakeMlirTensorTypeWithoutLayout(shape, mlir_context_),
          entry_block->getArgument(i),
          /*restrict=*/true,
          /*writable=*/false);
    }
  }
  return std::move(values);
}

absl::Status CpuKernelEmitter::EmitEpilog(EmitterLocOpBuilder& b,
                                          mlir::Block* entry_block,
                                          mlir::MemRefType ret_type,
                                          mlir::Value result) const {
  const Shape& shape = instr_->shape();
  mlir::Value ret_value;
  if (LayoutUtil::IsSparseArray(shape)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unhandled sparse array layout: %s", instr_->ToString()));
  }

  if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
    if (ShapeUtil::IsScalar(shape)) {
      auto bit_width = primitive_util::BitWidth(shape.element_type());
      if (bit_width == 2 || bit_width == 4) {
        auto ext = b.create<mlir::arith::ExtUIOp>(b.getI8Type(), result);
        auto shl = b.create<mlir::arith::ShLIOp>(
            ext, b.create<mlir::arith::ConstantOp>(
                     b.getIntegerAttr(b.getI8Type(), 8 - bit_width)));
        ret_value = b.create<mlir::memref::AllocOp>(ret_type);
        b.create<mlir::memref::StoreOp>(shl, ret_value, {});
      } else {
        return absl::InternalError(absl::StrFormat(
            "Unhandled sub byte non pred type: %s", instr_->ToString()));
      }
    } else {
      return absl::UnimplementedError(absl::StrFormat(
          "tensor output with sub byte non pred type is not supported: %s",
          instr_->ToString()));
    }
  } else {
    pass_flag_.enable_one_shot_bufferize = true;
    ret_value = b.create<mlir::bufferization::ToBufferOp>(ret_type, result);
  }

  b.create<mlir::func::ReturnOp>(mlir::ValueRange{ret_value});
  return absl::OkStatus();
}

absl::StatusOr<KernelDefinition> CpuKernelEmitter::EmitKernelDefinition() {
  VLOG(2) << "Emit host kernel: " << instr_->name();

  auto llvm_context = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return absl::InternalError("HloModule is null");
  }

  LoadMlirDialects(mlir_context_);

  auto loc =
      mlir::NameLoc::get(mlir::StringAttr::get(mlir_context_, instr_->name()));
  EmitterLocOpBuilder b(loc, mlir_context_);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      llvm_ir::CreateMlirModuleOp(std::move(loc));
  b.setInsertionPointToEnd(mlir_module->getBody());

  TF_ASSIGN_OR_RETURN(llvm::SmallVector<mlir::Type> fn_arg_types,
                      MakeFuncArguments());
  llvm::SmallVector<mlir::Type> fn_ret_types;
  mlir::func::FuncOp fn;
  TF_ASSIGN_OR_RETURN(fn_ret_types, MakeFuncReturnTypes());
  fn = b.create<mlir::func::FuncOp>(
      instr_->name(), b.getFunctionType(fn_arg_types, fn_ret_types));
  for (int64_t i = 0; i < fn_arg_types.size(); ++i) {
    fn.setArgAttr(i,
                  mlir::bufferization::BufferizationDialect::kWritableAttrName,
                  b.getBoolAttr(false));
  }
  fn->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
              mlir::UnitAttr::get(mlir_context_));

  mlir::Block* entry_block = fn.addEntryBlock();
  b.setInsertionPointToEnd(entry_block);

  absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
  TF_ASSIGN_OR_RETURN(values, EmitOperands(b, entry_block));

  TF_ASSIGN_OR_RETURN(mlir::Value res, EmitOp(instr_, b, values));

  TF_RETURN_IF_ERROR(EmitEpilog(
      b, entry_block, mlir::cast<mlir::MemRefType>(fn_ret_types[0]), res));

  std::unique_ptr<llvm::Module> llvm_module = CreateLLVMModule(
      mlir_context_, mlir_module.get(), llvm_context.get(), pass_flag_);

  KernelApiIrBuilder kernel_api_ir_builder(
      *llvm_context,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));

  std::unique_ptr<llvm::Module> new_llvm_module =
      KernelApiIrBuilder::CreateModule(
          absl::StrCat(instr_->name(), "_kernel_module"), *llvm_context);

  TF_ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype kernel_prototype,
      kernel_api_ir_builder.EmitKernelPrototype(*new_llvm_module, instr_,
                                                buffer_assignment_, "_kernel"));

  Postprocess(std::move(llvm_module), new_llvm_module.get(), kernel_prototype,
              instr_->name());

  auto source = std::make_unique<LlvmIrKernelSource>(
      std::move(llvm_context), std::move(new_llvm_module));

  KernelSpec spec(kernel_prototype.function->getName(), se::ThreadDim(),
                  std::move(kernel_prototype.buffer_uses));

  return KernelDefinition(std::move(spec), std::move(source));
}

absl::StatusOr<std::unique_ptr<KernelSource>> CpuKernelEmitter::EmitComparator(
    const HloComputation* comparator) {
  VLOG(2) << "Emit comparator: " << comparator->name();

  auto llvm_context = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return absl::InternalError("HloModule is null");
  }

  LoadMlirDialects(mlir_context_);

  auto loc = mlir::NameLoc::get(
      mlir::StringAttr::get(mlir_context_, comparator->name()));
  EmitterLocOpBuilder b(loc, mlir_context_);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      llvm_ir::CreateMlirModuleOp(std::move(loc));
  b.setInsertionPointToEnd(mlir_module->getBody());

  auto ptr_type = mlir::LLVM::LLVMPointerType::get(b.getContext());
  auto fn = b.create<mlir::func::FuncOp>(
      comparator->name(), b.getFunctionType(ptr_type, b.getI1Type()));

  mlir::Block* entry_block = fn.addEntryBlock();
  b.setInsertionPointToEnd(entry_block);

  auto base_ptr = entry_block->getArgument(0);
  absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
  for (int64_t i = 0; i < comparator->num_parameters(); ++i) {
    const HloInstruction* instr = comparator->parameter_instruction(i);

    mlir::Type param_type;
    if (ShapeUtil::IsScalar(instr->shape())) {
      param_type = mlir_utils::PrimitiveTypeToMlirType(
          instr->shape().element_type(), b.getContext());
    } else {
      param_type =
          MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext());
    }

    llvm::SmallVector<mlir::Value> indices = {
        {b.create<mlir::arith::ConstantIntOp>(b.getI64Type(), i)}};
    mlir::Value ptr = b.create<mlir::LLVM::GEPOp>(
        ptr_type, ptr_type,
        /*basePtr=*/base_ptr, indices, mlir::LLVM::GEPNoWrapFlags::inbounds);

    mlir::Value value_ptr = b.create<mlir::LLVM::LoadOp>(ptr_type, ptr);

    values[comparator->parameter_instruction(i)] =
        b.create<mlir::LLVM::LoadOp>(param_type, value_ptr);
  }

  comparator->ForEachInstructionPostOrder(absl::bind_front(
      &CpuKernelEmitter::EmitOpInToApply, this, std::ref(b), std::ref(values)));

  b.create<mlir::func::ReturnOp>(
      mlir::ValueRange{values[comparator->root_instruction()]});

  std::unique_ptr<llvm::Module> llvm_module = CreateLLVMModule(
      mlir_context_, mlir_module.get(), llvm_context.get(), pass_flag_);

  return std::make_unique<LlvmIrKernelSource>(std::move(llvm_context),
                                              std::move(llvm_module));
}

namespace {

template <typename T>
mlir::DenseIntElementsAttr CreateDenseIntElementsAttr(EmitterLocOpBuilder& b,
                                                      const Shape& shape,
                                                      const Literal& literal) {
  mlir::RankedTensorType type =
      MakeMlirTensorTypeWithoutLayout(shape, b.getContext());
  absl::Span<const T> data = literal.data<T>();

  if constexpr (is_big_int_v<T>) {
    type = type.clone(mlir::IntegerType::get(b.getContext(), T::kBitWidth));
    llvm::SmallVector<llvm::APInt> mlir_data;
    mlir_data.reserve(data.size());
    for (const T& value : data) {
      mlir_data.push_back(mlir_utils::ConvertUnderlyingValueToAPInt(value));
    }
    return mlir::DenseIntElementsAttr::get(type, mlir_data);
  } else if constexpr (is_specialized_integral_v<T>) {
    return mlir::DenseIntElementsAttr::get(
        type, llvm::ArrayRef<T>(data.data(), data.size()));
  } else {
    static_assert(zk_dtypes::IsPrimeField<T>);

    using UnderlyingType = typename T::UnderlyingType;

    type = type.clone(mlir::IntegerType::get(b.getContext(), T::kStorageBits));

    if constexpr (T::kStorageBits <= 64) {
      return mlir::DenseIntElementsAttr::get(
          type, llvm::ArrayRef<UnderlyingType>(
                    reinterpret_cast<const UnderlyingType*>(data.data()),
                    data.size()));
    } else {
      llvm::SmallVector<llvm::APInt> mlir_data;
      mlir_data.reserve(data.size());
      for (const T& value : data) {
        mlir_data.push_back(
            mlir_utils::ConvertUnderlyingValueToAPInt(value.value()));
      }
      return mlir::DenseIntElementsAttr::get(type, mlir_data);
    }
  }
}

}  // namespace

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitIntegerConstantOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b) {
  Shape shape = ShapeUtil::MakeShape(instr->literal().shape().element_type(),
                                     instr->literal().shape().dimensions());
  mlir::DenseIntElementsAttr attr;
  PrimitiveType element_type = shape.element_type();
  if (element_type == PRED) {
    attr = CreateDenseIntElementsAttr<bool>(b, shape, instr->literal());
  } else if (primitive_util::IsIntegralType(element_type)) {
    attr = primitive_util::IntegralTypeSwitch<mlir::DenseIntElementsAttr>(
        [&](auto type_constant) {
          using NativeT = primitive_util::NativeTypeOf<type_constant>;
          return CreateDenseIntElementsAttr<NativeT>(b, shape,
                                                     instr->literal());
        },
        element_type);
  } else {
    return absl::UnimplementedError(absl::StrFormat(
        "Unhandled integer constant type: %s",
        primitive_util::LowercasePrimitiveTypeName(element_type)));
  }
  return b.create<mlir::arith::ConstantOp>(
      MakeMlirTensorTypeWithoutLayout(shape, b.getContext()), attr);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFieldConstantOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b) {
  Shape shape = ShapeUtil::MakeShape(instr->literal().shape().element_type(),
                                     instr->literal().shape().dimensions());
  mlir::DenseIntElementsAttr attr;
  PrimitiveType element_type = shape.element_type();
  // TODO(chokobole): Currently, PrimeIR's constant representation is limited
  // to Prime Fields. We need to extend the IR and the attribute system to
  // support tensors of Extension Field constants before they can be emitted
  // here.
  switch (element_type) {
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2)                      \
  case enum:                                                                 \
    attr = CreateDenseIntElementsAttr<cpp_type>(b, shape, instr->literal()); \
    break;
    ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled field constant type: %s",
          primitive_util::LowercasePrimitiveTypeName(element_type)));
  }
  return b.create<mlir::prime_ir::field::ConstantOp>(
      MakeMlirTensorTypeWithoutLayout(shape, b.getContext()), attr);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitConstantOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b) {
  Shape shape = instr->shape();
  PrimitiveType type = instr->shape().element_type();
  if (type == PRED || ShapeUtil::ElementIsIntegral(shape)) {
    return EmitIntegerConstantOp(instr, b);
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldConstantOp(instr, b);
  }
  // TODO(chokobole): Support creating ec point constants.

  return absl::UnimplementedError(
      absl::StrFormat("Unhandled primitive type: %s",
                      primitive_util::LowercasePrimitiveTypeName(type)));
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitPredUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  switch (instr->opcode()) {
    case HloOpcode::kNot:
      return EmitIntegerUnaryOp(instr, b, value, false);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary pred op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitIntegerUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value,
    bool is_signed) {
  switch (instr->opcode()) {
    case HloOpcode::kAbs:
      return b.create<mlir::math::AbsIOp>(value);
    case HloOpcode::kBitcastConvert: {
      mlir::Type ret_type =
          MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext());
      return b.create<mlir::arith::BitcastOp>(ret_type, value);
    }
    case HloOpcode::kClz:
      return b.create<mlir::math::CountLeadingZerosOp>(value);
    case HloOpcode::kConvert: {
      mlir::Type ret_type =
          MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext());
      return mlir_utils::ConvertInteger(b, {ret_type}, value.getType(),
                                        ret_type, {value}, is_signed);
    }
    case HloOpcode::kNegate:
      return b.create<mlir::arith::SubIOp>(
          mlir_utils::GetConstantOrSplat(
              b, value.getType(),
              b.getZeroAttr(getElementTypeOrSelf(value.getType()))),
          value);
    case HloOpcode::kNot:
      return b.create<mlir::arith::XOrIOp>(
          value,
          mlir_utils::GetConstantOrSplat(
              b, value.getType(),
              b.getIntegerAttr(getElementTypeOrSelf(value.getType()), -1)));
    case HloOpcode::kPopulationCount:
      return b.create<mlir::math::CtPopOp>(value);
    case HloOpcode::kSign:
      return mlir_utils::SignInteger(b, value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary integer op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFieldUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  switch (instr->opcode()) {
    case HloOpcode::kConvert: {
      mlir::Type ret_type =
          MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext());
      return mlir_utils::ConvertField(b, {ret_type}, value.getType(), ret_type,
                                      {value});
    }
    case HloOpcode::kInverse:
      return b.create<mlir::prime_ir::field::InverseOp>(value);
    case HloOpcode::kNegate:
      return b.create<mlir::prime_ir::field::NegateOp>(value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitEcPointUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  switch (instr->opcode()) {
    case HloOpcode::kConvert: {
      mlir::Type ret_type =
          MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext());
      return mlir_utils::ConvertEcPoint(b, {ret_type}, value.getType(),
                                        ret_type, {value});
    }
    case HloOpcode::kNegate:
      return b.create<mlir::prime_ir::elliptic_curve::NegateOp>(value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary ec point op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  Shape shape = instr->operand(0)->shape();
  PrimitiveType operand_type = shape.element_type();
  if (operand_type == PRED) {
    return EmitPredUnaryOp(instr, b, value);
  } else if (ShapeUtil::ElementIsIntegral(shape)) {
    if (instr->opcode() == HloOpcode::kConvert) {
      Shape output_shape = instr->shape();
      if (ShapeUtil::ElementIsIntegral(output_shape)) {
        return EmitIntegerUnaryOp(
            instr, b, value,
            primitive_util::IsSignedIntegralType(operand_type));
      } else if (ShapeUtil::ElementIsField(output_shape)) {
        return EmitFieldUnaryOp(instr, b, value);
      } else {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported output type for integer convert operation: %s",
            output_shape.ToString()));
      }
    }
    return EmitIntegerUnaryOp(
        instr, b, value, primitive_util::IsSignedIntegralType(operand_type));
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldUnaryOp(instr, b, value);
  } else if (ShapeUtil::ElementIsEcPoint(shape)) {
    return EmitEcPointUnaryOp(instr, b, value);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unhandled primitive type: %s",
      primitive_util::LowercasePrimitiveTypeName(operand_type)));
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitPredBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  switch (instr->opcode()) {
    case HloOpcode::kAnd:
    case HloOpcode::kCompare:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      return EmitIntegerBinaryOp(instr, b, lhs_value, rhs_value, false);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary pred op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitIntegerBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value, bool is_signed) {
  switch (instr->opcode()) {
    // TODO(jingyue): add the "nsw" attribute for signed types.
    case HloOpcode::kAdd:
      return b.create<mlir::arith::AddIOp>(lhs_value, rhs_value);
    case HloOpcode::kAnd:
      return b.create<mlir::arith::AndIOp>(lhs_value, rhs_value);
    case HloOpcode::kCompare:
      return b.create<mlir::arith::CmpIOp>(
          mlir_utils::CreateMlirArithCmpIPredicate(
              instr->comparison_direction(), is_signed),
          lhs_value, rhs_value);
    case HloOpcode::kDivide:
      return mlir_utils::DivideInteger(b, lhs_value, rhs_value, is_signed);
    case HloOpcode::kMaximum:
      return CreateIntegerMaximum(b, lhs_value, rhs_value, is_signed);
    case HloOpcode::kMinimum:
      return CreateIntegerMinimum(b, lhs_value, rhs_value, is_signed);
    case HloOpcode::kMultiply:
      return b.create<mlir::arith::MulIOp>(lhs_value, rhs_value);
    case HloOpcode::kOr:
      return b.create<mlir::arith::OrIOp>(lhs_value, rhs_value);
    case HloOpcode::kPower: {
      pass_flag_.enable_linalg_to_parallel_loops = true;

      auto output = b.create<mlir::tensor::EmptyOp>(
          MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext()),
          mlir::ValueRange{});
      return b
          .create<mlir::linalg::MapOp>(
              mlir::ValueRange{lhs_value, rhs_value}, output,
              [is_signed](mlir::OpBuilder& nested_b, mlir::Location loc,
                          mlir::ValueRange loop_vars) {
                mlir::ImplicitLocOpBuilder b(loc, nested_b);
                mlir::Value elementwise_result = mlir_utils::PowerInteger(
                    b, loop_vars[0], loop_vars[1], is_signed);
                b.create<mlir::linalg::YieldOp>(elementwise_result);
              })
          ->getResult(0);
    }
    case HloOpcode::kRemainder:
      return mlir_utils::RemainderInteger(b, lhs_value, rhs_value, is_signed);
    case HloOpcode::kShiftLeft:
      return mlir_utils::ShiftLeftInteger(b, lhs_value, rhs_value);
    case HloOpcode::kShiftRightArithmetic:
      return mlir_utils::ShiftRightArithmeticInteger(b, lhs_value, rhs_value);
    case HloOpcode::kShiftRightLogical:
      return mlir_utils::ShiftRightLogicalInteger(b, lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b.create<mlir::arith::SubIOp>(lhs_value, rhs_value);
    case HloOpcode::kXor:
      return b.create<mlir::arith::XOrIOp>(lhs_value, rhs_value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary integer op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFieldBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  switch (instr->opcode()) {
    case HloOpcode::kAdd:
      return b.create<mlir::prime_ir::field::AddOp>(lhs_value, rhs_value);
    case HloOpcode::kCompare:
      return CreateFieldCompare(b, instr->comparison_direction(), lhs_value,
                                rhs_value);
    case HloOpcode::kDivide: {
      auto inv = b.create<mlir::prime_ir::field::InverseOp>(rhs_value);
      return b.create<mlir::prime_ir::field::MulOp>(lhs_value, inv);
    }
    case HloOpcode::kMaximum:
      return CreateFieldMaximum(b, lhs_value, rhs_value);
    case HloOpcode::kMinimum:
      return CreateFieldMinimum(b, lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return b.create<mlir::prime_ir::field::MulOp>(lhs_value, rhs_value);
    case HloOpcode::kPower: {
      const PrimitiveType exponent_type =
          instr->operand(1)->shape().element_type();
      if (!primitive_util::IsUnsignedIntegralType(exponent_type)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "The exponent for a power operation on a field must be an "
            "unsigned integer, but got %s",
            primitive_util::LowercasePrimitiveTypeName(exponent_type)));
      }
      return b.create<mlir::prime_ir::field::PowUIOp>(lhs_value.getType(),
                                                      lhs_value, rhs_value);
    }
    case HloOpcode::kSubtract:
      return b.create<mlir::prime_ir::field::SubOp>(lhs_value, rhs_value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitEcPointBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  const Shape& shape = instr->shape();
  mlir::Type ret_type = MakeMlirTensorTypeWithoutLayout(shape, b.getContext());

  switch (instr->opcode()) {
    case HloOpcode::kAdd:
      return b.create<mlir::prime_ir::elliptic_curve::AddOp>(
          ret_type, lhs_value, rhs_value);
    case HloOpcode::kCompare:
      if (instr->comparison_direction() != ComparisonDirection::kEq &&
          instr->comparison_direction() != ComparisonDirection::kNe) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported comparison direction for EC points: %s",
            ComparisonDirectionToString(instr->comparison_direction())));
      }
      return b.create<mlir::prime_ir::elliptic_curve::CmpOp>(
          mlir_utils::CreateMlirArithCmpIPredicate(
              instr->comparison_direction(), false),
          lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return b.create<mlir::prime_ir::elliptic_curve::ScalarMulOp>(
          ret_type, lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b.create<mlir::prime_ir::elliptic_curve::SubOp>(
          ret_type, lhs_value, rhs_value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary ec point op %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  Shape shape = instr->operand(0)->shape();
  PrimitiveType operand_type = shape.element_type();
  if (operand_type == PRED) {
    return EmitPredBinaryOp(instr, b, lhs_value, rhs_value);
  } else if (ShapeUtil::ElementIsIntegral(shape)) {
    return EmitIntegerBinaryOp(
        instr, b, lhs_value, rhs_value,
        primitive_util::IsSignedIntegralType(operand_type));
  } else if (ShapeUtil::ElementIsEcPoint(shape) ||
             ShapeUtil::ElementIsEcPoint(instr->operand(1)->shape())) {
    return EmitEcPointBinaryOp(instr, b, lhs_value, rhs_value);
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldBinaryOp(instr, b, lhs_value, rhs_value);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unhandled primitive type: %s",
      primitive_util::LowercasePrimitiveTypeName(operand_type)));
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitPredTernaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value1,
    mlir::Value value2, mlir::Value value3) {
  switch (instr->opcode()) {
    case HloOpcode::kSelect:
      return EmitIntegerTernaryOp(instr, b, value1, value2, value3, false);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled ternary pred op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitIntegerTernaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value1,
    mlir::Value value2, mlir::Value value3, bool is_signed) {
  switch (instr->opcode()) {
    case HloOpcode::kClamp:
      return CreateIntegerMinimum(
          b,
          CreateIntegerMaximum(b, /*min=*/value1, /*operand=*/value2,
                               is_signed),
          /*max=*/value3, is_signed);
    case HloOpcode::kSelect:
      return b.create<mlir::arith::SelectOp>(value1, value2, value3);
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled ternary integer op: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFieldTernaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value1,
    mlir::Value value2, mlir::Value value3) {
  switch (instr->opcode()) {
    case HloOpcode::kClamp:
      return CreateFieldMinimum(
          b, CreateFieldMaximum(b, /*min=*/value1, /*operand=*/value2),
          /*max=*/value3);
    case HloOpcode::kSelect:
      return b.create<mlir::arith::SelectOp>(value1, value2, value3);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled ternary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitEcPointTernaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value1,
    mlir::Value value2, mlir::Value value3) {
  switch (instr->opcode()) {
    case HloOpcode::kSelect:
      return b.create<mlir::arith::SelectOp>(value1, value2, value3);
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled ternary ec point op: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitTernaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value1,
    mlir::Value value2, mlir::Value value3) {
  Shape shape =
      instr->operand(instr->opcode() == HloOpcode::kSelect ? 1 : 0)->shape();
  PrimitiveType operand_type = shape.element_type();
  if (operand_type == PRED) {
    return EmitPredTernaryOp(instr, b, value1, value2, value3);
  } else if (ShapeUtil::ElementIsIntegral(shape)) {
    return EmitIntegerTernaryOp(
        instr, b, value1, value2, value3,
        primitive_util::IsSignedIntegralType(operand_type));
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldTernaryOp(instr, b, value1, value2, value3);
  } else if (ShapeUtil::ElementIsEcPoint(shape)) {
    return EmitEcPointTernaryOp(instr, b, value1, value2, value3);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unhandled primitive type: %s",
      primitive_util::LowercasePrimitiveTypeName(operand_type)));
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFftOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value,
    mlir::Value twiddle_factor) {
  pass_flag_.enable_poly_to_field = true;
  pass_flag_.enable_lower_affine = true;
  if (instr->fft_type() == FftType::IFFT) {
    pass_flag_.enable_linalg_to_parallel_loops = true;
  }

  PrimitiveType operand_type = instr->operand(0)->shape().element_type();
  absl::StatusOr<mlir::prime_ir::field::RootOfUnityAttr> root;
  mlir::prime_ir::field::RootOfUnityAttr root_attr;
  switch (operand_type) {
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2)                        \
  case enum:                                                                   \
    root =                                                                     \
        GetRootOfUnityAttr<cpp_type>(value.getContext(), instr->fft_length()); \
    break;
    ZK_DTYPES_PUBLIC_PRIME_FIELD_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid primitive type: %s",
          primitive_util::LowercasePrimitiveTypeName(operand_type)));
  }
  if (!root.ok()) return root.status();
  root_attr = root.value();
  pass_flag_.enable_tensor_ext_to_tensor = true;

  auto alloc_tensor = b.create<mlir::bufferization::AllocTensorOp>(
      MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext()),
      mlir::ValueRange{});
  switch (instr->fft_type()) {
    case FftType::FFT: {
      return b.create<mlir::prime_ir::poly::NTTOp>(
          value, alloc_tensor, twiddle_factor, root_attr,
          /*tileX=*/nullptr, /*gridSizeX=*/nullptr,
          instr->fft_do_bit_reverse());
    }
    case FftType::IFFT: {
      return b.create<mlir::prime_ir::poly::NTTOp>(
          value, alloc_tensor, twiddle_factor, root_attr,
          /*tileX=*/nullptr, /*gridSizeX=*/nullptr, instr->fft_do_bit_reverse(),
          /*inverse=*/true);
    }

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled fft type: %s", FftType_Name(instr->fft_type())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFusionOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b,
    mlir::ValueRange operands) {
  auto* fusion_instr = Cast<HloFusionInstruction>(instr);
  HloComputation* computation = fusion_instr->called_computation();
  absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
  for (int64_t i = 0; i < computation->num_parameters(); ++i) {
    values[computation->parameter_instruction(i)] = operands[i];
  }
  computation->ForEachInstructionPostOrder(absl::bind_front(
      &CpuKernelEmitter::EmitOpInToApply, this, std::ref(b), std::ref(values)));
  return values[computation->root_instruction()];
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitMsmOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value scalars,
    mlir::Value bases) {
  pass_flag_.enable_elementwise_to_linalg = true;

  if (auto tensor_type =
          mlir::dyn_cast<mlir::RankedTensorType>(bases.getType())) {
    int64_t num_scalar_mul = instr->operand(0)->shape().dimensions(0);
    CHECK_GT(num_scalar_mul, 0);
    int64_t num_threads = tsl::port::MaxParallelism();
    int32_t window_bits = instr->window_bits();

    const Shape& shape = instr->shape();
    auto result_type = mlir_utils::PrimitiveTypeToMlirType(shape.element_type(),
                                                           b.getContext());
    if (num_scalar_mul < num_threads) {
      int32_t degree = static_cast<int32_t>(
          base::Log2Ceiling(static_cast<uint64_t>(num_scalar_mul)));
      return b.create<mlir::prime_ir::elliptic_curve::MSMOp>(
          result_type, scalars, bases, degree, window_bits);
    }

    pass_flag_.enable_expand_strided_metadata = true;

    int64_t chunk_size = (num_scalar_mul + num_threads - 1) / num_threads;
    int32_t degree = static_cast<int32_t>(
        base::Log2Ceiling(static_cast<uint64_t>(chunk_size)));

    auto zero = b.create<mlir::arith::ConstantIndexOp>(0);
    auto one = b.create<mlir::arith::ConstantIndexOp>(1);

    mlir::Value num_scalar_mul_value =
        b.create<mlir::arith::ConstantIndexOp>(num_scalar_mul);
    mlir::Value num_threads_value =
        b.create<mlir::arith::ConstantIndexOp>(num_threads);
    mlir::Value chunk_size_value =
        b.create<mlir::arith::ConstantIndexOp>(chunk_size);

    TF_ASSIGN_OR_RETURN(mlir::Value zero_point, CreateZeroPoint(b, shape));

    auto results_type = mlir::MemRefType::get({num_threads}, result_type);
    mlir::Value results = b.create<mlir::memref::AllocOp>(results_type);

    // Initialize results
    b.create<mlir::scf::ForOp>(
        /*lowerBound=*/zero, /*upperBound=*/num_threads_value, /*step=*/one,
        /*initArgs=*/std::nullopt,
        [zero_point, results](mlir::OpBuilder& nested_b, mlir::Location loc,
                              mlir::Value iv, mlir::ValueRange loop_vars) {
          mlir::ImplicitLocOpBuilder b(loc, nested_b);
          b.create<mlir::memref::StoreOp>(zero_point, results, iv);
          b.create<mlir::scf::YieldOp>();
        });

    // Parallel MSM computation
    b.create<mlir::scf::ParallelOp>(
        /*lowerBounds=*/mlir::ValueRange{zero},
        /*upperBounds=*/mlir::ValueRange{num_threads_value},
        /*steps=*/mlir::ValueRange{one},
        [&](mlir::OpBuilder& nested_b, mlir::Location loc,
            mlir::ValueRange ivs) {
          mlir::ImplicitLocOpBuilder b(loc, nested_b);
          auto i = ivs[0];

          auto start = b.create<mlir::arith::MulIOp>(i, chunk_size_value);
          auto next_i = b.create<mlir::arith::AddIOp>(i, one);
          auto end = b.create<mlir::arith::MulIOp>(next_i, chunk_size_value);

          auto actual_end =
              b.create<mlir::arith::MinUIOp>(end, num_scalar_mul_value);
          auto actual_chunk_size =
              b.create<mlir::arith::SubIOp>(actual_end, start);

          // Extract chunks
          auto scalars_chunk = b.create<mlir::tensor::ExtractSliceOp>(
              scalars, mlir::ValueRange{start},
              mlir::ValueRange{actual_chunk_size}, mlir::ValueRange{one});
          auto bases_chunk = b.create<mlir::tensor::ExtractSliceOp>(
              bases, mlir::ValueRange{start},
              mlir::ValueRange{actual_chunk_size}, mlir::ValueRange{one});

          // Compute MSM for chunk
          auto msm_result = b.create<mlir::prime_ir::elliptic_curve::MSMOp>(
              result_type, scalars_chunk, bases_chunk, degree, window_bits);

          b.create<mlir::memref::StoreOp>(msm_result, results, i);
        });

    // Combine results using reduction loop
    auto final_result = b.create<mlir::scf::ForOp>(
        /*lowerBound=*/zero, /*upperBound=*/num_threads_value, /*step=*/one,
        /*initArgs=*/mlir::ValueRange{zero_point},
        [result_type, results](mlir::OpBuilder& nested_b, mlir::Location loc,
                               mlir::Value iv, mlir::ValueRange loop_vars) {
          mlir::ImplicitLocOpBuilder b(loc, nested_b);
          auto acc = loop_vars[0];
          auto partial_result = b.create<mlir::memref::LoadOp>(results, iv);
          auto sum = b.create<mlir::prime_ir::elliptic_curve::AddOp>(
              result_type, partial_result, acc);
          b.create<mlir::scf::YieldOp>(mlir::ValueRange{sum});
        });

    return b.create<mlir::tensor::FromElementsOp>(
        MakeMlirTensorTypeWithoutLayout(shape, b.getContext()),
        mlir::ValueRange{final_result.getResult(0)});
  } else {
    return absl::InvalidArgumentError("bases is not a tensor");
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitBroadcastOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input,
    absl::Span<const int64_t> source_dimensions) {
  pass_flag_.enable_linalg_to_parallel_loops = true;

  int64_t rank = instr->shape().rank();
  auto target_dimensions = [source_dimensions, rank]() {
    std::unordered_set<int64_t> source_set(source_dimensions.begin(),
                                           source_dimensions.end());
    std::vector<int64_t> target_dimensions;

    if (source_dimensions.empty()) {
      // If no source dims, return all dims in order
      for (int64_t i = 0; i < rank; ++i) {
        target_dimensions.push_back(i);
      }
      return target_dimensions;
    }

    int64_t pivot = source_dimensions[0];

    // Step 1: dims > pivot (after)
    for (int64_t i = pivot + 1; i < rank; ++i) {
      if (source_set.count(i) == 0) {
        target_dimensions.push_back(i);
      }
    }

    // Step 2: dims < pivot (before)
    for (int64_t i = 0; i < pivot; ++i) {
      if (source_set.count(i) == 0) {
        target_dimensions.push_back(i);
      }
    }

    return target_dimensions;
  };

  auto init = b.create<mlir::tensor::EmptyOp>(
      MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext()),
      mlir::ValueRange{});

  auto broadcast =
      b.create<mlir::linalg::BroadcastOp>(input, init, target_dimensions());
  return broadcast.getResult()[0];
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitConcatenateOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b,
    mlir::ValueRange inputs) {
  pass_flag_.enable_expand_strided_metadata = true;

  return b.create<mlir::tensor::ConcatOp>(instr->concatenate_dimension(),
                                          inputs);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitMatrixVectorMultiplicationOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs,
    mlir::Value rhs) {
  if (LayoutUtil::IsDenseArray(instr->operand(0)->shape())) {
    return absl::UnimplementedError(
        "Dense matrix vector multiplication is not supported");
  }
  if (!LayoutUtil::IsCSRArray(instr->operand(0)->shape())) {
    return absl::UnimplementedError(
        "Only CSR matrix vector multiplication is supported");
  }
  pass_flag_.enable_linalg_to_parallel_loops = true;

  mlir::MLIRContext* ctx = lhs.getContext();
  auto result_type = mlir::cast<mlir::RankedTensorType>(
      MakeMlirTensorTypeWithoutLayout(instr->shape(), ctx));
  llvm::SmallVector<int64_t> shapes;
  for (int64_t i = 0; i < instr->shape().dimensions_size(); ++i) {
    shapes.push_back(instr->shape().dimensions(i));
  }
  auto result =
      b.create<mlir::tensor::EmptyOp>(shapes, result_type.getElementType());

  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);

  auto generic_op = b.create<mlir::linalg::GenericOp>(
      /*resultTensorTypes=*/mlir::TypeRange{result_type},
      /*inputs=*/mlir::ValueRange{lhs, rhs},
      /*outputs=*/mlir::ValueRange{result},
      /*indexingMaps=*/
      llvm::SmallVector<mlir::AffineMap>{
          mlir::AffineMap::get(2, 0, {d0, d1}, ctx),
          mlir::AffineMap::get(2, 0, {d1}, ctx),
          mlir::AffineMap::get(2, 0, {d0}, ctx),
      },
      /*iteratorTypes=*/
      llvm::SmallVector<mlir::utils::IteratorType>{
          mlir::utils::IteratorType::parallel,
          mlir::utils::IteratorType::reduction,
      },
      /*doc=*/"matrix vector multiplication",
      /*libraryCall=*/mlir::StringRef(),
      [](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) {
        mlir::ImplicitLocOpBuilder b(loc, builder);
        auto x = args[0];
        auto y = args[1];
        auto z = args[2];

        auto mul_op =
            b.create<mlir::sparse_tensor::BinaryOp>(x.getType(), x, y);
        {
          mlir::Region& overlap_region = mul_op.getOverlapRegion();
          mlir::Block* block = b.createBlock(&overlap_region);
          block->addArguments({x.getType(), y.getType()}, {loc, loc});
          b.setInsertionPointToStart(block);
          auto mul = b.create<mlir::prime_ir::field::MulOp>(
              block->getArgument(0), block->getArgument(1));
          b.create<mlir::sparse_tensor::YieldOp>(mul);
        }

        b.setInsertionPointAfter(mul_op);
        auto reduce_op = b.create<mlir::sparse_tensor::ReduceOp>(
            mul_op.getType(), mul_op, z,
            /*identity=*/
            b.create<mlir::prime_ir::field::ConstantOp>(mul_op.getType(), 0));
        {
          mlir::Region& reduce_region = reduce_op.getRegion();
          mlir::Block* block = b.createBlock(&reduce_region);
          block->addArguments({mul_op.getType(), z.getType()}, {loc, loc});
          b.setInsertionPointToStart(block);
          auto add = b.create<mlir::prime_ir::field::AddOp>(
              block->getArgument(0), block->getArgument(1));
          b.create<mlir::sparse_tensor::YieldOp>(add);
        }

        b.setInsertionPointAfter(reduce_op);
        b.create<mlir::linalg::YieldOp>(
            mlir::ValueRange{reduce_op.getOutput()});
      });
  return generic_op.getResult(0);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitDotOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs,
    mlir::Value rhs) {
  int64_t rank0 = instr->operand(0)->shape().rank();
  int64_t rank1 = instr->operand(1)->shape().rank();

  if (rank0 == 1) {
    if (rank1 == 1) {
      return absl::UnimplementedError(
          "Dot op with vector and vector is not supported");
    } else if (rank1 == 2) {
      return absl::UnimplementedError(
          "Dot op with vector and matrix is not supported");
    }
  } else if (rank0 == 2) {
    if (rank1 == 1) {
      return EmitMatrixVectorMultiplicationOp(instr, b, lhs, rhs);
    } else if (rank1 == 2) {
      return absl::UnimplementedError(
          "Dot op with matrix and matrix is not supported");
    }
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Dot op with rank %d and rank %d is not supported", rank0, rank1));
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitDynamicSliceOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input,
    mlir::ValueRange start_indices) {
  pass_flag_.enable_expand_strided_metadata = true;

  llvm::SmallVector<mlir::Value> offsets;
  llvm::SmallVector<int64_t> static_offsets;
  llvm::SmallVector<int64_t> static_strides;
  for (size_t i = 0; i < instr->shape().rank(); ++i) {
    auto start_index = b.create<mlir::tensor::ExtractOp>(start_indices[i], {});
    offsets.push_back(
        b.create<mlir::arith::IndexCastOp>(b.getIndexType(), start_index));
    static_offsets.push_back(mlir::ShapedType::kDynamic);
    static_strides.push_back(1);
  }

  return b.create<mlir::tensor::ExtractSliceOp>(
      MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext()), input,
      offsets,
      /*sizes=*/mlir::ValueRange{},
      /*strides=*/mlir::ValueRange{},
      /*static_offsets=*/static_offsets, instr->dynamic_slice_sizes(),
      static_strides);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitDynamicUpdateSliceOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value dest,
    mlir::Value update, mlir::ValueRange start_indices) {
  pass_flag_.enable_expand_strided_metadata = true;

  llvm::SmallVector<mlir::Value> offsets;
  llvm::SmallVector<mlir::Value> sizes;
  llvm::SmallVector<int64_t> static_offsets;
  llvm::SmallVector<int64_t> static_sizes;
  llvm::SmallVector<int64_t> static_strides;
  for (size_t i = 0; i < instr->shape().rank(); ++i) {
    auto start_index = b.create<mlir::tensor::ExtractOp>(start_indices[i], {});
    offsets.push_back(
        b.create<mlir::arith::IndexCastOp>(b.getIndexType(), start_index));
    sizes.push_back(b.create<mlir::tensor::DimOp>(update, i));
    static_offsets.push_back(mlir::ShapedType::kDynamic);
    static_sizes.push_back(mlir::ShapedType::kDynamic);
    static_strides.push_back(1);
  }
  auto update_shape_type = mlir::cast<mlir::ShapedType>(update.getType());
  llvm::SmallVector<int64_t> update_shape;
  for (int64_t i = 0; i < instr->shape().rank(); ++i) {
    update_shape.push_back(mlir::ShapedType::kDynamic);
  }
  auto dynamic_update_tensor_type = mlir::RankedTensorType::get(
      update_shape, update_shape_type.getElementType());
  auto casted_update =
      b.create<mlir::tensor::CastOp>(dynamic_update_tensor_type, update);

  return b.create<mlir::tensor::InsertSliceOp>(
      casted_update, dest, offsets, sizes,
      /*strides=*/mlir::ValueRange{},
      /*static_offsets=*/static_offsets, static_sizes, static_strides);
}

namespace {

struct GatherMetadata {
  llvm::SmallVector<int64_t> batch_dims;
  absl::flat_hash_set<int64_t> collapsed_set;
};

GatherMetadata BuildGatherMetadata(const HloGatherInstruction* gather) {
  const auto& dim_numbers = gather->gather_dimension_numbers();
  const Shape& output_shape = gather->shape();

  absl::Span<const int64_t> offset_dims = dim_numbers.offset_dims();
  absl::Span<const int64_t> collapsed_slice_dims =
      dim_numbers.collapsed_slice_dims();

  // Identify batch dims (output dims not in offset_dims)
  absl::flat_hash_set<int64_t> offset_dims_set(offset_dims.begin(),
                                               offset_dims.end());
  llvm::SmallVector<int64_t> batch_dims;
  for (int64_t d = 0; d < output_shape.rank(); ++d) {
    if (!offset_dims_set.contains(d)) {
      batch_dims.push_back(d);
    }
  }

  absl::flat_hash_set<int64_t> collapsed_set(collapsed_slice_dims.begin(),
                                             collapsed_slice_dims.end());

  return GatherMetadata{
      .batch_dims = std::move(batch_dims),
      .collapsed_set = std::move(collapsed_set),
  };
}

}  // namespace

llvm::SmallVector<mlir::Value> CpuKernelEmitter::LookUpGatherStartValues(
    EmitterLocOpBuilder& b, mlir::Value start_indices_tensor,
    llvm::ArrayRef<mlir::Value> batch_idx, int64_t index_vector_dim,
    const Shape& indices_shape,
    absl::Span<const int64_t> start_index_map) const {
  int64_t num_index_dims = static_cast<int64_t>(start_index_map.size());
  llvm::SmallVector<mlir::Value> start_vals(num_index_dims);
  for (int64_t i = 0; i < num_index_dims; ++i) {
    llvm::SmallVector<mlir::Value> lookup_idx;
    if (indices_shape.rank() == 1) {
      lookup_idx.push_back(batch_idx[0]);
    } else {
      lookup_idx.assign(batch_idx.begin(), batch_idx.end());
      if (index_vector_dim < indices_shape.rank()) {
        auto i_val = b.create<mlir::arith::ConstantIndexOp>(i);
        lookup_idx.insert(lookup_idx.begin() + index_vector_dim, i_val);
      }
    }
    auto raw =
        b.create<mlir::tensor::ExtractOp>(start_indices_tensor, lookup_idx);
    start_vals[i] = b.create<mlir::arith::IndexCastOp>(b.getIndexType(), raw);
  }
  return start_vals;
}

llvm::SmallVector<mlir::Value> CpuKernelEmitter::ComputeGatherOperandIndices(
    EmitterLocOpBuilder& b, mlir::ValueRange output_indices,
    llvm::ArrayRef<mlir::Value> start_vals, const Shape& operand_shape,
    absl::Span<const int64_t> offset_dims,
    absl::Span<const int64_t> slice_sizes,
    absl::Span<const int64_t> start_index_map,
    const absl::flat_hash_set<int64_t>& collapsed_set) const {
  // Build a mapping: operand dim -> position in index vector
  absl::flat_hash_map<int64_t, int64_t> operand_dim_to_start_index;
  for (int64_t i = 0; i < start_index_map.size(); ++i) {
    operand_dim_to_start_index[start_index_map[i]] = i;
  }

  int64_t operand_rank = operand_shape.rank();
  mlir::Value zero = b.create<mlir::arith::ConstantIndexOp>(0);

  llvm::SmallVector<mlir::Value> operand_indices(operand_rank, zero);
  int64_t offset_idx = 0;
  for (int64_t k = 0; k < operand_rank; ++k) {
    mlir::Value window_idx = zero;
    if (!collapsed_set.contains(k)) {
      window_idx = output_indices[offset_dims[offset_idx++]];
    }

    auto it = operand_dim_to_start_index.find(k);
    if (it != operand_dim_to_start_index.end()) {
      int64_t i = it->second;
      int64_t max_start = operand_shape.dimensions(k) - slice_sizes[k];
      auto max_val = b.create<mlir::arith::ConstantIndexOp>(max_start);
      // Clamp the start index to [0, max_start] so the slice never reads
      // out-of-bounds.  max_start = dim_size - slice_size.
      //   clamped = min(max(start, 0), max_start)
      auto clamped = b.create<mlir::arith::MinSIOp>(
          b.create<mlir::arith::MaxSIOp>(start_vals[i], zero), max_val);
      operand_indices[k] = b.create<mlir::arith::AddIOp>(clamped, window_idx);
    } else {
      operand_indices[k] = window_idx;
    }
  }
  return operand_indices;
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitGatherOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value operand,
    mlir::Value start_indices_tensor) {
  pass_flag_.enable_linalg_to_parallel_loops = true;
  pass_flag_.enable_lower_affine = true;

  auto* gather = Cast<HloGatherInstruction>(instr);
  const auto& dim_numbers = gather->gather_dimension_numbers();
  const Shape& operand_shape = instr->operand(0)->shape();
  const Shape& indices_shape = instr->operand(1)->shape();
  absl::Span<const int64_t> offset_dims = dim_numbers.offset_dims();
  absl::Span<const int64_t> start_index_map = dim_numbers.start_index_map();
  absl::Span<const int64_t> slice_sizes = gather->gather_slice_sizes();
  int64_t index_vector_dim = dim_numbers.index_vector_dim();

  auto meta = BuildGatherMetadata(gather);

  auto output_type =
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b.getContext());

  auto generate_op = b.create<mlir::tensor::GenerateOp>(
      output_type, mlir::ValueRange{},
      [&](mlir::OpBuilder& ob, mlir::Location loc, mlir::ValueRange indices) {
        EmitterLocOpBuilder gb(loc, ob);

        llvm::SmallVector<mlir::Value> batch_idx;
        for (int64_t d : meta.batch_dims) {
          batch_idx.push_back(indices[d]);
        }

        auto start_vals = LookUpGatherStartValues(
            gb, start_indices_tensor, batch_idx, index_vector_dim,
            indices_shape, start_index_map);
        auto operand_indices = ComputeGatherOperandIndices(
            gb, indices, start_vals, operand_shape, offset_dims, slice_sizes,
            start_index_map, meta.collapsed_set);

        auto result =
            gb.create<mlir::tensor::ExtractOp>(operand, operand_indices);
        gb.create<mlir::tensor::YieldOp>(result);
      });
  return generate_op.getResult();
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitIotaOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b) {
  pass_flag_.enable_linalg_to_parallel_loops = true;
  pass_flag_.enable_lower_affine = true;

  PrimitiveType element_type = instr->shape().element_type();
  if (!(primitive_util::IsIntegralType(element_type) ||
        primitive_util::IsFieldType(element_type))) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unhandled primitive type: %s",
        primitive_util::LowercasePrimitiveTypeName(element_type)));
  }

  auto output_type =
      MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext());
  bool is_signed = primitive_util::IsSignedIntegralType(element_type);

  int64_t iota_dimension = instr->iota_dimension();
  auto iota_op = b.create<mlir::tensor::GenerateOp>(
      output_type, /*dynamicExtents=*/mlir::ValueRange{},
      [&](mlir::OpBuilder& nested_b, mlir::Location loc,
          mlir::ValueRange indices) {
        mlir::ImplicitLocOpBuilder b(loc, nested_b);

        mlir::Value value_as_index = indices[iota_dimension];
        mlir::Type element_type =
            mlir::cast<mlir::RankedTensorType>(output_type).getElementType();

        mlir::Value value;
        if (element_type.isInteger()) {
          if (is_signed) {
            value = b.create<mlir::arith::IndexCastOp>(element_type,
                                                       value_as_index);
          } else {
            value = b.create<mlir::arith::IndexCastUIOp>(element_type,
                                                         value_as_index);
          }
        } else if (auto prime_field_type =
                       mlir::dyn_cast<mlir::prime_ir::field::PrimeFieldType>(
                           element_type)) {
          value = b.create<mlir::arith::IndexCastUIOp>(
              prime_field_type.getStorageType(), value_as_index);

          if (prime_field_type.isMontgomery()) {
            value = b.create<mlir::prime_ir::field::BitcastOp>(
                mlir::prime_ir::field::getStandardFormType(element_type),
                value);
            value =
                b.create<mlir::prime_ir::field::ToMontOp>(element_type, value);
          } else {
            value =
                b.create<mlir::prime_ir::field::BitcastOp>(element_type, value);
          }
        }

        b.create<mlir::tensor::YieldOp>(value);
      });

  return iota_op.getResult();
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitMapOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b,
    mlir::ValueRange inputs) {
  pass_flag_.enable_linalg_to_parallel_loops = true;

  auto output = b.create<mlir::tensor::EmptyOp>(
      MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext()),
      mlir::ValueRange{});

  HloComputation* to_apply = instr->to_apply();
  return b
      .create<mlir::linalg::MapOp>(
          inputs, output,
          [this, to_apply](mlir::OpBuilder& nested_b, mlir::Location loc,
                           mlir::ValueRange loop_vars) {
            EmitterLocOpBuilder b(loc, nested_b);
            absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
            for (int64_t i = 0; i < to_apply->num_parameters(); ++i) {
              values[to_apply->parameter_instruction(i)] = loop_vars[i];
            }
            to_apply->ForEachInstructionPostOrder(
                absl::bind_front(&CpuKernelEmitter::EmitOpInToApply, this,
                                 std::ref(b), std::ref(values)));
            b.create<mlir::linalg::YieldOp>(
                values[to_apply->root_instruction()]);
          })
      ->getResult(0);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitPadOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input,
    mlir::Value padding_value) {
  pass_flag_.enable_tensor_to_linalg = true;
  pass_flag_.enable_expand_strided_metadata = true;

  llvm::SmallVector<mlir::OpFoldResult> lower_edge_padding_low;
  llvm::SmallVector<mlir::OpFoldResult> lower_edge_padding_high;
  const PaddingConfig& padding_config = instr->padding_config();
  for (const PaddingConfig::PaddingConfigDimension& dimension :
       padding_config.dimensions()) {
    lower_edge_padding_low.push_back(
        b.getIndexAttr(dimension.edge_padding_low()));
    lower_edge_padding_high.push_back(
        b.getIndexAttr(dimension.edge_padding_high()));
  }

  mlir::Value padding_value_scalar =
      b.create<mlir::tensor::ExtractOp>(padding_value, {});
  return b.create<mlir::tensor::PadOp>(
      MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext()), input,
      lower_edge_padding_low, lower_edge_padding_high, padding_value_scalar);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitReduceOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b,
    mlir::ValueRange inputs, mlir::ValueRange inits) {
  if (instr->dimensions().empty()) {
    // NOTE(chokobole): Detected an execution discrepancy between XLA runners
    // (verified on XLA commit 8bac4a2).
    //
    // For a reduction over an empty dimension set (dimensions={}), the
    // behavior is inconsistent:
    // 1. Interpreter: Correctly returns the identity value (%init).
    // 2. CPU Runner: Incorrectly returns the input tensor (%x) instead of
    //    reducing it to a scalar.
    //
    // HLO Snippet:
    // %func {
    //   %a = s32[] parameter(0)
    //   %b = s32[] parameter(1)
    //   %mul = s32[] multiply(%a, %b)
    //   ROOT %ret = s32[] add(%mul, %a)
    // }
    //
    // ENTRY %main {
    //   %x = s32[4] constant({1, 2, 3, 4})
    //   %init = s32[] constant(0)
    //   ROOT %ret = s32[4] reduce(%x, %init), dimensions={}, to_apply=%func
    // }
    return absl::UnimplementedError(
        "ReduceOp with empty dimensions is not supported");
  }

  pass_flag_.enable_linalg_to_parallel_loops = true;

  HloComputation* to_apply = instr->to_apply();
  auto reduce = b.create<mlir::linalg::ReduceOp>(
      inputs, inits, instr->dimensions(),
      [this, to_apply](mlir::OpBuilder& nested_b, mlir::Location loc,
                       mlir::ValueRange loop_vars) {
        EmitterLocOpBuilder b(loc, nested_b);
        absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
        values[to_apply->parameter_instruction(0)] = loop_vars[1];
        values[to_apply->parameter_instruction(1)] = loop_vars[0];

        to_apply->ForEachInstructionPostOrder(
            absl::bind_front(&CpuKernelEmitter::EmitOpInToApply, this,
                             std::ref(b), std::ref(values)));

        b.create<mlir::linalg::YieldOp>(
            mlir::ValueRange{values[to_apply->root_instruction()]});
      });

  return reduce.getResult(0);
}

namespace {

struct WindowDimConstants {
  mlir::Value stride;
  mlir::Value dilation;
  mlir::Value pad_low;
  mlir::Value input_dim;
  mlir::Value base_dilation;  // null if base_dilation <= 1
  mlir::Value reversal_max;   // null if no reversal
};

struct ReduceWindowContext {
  int64_t rank;
  const Window& window;
  llvm::ArrayRef<WindowDimConstants> dim_constants;
  HloComputation* to_apply;
  mlir::Value input;
  mlir::Value init_scalar;
  mlir::Value zero;
  std::function<mlir::Value(EmitterLocOpBuilder&, HloComputation*, mlir::Value,
                            mlir::Value)>
      apply_reduction;
};

std::pair<mlir::Value, mlir::Value> ComputeWindowInputIndexForDim(
    EmitterLocOpBuilder& b, const WindowDimension& dim,
    const WindowDimConstants& constants, mlir::Value output_iv,
    mlir::Value window_iv, mlir::Value zero) {
  mlir::Value win_iv = window_iv;
  if (dim.window_reversal()) {
    win_iv = b.create<mlir::arith::SubIOp>(constants.reversal_max, win_iv);
  }

  // base_idx = output_iv * stride + window_iv * dilation - pad_low
  auto strided = b.create<mlir::arith::MulIOp>(output_iv, constants.stride);
  auto dilated = b.create<mlir::arith::MulIOp>(win_iv, constants.dilation);
  auto sum = b.create<mlir::arith::AddIOp>(strided, dilated);
  mlir::Value input_idx = b.create<mlir::arith::SubIOp>(sum, constants.pad_low);

  mlir::Value in_bounds =
      b.create<mlir::arith::ConstantOp>(b.getI1Type(), b.getBoolAttr(true));

  if (dim.base_dilation() > 1) {
    auto rem =
        b.create<mlir::arith::RemSIOp>(input_idx, constants.base_dilation);
    auto is_aligned = b.create<mlir::arith::CmpIOp>(
        mlir::arith::CmpIPredicate::eq, rem, zero);
    in_bounds = b.create<mlir::arith::AndIOp>(in_bounds, is_aligned);
    input_idx =
        b.create<mlir::arith::DivSIOp>(input_idx, constants.base_dilation);
  }

  auto ge_zero = b.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::sge,
                                               input_idx, zero);
  auto lt_dim = b.create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::slt,
                                              input_idx, constants.input_dim);
  auto dim_ok = b.create<mlir::arith::AndIOp>(ge_zero, lt_dim);
  in_bounds = b.create<mlir::arith::AndIOp>(in_bounds, dim_ok);

  return {input_idx, in_bounds};
}

mlir::Value EmitBoundsCheckedReduction(
    EmitterLocOpBuilder& b, ReduceWindowContext& ctx, mlir::Value in_bounds,
    mlir::Value acc, mlir::Value input,
    llvm::ArrayRef<mlir::Value> input_indices, mlir::Value init_scalar) {
  auto if_op = b.create<mlir::scf::IfOp>(
      in_bounds,
      [&](mlir::OpBuilder& then_b, mlir::Location loc) {
        EmitterLocOpBuilder tb(loc, then_b);
        auto element = tb.create<mlir::tensor::ExtractOp>(input, input_indices);
        auto result = ctx.apply_reduction(tb, ctx.to_apply, acc, element);
        tb.create<mlir::scf::YieldOp>(mlir::ValueRange{result});
      },
      [&](mlir::OpBuilder& else_b, mlir::Location loc) {
        EmitterLocOpBuilder eb(loc, else_b);
        auto result = ctx.apply_reduction(eb, ctx.to_apply, acc, init_scalar);
        eb.create<mlir::scf::YieldOp>(mlir::ValueRange{result});
      });
  return if_op.getResult(0);
}

mlir::Value BuildWindowLoopNest(
    EmitterLocOpBuilder& b, ReduceWindowContext& ctx, int64_t dim,
    mlir::Value acc, llvm::ArrayRef<mlir::Value> output_ivs,
    llvm::SmallVectorImpl<mlir::Value>& window_ivs) {
  if (dim == ctx.rank) {
    llvm::SmallVector<mlir::Value> input_indices;
    mlir::Value in_bounds =
        b.create<mlir::arith::ConstantOp>(b.getI1Type(), b.getBoolAttr(true));

    for (int64_t d = 0; d < ctx.rank; ++d) {
      auto [idx, bounds] = ComputeWindowInputIndexForDim(
          b, ctx.window.dimensions(d), ctx.dim_constants[d], output_ivs[d],
          window_ivs[d], ctx.zero);
      input_indices.push_back(idx);
      in_bounds = b.create<mlir::arith::AndIOp>(in_bounds, bounds);
    }

    return EmitBoundsCheckedReduction(b, ctx, in_bounds, acc, ctx.input,
                                      input_indices, ctx.init_scalar);
  }

  const auto& wd = ctx.window.dimensions(dim);
  auto loop = b.create<mlir::affine::AffineForOp>(
      0, wd.size(), 1, mlir::ValueRange{acc},
      [&](mlir::OpBuilder& nb, mlir::Location loc, mlir::Value iv,
          mlir::ValueRange iter_args) {
        EmitterLocOpBuilder inner_b(loc, nb);
        window_ivs.push_back(iv);
        mlir::Value new_acc = BuildWindowLoopNest(
            inner_b, ctx, dim + 1, iter_args[0], output_ivs, window_ivs);
        window_ivs.pop_back();
        inner_b.create<mlir::affine::AffineYieldOp>(mlir::ValueRange{new_acc});
      });
  return loop.getResult(0);
}

}  // namespace

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitReduceWindowOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b,
    mlir::ValueRange inputs, mlir::ValueRange inits) {
  pass_flag_.enable_lower_affine = true;

  auto* reduce_window = Cast<HloReduceWindowInstruction>(instr);
  const Shape& input_shape = reduce_window->inputs()[0]->shape();
  const Shape& output_shape = instr->shape();
  const Window& window = reduce_window->window();
  HloComputation* to_apply = reduce_window->to_apply();
  int64_t rank = output_shape.rank();

  mlir::Value input = inputs[0];
  mlir::Value zero = b.create<mlir::arith::ConstantIndexOp>(0);
  mlir::Value init_scalar =
      b.create<mlir::tensor::ExtractOp>(inits[0], mlir::ValueRange{});

  auto output_type =
      mlir_utils::ShapeToMlirTensorType(output_shape, b.getContext());

  // Precompute window dimension constants
  llvm::SmallVector<WindowDimConstants> dim_constants(rank);
  for (int64_t d = 0; d < rank; ++d) {
    const auto& wd = window.dimensions(d);
    auto& dc = dim_constants[d];
    dc.stride = b.create<mlir::arith::ConstantIndexOp>(wd.stride());
    dc.dilation = b.create<mlir::arith::ConstantIndexOp>(wd.window_dilation());
    dc.pad_low = b.create<mlir::arith::ConstantIndexOp>(wd.padding_low());
    dc.input_dim =
        b.create<mlir::arith::ConstantIndexOp>(input_shape.dimensions(d));
    if (wd.base_dilation() > 1) {
      dc.base_dilation =
          b.create<mlir::arith::ConstantIndexOp>(wd.base_dilation());
    }
    if (wd.window_reversal()) {
      dc.reversal_max = b.create<mlir::arith::ConstantIndexOp>(wd.size() - 1);
    }
  }

  auto apply_reduction = [this](EmitterLocOpBuilder& b,
                                HloComputation* to_apply, mlir::Value acc,
                                mlir::Value value) -> mlir::Value {
    absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
    values[to_apply->parameter_instruction(0)] = acc;
    values[to_apply->parameter_instruction(1)] = value;
    to_apply->ForEachInstructionPostOrder(
        absl::bind_front(&CpuKernelEmitter::EmitOpInToApply, this, std::ref(b),
                         std::ref(values)));
    return values[to_apply->root_instruction()];
  };

  ReduceWindowContext ctx{rank,  window,      dim_constants, to_apply,
                          input, init_scalar, zero,          apply_reduction};

  auto generate_op = b.create<mlir::tensor::GenerateOp>(
      output_type, mlir::ValueRange{},
      [&](mlir::OpBuilder& ob, mlir::Location loc, mlir::ValueRange indices) {
        EmitterLocOpBuilder inner_b(loc, ob);
        llvm::SmallVector<mlir::Value> output_ivs(indices.begin(),
                                                  indices.end());
        llvm::SmallVector<mlir::Value> window_ivs;
        mlir::Value reduced = BuildWindowLoopNest(inner_b, ctx, 0, init_scalar,
                                                  output_ivs, window_ivs);
        inner_b.create<mlir::tensor::YieldOp>(reduced);
      });
  return generate_op.getResult();
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitReshapeOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input) {
  auto output_type =
      MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext());
  mlir::Value shape = b.create<mlir::arith::ConstantOp>(
      b.getIndexTensorAttr(instr->shape().dimensions()));
  return b.create<mlir::tensor::ReshapeOp>(output_type, input, shape);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitBitReverseOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input) {
  // Enable passes needed to lower tensor_ext::BitReverseOp
  // The TensorExtToTensor pass converts BitReverseOp to scf::ParallelOp,
  // memref ops, and LLVM::BitReverseOp
  pass_flag_.enable_tensor_ext_to_tensor = true;
  pass_flag_.enable_scf_to_cf = true;

  if (instr->dimensions().size() != 1) {
    return absl::FailedPreconditionError(
        "CPU backend for BitReverseOp only supports one dimension.");
  }

  auto dest = b.create<mlir::tensor::EmptyOp>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b.getContext()),
      mlir::ValueRange{});

  return b.create<mlir::prime_ir::tensor_ext::BitReverseOp>(
      input, dest, instr->dimensions()[0]);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitReverseOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input) {
  if (instr->dimensions().empty()) {
    return input;
  }

  pass_flag_.enable_linalg_to_parallel_loops = true;
  pass_flag_.enable_lower_affine = true;

  CHECK(mlir::cast<mlir::RankedTensorType>(input.getType()).hasStaticShape());

  const Shape& shape = instr->shape();
  auto output = b.create<mlir::tensor::EmptyOp>(
      MakeMlirTensorTypeWithoutLayout(shape, b.getContext()),
      mlir::ValueRange{});

  llvm::SmallVector<mlir::AffineExpr, 3> output_exprs;
  for (int64_t i = 0; i < shape.rank(); ++i) {
    auto it =
        std::find(instr->dimensions().begin(), instr->dimensions().end(), i);
    if (it != instr->dimensions().end()) {
      output_exprs.push_back(b.getAffineConstantExpr(shape.dimensions(i) - 1) -
                             b.getAffineDimExpr(i));
    } else {
      output_exprs.push_back(b.getAffineDimExpr(i));
    }
  }

  llvm::SmallVector<mlir::AffineMap, 2> indexing_maps;
  indexing_maps.push_back(
      mlir::AffineMap::getMultiDimIdentityMap(shape.rank(), b.getContext()));
  indexing_maps.push_back(
      mlir::AffineMap::get(shape.rank(), 0, output_exprs, b.getContext()));

  llvm::SmallVector<mlir::utils::IteratorType, 3> iterator_types;
  for (int64_t i = 0; i < instr->shape().rank(); ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }

  return b
      .create<mlir::linalg::GenericOp>(
          mlir::TypeRange{output.getType()}, mlir::ValueRange{input},
          mlir::ValueRange{output}, indexing_maps, iterator_types,
          /*doc=*/"reverse",
          /*libraryCall=*/mlir::StringRef(),
          [](mlir::OpBuilder& builder, mlir::Location loc,
             mlir::ValueRange args) {
            mlir::ImplicitLocOpBuilder b(loc, builder);
            b.create<mlir::linalg::YieldOp>(mlir::ValueRange{args[0]});
          })
      .getResult(0);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitSliceOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  pass_flag_.enable_expand_strided_metadata = true;

  const Shape& shape = instr->shape();

  auto result_type = MakeMlirTensorTypeWithoutLayout(shape, b.getContext());

  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;

  absl::Span<const int64_t> slices_starts = instr->slice_starts();
  absl::Span<const int64_t> slices_limits = instr->slice_limits();
  absl::Span<const int64_t> slices_strides = instr->slice_strides();

  for (int64_t i = 0; i < shape.rank(); ++i) {
    offsets.push_back(b.getIndexAttr(slices_starts[i]));
    sizes.push_back(b.getIndexAttr(slices_limits[i] - slices_starts[i]));
    strides.push_back(b.getIndexAttr(slices_strides[i]));
  }

  return b.create<mlir::tensor::ExtractSliceOp>(result_type, value, offsets,
                                                sizes, strides);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitTransposeOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input) {
  if (instr->dimensions().empty()) {
    return input;
  }

  pass_flag_.enable_linalg_to_parallel_loops = true;

  auto output = b.create<mlir::tensor::EmptyOp>(
      MakeMlirTensorTypeWithoutLayout(instr->shape(), b.getContext()),
      mlir::ValueRange{});

  return b
      .create<mlir::linalg::TransposeOp>(input, output, instr->dimensions())
      ->getResult(0);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b,
    absl::flat_hash_map<const HloInstruction*, mlir::Value>& values) {
  auto enable_flag = [this](PrimitiveType element_type) {
    if (primitive_util::IsFieldType(element_type)) {
      pass_flag_.enable_field_to_arith = true;
      return;
    } else if (primitive_util::IsEcPointType(element_type)) {
      pass_flag_.enable_elliptic_curve_to_field = true;
      pass_flag_.enable_elliptic_curve_to_llvm = true;
      switch (element_type) {
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) case enum:
        ZK_DTYPES_PUBLIC_R2_EC_POINT_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
        pass_flag_.enable_ext_field_to_llvm = true;
        break;
        default:
          break;
      }
    }
  };

  enable_flag(instr->shape().element_type());
  if (instr->IsElementwise() && instr->shape().IsArray()) {
    pass_flag_.enable_elementwise_to_linalg = true;
  }

  switch (instr->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kInverse:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kSign: {
      enable_flag(instr->operand(0)->shape().element_type());
      return EmitUnaryOp(instr, b, values[instr->operand(0)]);
    }
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kCompare:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kOr:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSubtract:
    case HloOpcode::kXor: {
      enable_flag(instr->operand(0)->shape().element_type());
      enable_flag(instr->operand(1)->shape().element_type());
      return EmitBinaryOp(instr, b, values[instr->operand(0)],
                          values[instr->operand(1)]);
    }
    case HloOpcode::kBitReverse:
      return EmitBitReverseOp(instr, b, values[instr->operand(0)]);
    case HloOpcode::kBroadcast:
      return EmitBroadcastOp(instr, b, values[instr->operand(0)],
                             instr->dimensions());
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
      return EmitTernaryOp(instr, b, values[instr->operand(0)],
                           values[instr->operand(1)],
                           values[instr->operand(2)]);
    case HloOpcode::kConcatenate: {
      llvm::SmallVector<mlir::Value> inputs;
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        inputs.push_back(values[instr->operand(i)]);
      }
      return EmitConcatenateOp(instr, b, inputs);
    }
    case HloOpcode::kDot:
      return EmitDotOp(instr, b, values[instr->operand(0)],
                       values[instr->operand(1)]);
    case HloOpcode::kDynamicSlice: {
      llvm::SmallVector<mlir::Value> start_indices;
      for (int64_t i = 1; i < instr->operand_count(); ++i) {
        start_indices.push_back(values[instr->operand(i)]);
      }
      return EmitDynamicSliceOp(instr, b, values[instr->operand(0)],
                                start_indices);
    }
    case HloOpcode::kDynamicUpdateSlice: {
      llvm::SmallVector<mlir::Value> start_indices;
      for (int64_t i = 2; i < instr->operand_count(); ++i) {
        start_indices.push_back(values[instr->operand(i)]);
      }
      return EmitDynamicUpdateSliceOp(instr, b, values[instr->operand(0)],
                                      values[instr->operand(1)], start_indices);
    }
    case HloOpcode::kFft: {
      if (instr->operand_count() == 1) {
        return EmitFftOp(instr, b, values[instr->operand(0)]);
      } else if (instr->operand_count() == 2) {
        return EmitFftOp(instr, b, values[instr->operand(0)],
                         values[instr->operand(1)]);
      } else {
        return absl::InternalError(
            "HloFftInstruction shouldn't have more than 2 arguments");
      }
    }
    case HloOpcode::kGather:
      return EmitGatherOp(instr, b, values[instr->operand(0)],
                          values[instr->operand(1)]);
    case HloOpcode::kFusion: {
      llvm::SmallVector<mlir::Value> operands;
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        operands.push_back(values[instr->operand(i)]);
      }
      return EmitFusionOp(instr, b, operands);
    }
    case HloOpcode::kIota:
      return EmitIotaOp(instr, b);
    case HloOpcode::kMap: {
      llvm::SmallVector<mlir::Value> inputs;
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        inputs.push_back(values[instr->operand(i)]);
      }
      return EmitMapOp(instr, b, inputs);
    }
    case HloOpcode::kMsm: {
      enable_flag(instr->operand(0)->shape().element_type());
      enable_flag(instr->operand(1)->shape().element_type());
      return EmitMsmOp(instr, b, values[instr->operand(0)],
                       values[instr->operand(1)]);
    }
    case HloOpcode::kPad:
      return EmitPadOp(instr, b, values[instr->operand(0)],
                       values[instr->operand(1)]);
    case HloOpcode::kReduce: {
      auto* reduce_instr = Cast<HloReduceInstruction>(instr);
      llvm::SmallVector<mlir::Value> inputs;
      for (const HloInstruction* input : reduce_instr->inputs()) {
        inputs.push_back(values.at(input));
      }
      llvm::SmallVector<mlir::Value> inits;
      for (const HloInstruction* init : reduce_instr->init_values()) {
        inits.push_back(values.at(init));
      }
      return EmitReduceOp(instr, b, inputs, inits);
    }
    case HloOpcode::kReduceWindow: {
      auto* reduce_window = Cast<HloReduceWindowInstruction>(instr);
      llvm::SmallVector<mlir::Value> input_vals, init_vals;
      for (const HloInstruction* inp : reduce_window->inputs()) {
        input_vals.push_back(values.at(inp));
      }
      for (const HloInstruction* ini : reduce_window->init_values()) {
        init_vals.push_back(values.at(ini));
      }
      return EmitReduceWindowOp(instr, b, input_vals, init_vals);
    }
    case HloOpcode::kReshape:
      return EmitReshapeOp(instr, b, values[instr->operand(0)]);
    case HloOpcode::kReverse:
      return EmitReverseOp(instr, b, values[instr->operand(0)]);
    case HloOpcode::kSlice:
      return EmitSliceOp(instr, b, values[instr->operand(0)]);
    case HloOpcode::kTranspose:
      return EmitTransposeOp(instr, b, values[instr->operand(0)]);

    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled opcode for IR emission: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

void CpuKernelEmitter::EmitOpInToApply(
    EmitterLocOpBuilder& b,
    absl::flat_hash_map<const HloInstruction*, mlir::Value>& values,
    const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kParameter) return;

  absl::StatusOr<mlir::Value> result;
  if (instr->opcode() == HloOpcode::kConstant) {
    result = EmitConstantOp(instr, b);
  } else {
    result = EmitOp(instr, b, values);
  }

  CHECK(result.ok()) << result.status();
  values[instr] = result.value();
}

}  // namespace zkx::cpu
