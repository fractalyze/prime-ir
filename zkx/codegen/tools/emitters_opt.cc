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

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "zkx/backends/gpu/codegen/emitters/emitter_base.h"
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.h"
#include "zkx/backends/gpu/codegen/emitters/transforms/passes.h"
#include "zkx/codegen/emitters/ir/zkx_ops.h"
#include "zkx/codegen/emitters/transforms/passes.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "zkx/service/gpu/gpu_device_info_for_tests.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<
      // clang-format off
      mlir::affine::AffineDialect,
      mlir::arith::ArithDialect,
      mlir::DLTIDialect,
      mlir::func::FuncDialect,
      mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect,
      mlir::math::MathDialect,
      mlir::mhlo::MhloDialect,
      mlir::NVVM::NVVMDialect,
      mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect,
      mlir::vector::VectorDialect,
      zkx::ZkxDialect,
      // TODO(chokobole): Uncomment this. Dependency: zkx::cpu::ZkxCpuDialect
    //   zkx::cpu::ZkxCpuDialect,
      zkx::gpu::ZkxGpuDialect,
      mlir::prime_ir::field::FieldDialect
      // clang-format on
      >();
  mlir::func::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerInliner();
  zkx::emitters::registerTransformsPasses();
  zkx::gpu::registerGpuFusionTransformsPasses();
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: zkx::cpu::registerZkxCpuTransformsPasses
  // clang-format on
  // zkx::cpu::registerZkxCpuTransformsPasses();
  mlir::registerPassPipeline(
      "zkx-gpu-test-optimize",
      "Test pipeline of passes up to inlining. No vectorization, also does not "
      "lower zkx_gpu.Intended to simplify IR in tests.",
      [=](mlir::OpPassManager& pm, llvm::StringRef options,
          llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)>
              errorHandler) {
        if (!options.empty()) return mlir::failure();

        zkx::gpu::AddZkxGpuOpsOptimizationPasses(pm);
        return mlir::success();
      },
      [](llvm::function_ref<void(const mlir::detail::PassOptions&)>) {});
  mlir::registerPassPipeline(
      "zkx-gpu-test-transform-loops",
      "Test pipeline for vectorization. Should run after "
      "zkx-gpu-test-to-inline.",
      [=](mlir::OpPassManager& pm, llvm::StringRef options,
          llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)>
              errorHandler) {
        if (!options.empty()) return mlir::failure();
        zkx::gpu::AddLoopTransformationPasses(
            pm, zkx::gpu::TestGpuDeviceInfo::RTXA6000DeviceInfo());
        return mlir::success();
      },
      [](llvm::function_ref<void(const mlir::detail::PassOptions&)>) {});

  return mlir::failed(
      MlirOptMain(argc, argv, "ZKX Emitters Pass Driver\n", registry));
}
