/* Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_CODEGEN_EMITTERS_HLO_TO_MLIR_OP_MAP_H_
#define ZKX_CODEGEN_EMITTERS_HLO_TO_MLIR_OP_MAP_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkx/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"

namespace zkx::emitters {

// Bridge from HloOpcode to mhlo template instantiation.
// Reuses map_mhlo_to_scalar_op.h for type-dispatched scalar op creation.
//
// Returns a single Value. Callers that need SmallVector<Value, 1> (e.g.
// elemental_hlo_to_mlir) wrap at the return site with {{...}}.
//
// Accepts ImplicitLocOpBuilder& so both elemental_hlo_to_mlir (which uses
// ImplicitLocOpBuilder directly) and cpu_kernel_emitter (which uses
// EmitterLocOpBuilder, a subclass) can share this implementation.
template <typename MhloOp, typename... ExtraArgs>
mlir::Value MapHloOp(mlir::Type result_type,
                     llvm::ArrayRef<mlir::Type> arg_types,
                     llvm::ArrayRef<mlir::Value> args,
                     llvm::ArrayRef<mlir::NamedAttribute> attributes,
                     mlir::ImplicitLocOpBuilder& b, ExtraArgs&&... extra_args) {
  if constexpr (std::is_same_v<MhloOp, mlir::mhlo::AddOp> ||
                std::is_same_v<MhloOp, mlir::mhlo::SubtractOp> ||
                std::is_same_v<MhloOp, mlir::mhlo::MulOp>) {
    // In case of affine points, we convert the result type to Jacobian points.
    // Affine + Affine -> Jacobian
    // Affine - Affine -> Jacobian
    // Affine * ScalarField -> Jacobian
    // Handle both scalar and tensor-wrapped affine types.
    auto element_type = mlir::getElementTypeOrSelf(result_type);
    if (auto affine_type =
            mlir::dyn_cast<mlir::prime_ir::elliptic_curve::AffineType>(
                element_type)) {
      auto jacobian_type = mlir::prime_ir::elliptic_curve::JacobianType::get(
          b.getContext(), affine_type.getCurve());
      if (auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(result_type)) {
        result_type = shaped_type.clone(jacobian_type);
      } else {
        result_type = jacobian_type;
      }
    }
  }
  return mlir::mhlo::MhloOpToStdScalarOp::mapOpOfType<MhloOp>(
      b.getLoc(), result_type, arg_types,
      typename MhloOp::Adaptor(args, std::forward<ExtraArgs>(extra_args)...),
      attributes, &b);
}

// Convenience wrapper that uses the last arg's type as the result type.
// This works for most elementwise ops; select uses the last arg (true_value)
// type, which is correct.
template <typename MhloOp>
mlir::Value MapElementwiseOp(
    llvm::ArrayRef<mlir::Type> arg_types, llvm::ArrayRef<mlir::Value> args,
    mlir::ImplicitLocOpBuilder& b,
    llvm::ArrayRef<mlir::NamedAttribute> attributes = std::nullopt) {
  // We use the last argument's type because of select.
  return MapHloOp<MhloOp>(args.back().getType(), arg_types, args, attributes,
                          b);
}

}  // namespace zkx::emitters

#endif  // ZKX_CODEGEN_EMITTERS_HLO_TO_MLIR_OP_MAP_H_
