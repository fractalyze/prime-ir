/* Copyright 2026 The PrimeIR Authors.

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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/IntrinsicFunctionGenerator.h"

#include "llvm/ADT/Twine.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::elliptic_curve {

bool IntrinsicFunctionGenerator::shouldUseIntrinsic(Operation *op,
                                                    Type pointType,
                                                    LoweringMode mode) {
  if (mode == LoweringMode::Inline)
    return false;

  if (isInsideIntrinsicFunction(op))
    return false;

  // For Intrinsic mode, always use intrinsic for scalar_mul
  if (mode == LoweringMode::Intrinsic)
    return true;

  // Auto mode: use intrinsic for points over extension fields (G2, etc.)
  auto pointTypeInterface = dyn_cast<PointTypeInterface>(pointType);
  if (!pointTypeInterface)
    return false;

  Type baseFieldType = pointTypeInterface.getBaseFieldType();
  if (isa<field::ExtensionFieldType>(baseFieldType)) {
    // Use intrinsic for extension fields (G2 uses Fp2)
    return true;
  }

  // Prime field points (G1) - inline is usually fine
  return false;
}

std::string IntrinsicFunctionGenerator::mangleFunctionName(StringRef baseName,
                                                           Type pointType,
                                                           Type scalarType) {
  auto pointTypeInterface = cast<PointTypeInterface>(pointType);
  Type baseFieldType = pointTypeInterface.getBaseFieldType();
  PointKind kind = pointTypeInterface.getPointKind();

  // Build name: __prime_ir_ec_<baseName>_<pointKind>_<fieldInfo>
  std::string kindStr;
  switch (kind) {
  case PointKind::kAffine:
    kindStr = "affine";
    break;
  case PointKind::kJacobian:
    kindStr = "jacobian";
    break;
  case PointKind::kXYZZ:
    kindStr = "xyzz";
    break;
  }

  uint64_t fieldHash = 0;
  if (auto pfType = dyn_cast<field::PrimeFieldType>(baseFieldType)) {
    fieldHash = pfType.getModulus().getValue().getLimitedValue();
  } else if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType)) {
    auto baseField = cast<field::PrimeFieldType>(efType.getBaseField());
    fieldHash = baseField.getModulus().getValue().getLimitedValue();
    fieldHash ^= efType.getDegree() << 16;
  }

  // Include scalar type info (Montgomery vs standard form)
  std::string scalarSuffix;
  if (auto scalarPfType = dyn_cast<field::PrimeFieldType>(scalarType)) {
    scalarSuffix = scalarPfType.isMontgomery() ? "_mont" : "_std";
  }

  return ("__prime_ir_ec_" + baseName + "_" + kindStr + "_" + Twine(fieldHash) +
          scalarSuffix)
      .str();
}

func::FuncOp
IntrinsicFunctionGenerator::getOrCreateScalarMulFunction(Type pointType,
                                                         Type scalarType) {
  return getOrCreateFunction(
      mangleFunctionName("scalar_mul", pointType, scalarType),
      {pointType, scalarType}, {pointType}, [&](func::FuncOp func) {
        OpBuilder builder(func.getContext());
        auto args = setupFunctionBody(func, builder);
        // args[0] = point, args[1] = scalar
        Value result = builder.create<ScalarMulOp>(func.getLoc(), pointType,
                                                   args[1], args[0]);
        emitReturn(builder, func.getLoc(), result);
      });
}

Value IntrinsicFunctionGenerator::emitScalarMulCall(OpBuilder &builder,
                                                    Location loc,
                                                    Type pointType,
                                                    Type outputType,
                                                    Value point, Value scalar) {
  func::FuncOp func =
      getOrCreateScalarMulFunction(outputType, scalar.getType());

  // Convert point to output type if needed
  Value convertedPoint = point;
  if (point.getType() != outputType)
    convertedPoint = builder.create<ConvertPointTypeOp>(loc, outputType, point);

  return emitCall(builder, loc, func, {convertedPoint, scalar});
}

} // namespace mlir::prime_ir::elliptic_curve
