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

#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/IntrinsicFunctionGenerator.h"

#include "llvm/ADT/Twine.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"

namespace mlir::prime_ir::field {

bool IntrinsicFunctionGenerator::shouldUseIntrinsic(Operation *op,
                                                    Type fieldType,
                                                    LoweringMode mode) {
  if (mode == LoweringMode::Inline)
    return false;

  if (isInsideIntrinsicFunction(op))
    return false;

  auto efType = dyn_cast<ExtensionFieldType>(fieldType);
  if (!efType)
    return false;

  // Both Intrinsic and Auto modes use intrinsic for quartic (degree >= 4)
  return efType.getDegree() >= 4;
}

std::string
IntrinsicFunctionGenerator::mangleFunctionName(StringRef baseName,
                                               ExtensionFieldType type) {
  // Create a unique name based on the type
  // Format: __prime_ir_<baseName>_<degree>_<modulus_hash>
  auto baseField = cast<PrimeFieldType>(type.getBaseField());
  APInt modulus = baseField.getModulus().getValue();

  // Use a simple hash of the modulus to keep names manageable
  uint64_t modulusHash = modulus.getLimitedValue();
  if (modulus.getBitWidth() > 64) {
    // For larger moduli, use a combination of high and low bits
    modulusHash ^= modulus.extractBits(64, 64).getLimitedValue();
  }

  return ("__prime_ir_" + baseName + "_" + Twine(type.getDegree()) + "_" +
          Twine(modulusHash))
      .str();
}

func::FuncOp IntrinsicFunctionGenerator::getOrCreateQuarticMulFunction(
    ExtensionFieldType type) {
  return getOrCreateFunction(mangleFunctionName("ext4_mul", type), {type, type},
                             {type}, [&](func::FuncOp func) {
                               OpBuilder builder(func.getContext());
                               auto args = setupFunctionBody(func, builder);
                               Value result = builder.create<MulOp>(
                                   func.getLoc(), type, args[0], args[1]);
                               emitReturn(builder, func.getLoc(), result);
                             });
}

func::FuncOp IntrinsicFunctionGenerator::getOrCreateQuarticSquareFunction(
    ExtensionFieldType type) {
  return getOrCreateFunction(mangleFunctionName("ext4_square", type), {type},
                             {type}, [&](func::FuncOp func) {
                               OpBuilder builder(func.getContext());
                               auto args = setupFunctionBody(func, builder);
                               Value result = builder.create<SquareOp>(
                                   func.getLoc(), type, args[0]);
                               emitReturn(builder, func.getLoc(), result);
                             });
}

func::FuncOp IntrinsicFunctionGenerator::getOrCreateQuarticInverseFunction(
    ExtensionFieldType type) {
  return getOrCreateFunction(mangleFunctionName("ext4_inverse", type), {type},
                             {type}, [&](func::FuncOp func) {
                               OpBuilder builder(func.getContext());
                               auto args = setupFunctionBody(func, builder);
                               Value result = builder.create<InverseOp>(
                                   func.getLoc(), type, args[0]);
                               emitReturn(builder, func.getLoc(), result);
                             });
}

Value IntrinsicFunctionGenerator::emitQuarticMulCall(OpBuilder &builder,
                                                     Location loc,
                                                     ExtensionFieldType type,
                                                     Value lhs, Value rhs) {
  return emitCall(builder, loc, getOrCreateQuarticMulFunction(type),
                  {lhs, rhs});
}

Value IntrinsicFunctionGenerator::emitQuarticSquareCall(OpBuilder &builder,
                                                        Location loc,
                                                        ExtensionFieldType type,
                                                        Value input) {
  return emitCall(builder, loc, getOrCreateQuarticSquareFunction(type),
                  {input});
}

Value IntrinsicFunctionGenerator::emitQuarticInverseCall(
    OpBuilder &builder, Location loc, ExtensionFieldType type, Value input) {
  return emitCall(builder, loc, getOrCreateQuarticInverseFunction(type),
                  {input});
}

} // namespace mlir::prime_ir::field
