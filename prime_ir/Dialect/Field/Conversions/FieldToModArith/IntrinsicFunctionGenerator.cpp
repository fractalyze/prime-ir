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
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::field {

bool IntrinsicFunctionGenerator::shouldUseIntrinsic(Operation *op,
                                                    Type fieldType,
                                                    LoweringMode mode) {
  if (mode == LoweringMode::Inline)
    return false;

  auto efType = dyn_cast<ExtensionFieldType>(fieldType);
  if (!efType)
    return false;

  if (efType.getDegreeOverPrime() < 2)
    return false;

  // Only outline for large prime fields (>64-bit) where function call overhead
  // is negligible. Small fields (BabyBear=31bit, Goldilocks=64bit) benefit
  // more from inlining for LLVM cross-operation optimization.
  if (efType.getBasePrimeField().getStorageBitWidth() <= 64)
    return false;

  // Inside an intrinsic function, only use intrinsics for strictly lower-degree
  // extension fields to enable multi-level call chains (e.g., Fp12 -> Fp6 ->
  // Fp2) while preventing self-recursive calls.
  if (auto parentFunc = op->getParentOfType<func::FuncOp>()) {
    if (parentFunc.getName().starts_with("__prime_ir_")) {
      auto parentInputTypes = parentFunc.getArgumentTypes();
      if (!parentInputTypes.empty()) {
        if (auto parentEfType =
                dyn_cast<ExtensionFieldType>(parentInputTypes[0])) {
          if (efType.getDegreeOverPrime() >= parentEfType.getDegreeOverPrime())
            return false;
        }
      }
    }
  }

  return true;
}

std::string
IntrinsicFunctionGenerator::mangleFunctionName(StringRef baseName,
                                               ExtensionFieldType type) {
  // Format: __prime_ir_<baseName>_<degreeOverPrime>_<modulus_hash>
  // Use getBasePrimeField() to handle tower extensions (e.g., Fp6 = 3x Fp2).
  PrimeFieldType baseField = type.getBasePrimeField();
  APInt modulus = baseField.getModulus().getValue();

  uint64_t modulusHash = modulus.getLimitedValue();
  if (modulus.getBitWidth() > 64) {
    modulusHash ^= modulus.extractBits(64, 64).getLimitedValue();
  }

  return ("__prime_ir_" + baseName + "_" + Twine(type.getDegreeOverPrime()) +
          "_" + Twine(modulusHash))
      .str();
}

void IntrinsicFunctionGenerator::preCreateIntrinsicsForTower(
    ExtensionFieldType type) {
  if (type.getDegreeOverPrime() < 2)
    return;

  // Create intrinsics for this level (bottom-up: sub-levels first)
  if (auto subEfType = dyn_cast<ExtensionFieldType>(type.getBaseField()))
    preCreateIntrinsicsForTower(subEfType);

  getOrCreateAddFunction(type);
  getOrCreateSubFunction(type);
  getOrCreateNegateFunction(type);
  getOrCreateDoubleFunction(type);
  getOrCreateQuarticMulFunction(type);
  getOrCreateQuarticSquareFunction(type);
  getOrCreateQuarticInverseFunction(type);
}

func::FuncOp
IntrinsicFunctionGenerator::getOrCreateAddFunction(ExtensionFieldType type) {
  return getOrCreateFunction(mangleFunctionName("ext4_add", type), {type, type},
                             {type}, [&](func::FuncOp func) {
                               OpBuilder builder(func.getContext());
                               auto args = setupFunctionBody(func, builder);
                               Value result = builder.create<AddOp>(
                                   func.getLoc(), type, args[0], args[1]);
                               emitReturn(builder, func.getLoc(), result);
                             });
}

func::FuncOp
IntrinsicFunctionGenerator::getOrCreateSubFunction(ExtensionFieldType type) {
  return getOrCreateFunction(mangleFunctionName("ext4_sub", type), {type, type},
                             {type}, [&](func::FuncOp func) {
                               OpBuilder builder(func.getContext());
                               auto args = setupFunctionBody(func, builder);
                               Value result = builder.create<SubOp>(
                                   func.getLoc(), type, args[0], args[1]);
                               emitReturn(builder, func.getLoc(), result);
                             });
}

func::FuncOp
IntrinsicFunctionGenerator::getOrCreateNegateFunction(ExtensionFieldType type) {
  return getOrCreateFunction(mangleFunctionName("ext4_negate", type), {type},
                             {type}, [&](func::FuncOp func) {
                               OpBuilder builder(func.getContext());
                               auto args = setupFunctionBody(func, builder);
                               Value result = builder.create<NegateOp>(
                                   func.getLoc(), type, args[0]);
                               emitReturn(builder, func.getLoc(), result);
                             });
}

Value IntrinsicFunctionGenerator::emitAddCall(OpBuilder &builder, Location loc,
                                              ExtensionFieldType type,
                                              Value lhs, Value rhs) {
  return emitCall(builder, loc, getOrCreateAddFunction(type), {lhs, rhs});
}

Value IntrinsicFunctionGenerator::emitSubCall(OpBuilder &builder, Location loc,
                                              ExtensionFieldType type,
                                              Value lhs, Value rhs) {
  return emitCall(builder, loc, getOrCreateSubFunction(type), {lhs, rhs});
}

Value IntrinsicFunctionGenerator::emitNegateCall(OpBuilder &builder,
                                                 Location loc,
                                                 ExtensionFieldType type,
                                                 Value input) {
  return emitCall(builder, loc, getOrCreateNegateFunction(type), {input});
}

func::FuncOp
IntrinsicFunctionGenerator::getOrCreateDoubleFunction(ExtensionFieldType type) {
  return getOrCreateFunction(mangleFunctionName("ext4_double", type), {type},
                             {type}, [&](func::FuncOp func) {
                               OpBuilder builder(func.getContext());
                               auto args = setupFunctionBody(func, builder);
                               Value result = builder.create<DoubleOp>(
                                   func.getLoc(), type, args[0]);
                               emitReturn(builder, func.getLoc(), result);
                             });
}

Value IntrinsicFunctionGenerator::emitDoubleCall(OpBuilder &builder,
                                                 Location loc,
                                                 ExtensionFieldType type,
                                                 Value input) {
  return emitCall(builder, loc, getOrCreateDoubleFunction(type), {input});
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
