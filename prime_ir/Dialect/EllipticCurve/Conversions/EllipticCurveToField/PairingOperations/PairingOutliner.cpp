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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingOutliner.h"

#include "llvm/ADT/Twine.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingFieldCodeGen.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::elliptic_curve {

std::string PairingOutliner::mangleName(StringRef baseName,
                                        field::ExtensionFieldType type) {
  field::PrimeFieldType baseField = type.getBasePrimeField();
  APInt modulus = baseField.getModulus().getValue();
  uint64_t modulusHash = modulus.getLimitedValue();
  if (modulus.getBitWidth() > 64)
    modulusHash ^= modulus.extractBits(64, 64).getLimitedValue();
  return ("__prime_ir_" + baseName + "_" + Twine(type.getDegreeOverPrime()) +
          "_" + Twine(modulusHash))
      .str();
}

Value PairingOutliner::emitCyclotomicSquareCall(ImplicitLocOpBuilder &b,
                                                Value input) {
  auto fp12Type = cast<field::ExtensionFieldType>(input.getType());
  std::string funcName = mangleName("cyclotomic_square", fp12Type);

  auto func = getOrCreateFunction(
      funcName, {fp12Type}, {fp12Type}, [&](func::FuncOp func) {
        OpBuilder builder(func.getContext());
        auto args = setupFunctionBody(func, builder);

        ImplicitLocOpBuilder bodyBuilder(func.getLoc(), builder);
        ScopedBuilderContext scope(&bodyBuilder);
        BodyGenGuard guard(*this);

        PairingFp12CodeGen inp(args[0]);
        PairingFp12CodeGen result = inp.CyclotomicSquare();

        emitReturn(builder, func.getLoc(), static_cast<Value>(result));
      });

  return emitCall(b, b.getLoc(), func, {input});
}

Value PairingOutliner::emitFp12MulCall(ImplicitLocOpBuilder &b, Value lhs,
                                       Value rhs) {
  auto fp12Type = cast<field::ExtensionFieldType>(lhs.getType());
  std::string funcName = mangleName("fp12_mul", fp12Type);

  auto func = getOrCreateFunction(
      funcName, {fp12Type, fp12Type}, {fp12Type}, [&](func::FuncOp func) {
        OpBuilder builder(func.getContext());
        auto args = setupFunctionBody(func, builder);
        Value result = builder.create<field::MulOp>(func.getLoc(), fp12Type,
                                                    args[0], args[1]);
        emitReturn(builder, func.getLoc(), result);
      });

  return emitCall(b, b.getLoc(), func, {lhs, rhs});
}

Value PairingOutliner::emitMulBy034Call(ImplicitLocOpBuilder &b, Value fp12,
                                        Value c0, Value c3, Value c4) {
  auto fp12Type = cast<field::ExtensionFieldType>(fp12.getType());
  auto fp2Type = cast<field::ExtensionFieldType>(c0.getType());
  std::string funcName = mangleName("mul_by_034", fp12Type);

  auto func = getOrCreateFunction(
      funcName, {fp12Type, fp2Type, fp2Type, fp2Type}, {fp12Type},
      [&](func::FuncOp func) {
        OpBuilder builder(func.getContext());
        auto args = setupFunctionBody(func, builder);

        ImplicitLocOpBuilder bodyBuilder(func.getLoc(), builder);
        ScopedBuilderContext scope(&bodyBuilder);
        BodyGenGuard guard(*this);

        PairingFp12CodeGen inp(args[0]);
        PairingFp2CodeGen beta0(args[1]), beta3(args[2]), beta4(args[3]);
        PairingFp12CodeGen result = inp.MulBy034(beta0, beta3, beta4);

        emitReturn(builder, func.getLoc(), static_cast<Value>(result));
      });

  return emitCall(b, b.getLoc(), func, {fp12, c0, c3, c4});
}

Value PairingOutliner::emitMulBy014Call(ImplicitLocOpBuilder &b, Value fp12,
                                        Value c0, Value c1, Value c4) {
  auto fp12Type = cast<field::ExtensionFieldType>(fp12.getType());
  auto fp2Type = cast<field::ExtensionFieldType>(c0.getType());
  std::string funcName = mangleName("mul_by_014", fp12Type);

  auto func = getOrCreateFunction(
      funcName, {fp12Type, fp2Type, fp2Type, fp2Type}, {fp12Type},
      [&](func::FuncOp func) {
        OpBuilder builder(func.getContext());
        auto args = setupFunctionBody(func, builder);

        ImplicitLocOpBuilder bodyBuilder(func.getLoc(), builder);
        ScopedBuilderContext scope(&bodyBuilder);
        BodyGenGuard guard(*this);

        PairingFp12CodeGen inp(args[0]);
        PairingFp2CodeGen beta0(args[1]), beta1(args[2]), beta4(args[3]);
        PairingFp12CodeGen result = inp.MulBy014(beta0, beta1, beta4);

        emitReturn(builder, func.getLoc(), static_cast<Value>(result));
      });

  return emitCall(b, b.getLoc(), func, {fp12, c0, c1, c4});
}

} // namespace mlir::prime_ir::elliptic_curve
