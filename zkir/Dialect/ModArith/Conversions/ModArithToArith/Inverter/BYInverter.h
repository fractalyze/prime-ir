/* Copyright 2025 The ZKIR Authors.

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

#ifndef ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_INVERTER_BYINVERTER_H_
#define ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_INVERTER_BYINVERTER_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

namespace mlir::zkir::mod_arith {

class BYInverter {
public:
  struct DEResult {
    Value d;
    Value e;
  };

  struct FGResult {
    Value f;
    Value g;
  };

  struct TMatrix {
    Value t00;
    Value t01;
    Value t10;
    Value t11;
  };

  struct JumpResult {
    TMatrix t;
    Value eta;
  };

  BYInverter(ImplicitLocOpBuilder &b, Type inputType);

  Value Generate(Value input, bool isMont);
  Value BatchGenerate(Value input, bool isMont, ShapedType shapedType);
  JumpResult GenerateJump(Value f, Value g, Value eta);
  FGResult GenerateFG(Value f, Value g, TMatrix t);
  DEResult GenerateDE(Value d, Value e, TMatrix t);
  Value GenerateNorm(Value value, Value antiunit);

private:
  ImplicitLocOpBuilder &b_;

  IntegerType intType_;
  IntegerType extIntType_;
  IntegerType limbType_;
  ModArithType modArithType_;

  Value maskN_;
  Value m_;
  Value mInv_;
  Value limbTypeOne_;
  Value limbTypeZero_;
  Value extIntTypeOne_;
  Value extIntTypeZero_;
  Value extIntTypeN_;
  Value limbTypeN_;
};

} // namespace mlir::zkir::mod_arith

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_INVERTER_BYINVERTER_H_
