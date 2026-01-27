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

#include "prime_ir/Utils/BitSerialAlgorithm.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::prime_ir {

Value generateBitSerialLoop(ImplicitLocOpBuilder &b, Value scalar, Value base,
                            Value identity, DoubleCallback doubleOp,
                            AccumulateCallback accumulateOp) {
  Type scalarType = scalar.getType();
  Type elementType = base.getType();

  Value zero = b.create<arith::ConstantIntOp>(scalarType, 0);
  Value one = b.create<arith::ConstantIntOp>(scalarType, 1);

  // Handle the first bit before entering the loop to avoid an extra iteration.
  auto ifOp = b.create<scf::IfOp>(
      b.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                              b.create<arith::AndIOp>(scalar, one), zero),
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder innerB(loc, builder);
        Value newResult = accumulateOp(innerB, identity, base);
        innerB.create<scf::YieldOp>(ValueRange{newResult});
      },
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder innerB(loc, builder);
        innerB.create<scf::YieldOp>(ValueRange{identity});
      });

  scalar = b.create<arith::ShRUIOp>(scalar, one);
  Value result = ifOp.getResult(0);

  // Main loop: while (scalar > 0)
  auto whileOp = b.create<scf::WhileOp>(
      TypeRange{scalarType, elementType, elementType},
      ValueRange{scalar, base, result},
      // Before block: condition check
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder innerB(loc, builder);
        auto cond = innerB.create<arith::CmpIOp>(arith::CmpIPredicate::ugt,
                                                 args[0], zero);
        innerB.create<scf::ConditionOp>(cond, args);
      },
      // After block: loop body
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        ImplicitLocOpBuilder innerB(loc, builder);
        Value currScalar = args[0];
        Value currBase = args[1];
        Value currResult = args[2];

        // Double the base
        Value newBase = doubleOp(innerB, currBase);

        // Check if current bit is set
        Value masked = innerB.create<arith::AndIOp>(currScalar, one);
        Value isSet = innerB.create<arith::CmpIOp>(arith::CmpIPredicate::ne,
                                                   masked, zero);

        // Conditionally accumulate
        auto innerIfOp = innerB.create<scf::IfOp>(
            isSet,
            [&](OpBuilder &thenBuilder, Location thenLoc) {
              ImplicitLocOpBuilder thenB(thenLoc, thenBuilder);
              Value newResult = accumulateOp(thenB, currResult, newBase);
              thenB.create<scf::YieldOp>(ValueRange{newResult});
            },
            [&](OpBuilder &elseBuilder, Location elseLoc) {
              ImplicitLocOpBuilder elseB(elseLoc, elseBuilder);
              elseB.create<scf::YieldOp>(ValueRange{currResult});
            });

        // Right shift scalar
        Value shifted = innerB.create<arith::ShRUIOp>(currScalar, one);
        innerB.create<scf::YieldOp>(
            ValueRange{shifted, newBase, innerIfOp.getResult(0)});
      });

  return whileOp.getResult(2);
}

} // namespace mlir::prime_ir
