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

#ifndef PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_CIOSMONTEMITTER_H_
#define PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_CIOSMONTEMITTER_H_

#include <utility>

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h" // IWYU pragma: keep

namespace mlir {
class IntegerAttr;
} // namespace mlir

namespace mlir::prime_ir::mod_arith {

// Emits 32-bit-limb fused-CIOS Montgomery arithmetic for multi-limb moduli
// with at least one spare storage bit. Uses the same R = 2^storageBits as
// MontReducer's wide path, so results are interchangeable.
//
// Each limb multiply-accumulate is lowered to the LLVM IR shape
// `add i64 (mul i64 (zext i32 a), (zext i32 b)), %acc`, which selects a single
// IMAD.WIDE.U32 on NVIDIA targets (no native 64-bit IMAD, so the wide-path
// `arith.mului_extended` would scalarize). See CIOS in Koç, Acar, Kaliski,
// "Analyzing and Comparing Montgomery Multiplication Algorithms", IEEE Micro
// 1996 (https://eprint.iacr.org/2017/1057 reproduces the algorithm).
class CiosMontEmitter {
public:
  explicit CiosMontEmitter(ImplicitLocOpBuilder &b, ModArithType modArithType);

  // True when the CIOS path applies: multi-limb, scalar integer storage,
  // modulus sign bit clear. The `target == "gpu"` gate is the caller's.
  static bool isEligible(ModArithType modArithType, Type convertedType);

  // a * b * R⁻¹ mod n. lazy=true returns [0, 2p), else [0, p).
  Value emitMontMul(Value a, Value b, bool lazy);

private:
  SmallVector<Value> decompose(Value wide);            // iW -> N x i32
  Value recompose(ArrayRef<Value> limbs, Type wideTy); // N x i32 -> iW
  // s = (i64)a*b + t + c. Returns {lo32(s), hi32(s)}. THE IMAD.WIDE shape.
  std::pair<Value, Value> mulAddStep(Value a32, Value b32, Value t32,
                                     Value c32);
  // s = (i64)t + (i64)c. Returns {lo32(s), hi32(s)} for the overflow columns.
  std::pair<Value, Value> addCarryStep(Value t32, Value c32);

  Value i32Const(uint64_t v);

  ImplicitLocOpBuilder &b;
  ModArithType modArithType;
  IntegerAttr modAttr;
  MontgomeryAttr montAttr;
};

} // namespace mlir::prime_ir::mod_arith

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_REDUCER_CIOSMONTEMITTER_H_
