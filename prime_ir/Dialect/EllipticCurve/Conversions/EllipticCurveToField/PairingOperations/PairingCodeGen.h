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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGCODEGEN_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGCODEGEN_H_

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingFieldCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/IR/KnownCurves.h"

namespace mlir::prime_ir::elliptic_curve {

// Emits the pairing check IR for a BN254 pairing.
//
// Given G1 and G2 point tensors, generates the field operations for:
//   1. G2 precomputation (line coefficient computation)
//   2. Multi-Miller loop
//   3. Final exponentiation
//   4. Comparison to Fp12 identity
//
// Returns an i1 value (true if pairing check passes).
Value emitBN254PairingCheck(ImplicitLocOpBuilder &builder,
                            PairingCurveFamily family, Value g1Points,
                            Value g2Points, bool isMontgomery);

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_PAIRINGOPERATIONS_PAIRINGCODEGEN_H_
