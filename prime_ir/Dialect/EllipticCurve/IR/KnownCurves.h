/* Copyright 2025 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_IR_KNOWNCURVES_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_IR_KNOWNCURVES_H_

#include <optional>
#include <string>

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"

namespace mlir::prime_ir::elliptic_curve {

std::optional<std::string> getKnownCurveAlias(ShortWeierstrassAttr attr);

// Supported pairing-friendly curve families.
enum class PairingCurveFamily {
  kBN254,
};

// Identifies the pairing curve family from a pair of G1/G2 curve attributes.
// Returns nullopt if the curves are not a recognized pairing-friendly pair.
std::optional<PairingCurveFamily>
getKnownPairingCurveFamily(ShortWeierstrassAttr g1Attr,
                           ShortWeierstrassAttr g2Attr);

// Builds the BN254 Fp12 tower extension type: Fp12 = (Fp6)² = ((Fp2)³)²
// The isMontgomery flag determines the field representation form.
field::ExtensionFieldType buildBN254Fp12Type(MLIRContext *ctx,
                                             bool isMontgomery);

// Builds the BN254 Fp6 tower extension type: Fp6 = (Fp2)³
field::ExtensionFieldType buildBN254Fp6Type(MLIRContext *ctx,
                                            bool isMontgomery);

// Builds the BN254 Fp2 extension type: Fp2 = Fp[u]/(u² + 1)
field::ExtensionFieldType buildBN254Fp2Type(MLIRContext *ctx,
                                            bool isMontgomery);

} // namespace mlir::prime_ir::elliptic_curve

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_IR_KNOWNCURVES_H_
