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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// IWYU pragma: begin_keep
// Headers needed for EllipticCurveToLLVM.h.inc
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

class DialectRegistry;
class RewritePatternSet;
class Pass;

namespace mlir::prime_ir::elliptic_curve {

#define GEN_PASS_DECL
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h.inc"

#define GEN_PASS_REGISTRATION
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h.inc" // NOLINT(build/include)

// Populate the type conversion for EllipticCurve to LLVM.
void populateEllipticCurveToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter);

// Populate the given list with patterns that convert from EllipticCurve to
// LLVM.
void populateEllipticCurveToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);

void registerConvertEllipticCurveToLLVMInterface(DialectRegistry &registry);

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_H_
