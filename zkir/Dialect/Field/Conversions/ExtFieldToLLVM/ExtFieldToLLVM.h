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

#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_EXTFIELDTOLLVM_EXTFIELDTOLLVM_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_EXTFIELDTOLLVM_EXTFIELDTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// IWYU pragma: begin_keep
// Headers needed for ExtFieldToLLVM.h.inc
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

class DialectRegistry;
class RewritePatternSet;
class Pass;

namespace mlir::zkir::field {

#define GEN_PASS_DECL
#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h.inc" // NOLINT(build/include)

// Populate the type conversion for ExtField to LLVM.
void populateExtFieldToLLVMTypeConversion(LLVMTypeConverter &typeConverter);

// Populate the given list with patterns that convert from ExtField to
// LLVM.
void populateExtFieldToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);

void registerConvertExtFieldToLLVMInterface(DialectRegistry &registry);

} // namespace mlir::zkir::field

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_FIELD_CONVERSIONS_EXTFIELDTOLLVM_EXTFIELDTOLLVM_H_
