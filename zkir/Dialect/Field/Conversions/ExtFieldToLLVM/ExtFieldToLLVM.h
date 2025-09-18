#ifndef ZKIR_DIALECT_FIELD_CONVERSIONS_EXTFIELDTOLLVM_EXTFIELDTOLLVM_H_
#define ZKIR_DIALECT_FIELD_CONVERSIONS_EXTFIELDTOLLVM_EXTFIELDTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// IWYU pragma: begin_keep
// Headers needed for ExtFieldToLLVM.h.inc
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
