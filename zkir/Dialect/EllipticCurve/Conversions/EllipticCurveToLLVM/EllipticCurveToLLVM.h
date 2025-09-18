#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// IWYU pragma: begin_keep
// Headers needed for EllipticCurveToLLVM.h.inc
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

class DialectRegistry;
class RewritePatternSet;
class Pass;

namespace mlir::zkir::elliptic_curve {

#define GEN_PASS_DECL
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h.inc" // NOLINT(build/include)

// Populate the type conversion for EllipticCurve to LLVM.
void populateEllipticCurveToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter);

// Populate the given list with patterns that convert from EllipticCurve to
// LLVM.
void populateEllipticCurveToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);

void registerConvertEllipticCurveToLLVMInterface(DialectRegistry &registry);

} // namespace mlir::zkir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_H_
