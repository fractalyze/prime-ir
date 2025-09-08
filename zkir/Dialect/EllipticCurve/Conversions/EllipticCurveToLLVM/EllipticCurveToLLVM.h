#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOLLVM_ELLIPTICCURVETOLLVM_H_

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
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

class AffineStructBuilder : public StructBuilder {
public:
  // Construct a helper for the given point value.
  using StructBuilder::StructBuilder;
  // Build IR creating an `undef` value of the affine type.
  static AffineStructBuilder poison(OpBuilder &builder, Location loc,
                                    Type type);

  // Build IR extracting the `x` value from the affine struct.
  Value x(OpBuilder &builder, Location loc);
  // Build IR inserting the `x` value into the point struct.
  void setX(OpBuilder &builder, Location loc, Value x);

  // Build IR extracting the `y` value from the affine struct.
  Value y(OpBuilder &builder, Location loc);
  // Build IR inserting the `y` value into the affine struct.
  void setY(OpBuilder &builder, Location loc, Value y);
};

class JacobianStructBuilder : public StructBuilder {
public:
  // Construct a helper for the given point value.
  using StructBuilder::StructBuilder;
  // Build IR creating an `undef` value of the jacobian type.
  static JacobianStructBuilder poison(OpBuilder &builder, Location loc,
                                      Type type);

  // Build IR extracting the `x` value from the jacobian struct.
  Value x(OpBuilder &builder, Location loc);
  // Build IR inserting the `x` value into the jacobian struct.
  void setX(OpBuilder &builder, Location loc, Value x);

  // Build IR extracting the `y` value from the jacobian struct.
  Value y(OpBuilder &builder, Location loc);
  // Build IR inserting the `y` value into the jacobian struct.
  void setY(OpBuilder &builder, Location loc, Value y);

  // Build IR extracting the `z` value from the jacobian struct.
  Value z(OpBuilder &builder, Location loc);
  // Build IR inserting the `z` value into the jacobian struct.
  void setZ(OpBuilder &builder, Location loc, Value z);
};

class XYZZStructBuilder : public StructBuilder {
public:
  // Construct a helper for the given point value.
  using StructBuilder::StructBuilder;
  // Build IR creating an `undef` value of the xyzz type.
  static XYZZStructBuilder poison(OpBuilder &builder, Location loc, Type type);

  // Build IR extracting the `x` value from the xyzz struct.
  Value x(OpBuilder &builder, Location loc);
  // Build IR inserting the `x` value into the xyzz struct.
  void setX(OpBuilder &builder, Location loc, Value x);

  // Build IR extracting the `y` value from the xyzz struct.
  Value y(OpBuilder &builder, Location loc);
  // Build IR inserting the `y` value into the xyzz struct.
  void setY(OpBuilder &builder, Location loc, Value y);

  // Build IR extracting the `z²` value from the xyzz struct.
  Value zz(OpBuilder &builder, Location loc);
  // Build IR inserting the `z²` value into the xyzz struct.
  void setZz(OpBuilder &builder, Location loc, Value zz);

  // Build IR extracting the `z³` value from the xyzz struct.
  Value zzz(OpBuilder &builder, Location loc);
  // Build IR inserting the `z³` value into the xyzz struct.
  void setZzz(OpBuilder &builder, Location loc, Value zzz);
};

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
