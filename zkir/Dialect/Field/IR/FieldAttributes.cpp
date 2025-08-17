#include "zkir/Dialect/Field/IR/FieldAttributes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::field {

LogicalResult PrimeFieldAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    PrimeFieldType type, IntegerAttr _value) {
  APInt modulus = type.getModulus().getValue();
  APInt value = _value.getValue();

  // check if storage type is same
  if (modulus.getBitWidth() != value.getBitWidth()) {
    emitError()
        << "prime field modulus bitwidth does not match the value bitwidth";
    return failure();
  }

  // check if value is in the field defined by modulus
  if (value.uge(modulus)) {
    emitError() << value.getZExtValue()
                << " is not in the field defined by modulus "
                << modulus.getZExtValue();
    return failure();
  }

  return success();
}

LogicalResult RootOfUnityAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    PrimeFieldAttr root, IntegerAttr degree) {
  if (root.getType().isMontgomery()) {
    // NOTE(batzor): Montgomery form is not supported for root of unity because
    // verification logic assumes standard form. Also, `PrimitiveRootAttr` in
    // the `Poly` dialect should also handle it if we want to allow this in the
    // future.
    emitError() << "root of unity must be in standard form";
    return failure();
  }
  APInt modulus = root.getType().getModulus().getValue();
  APInt rootOfUnity = root.getValue().getValue();
  unsigned degreeValue = degree.getValue().getZExtValue();

  if (!expMod(rootOfUnity, degreeValue, modulus).isOne()) {
    emitError() << rootOfUnity.getZExtValue()
                << " is not a root of unity of degree " << degreeValue;
    return failure();
  }

  return success();
}

} // namespace mlir::zkir::field
