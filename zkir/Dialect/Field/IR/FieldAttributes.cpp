#include "zkir/Dialect/Field/IR/FieldAttributes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::field {

LogicalResult PrimeFieldAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    PrimeFieldType type, IntegerAttr value) {
  if (type.getModulus().getValue().getBitWidth() !=
      value.getValue().getBitWidth()) {
    emitError()
        << "prime field modulus bitwidth does not match the value bitwidth";
    return failure();
  }
  return success();
}

LogicalResult RootOfUnityAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    PrimeFieldAttr root, IntegerAttr degree) {
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

}  // namespace mlir::zkir::field
