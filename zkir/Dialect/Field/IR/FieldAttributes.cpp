#include "zkir/Dialect/Field/IR/FieldAttributes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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

}  // namespace mlir::zkir::field
