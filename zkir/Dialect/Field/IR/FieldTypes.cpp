#include "zkir/Dialect/Field/IR/FieldTypes.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::zkir::field {

bool isMontgomery(Type type) {
  Type element;
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    element = shapedType.getElementType();
  } else if (auto memrefType = dyn_cast<MemRefType>(type)) {
    element = memrefType.getElementType();
  } else {
    element = type;
  }
  if (auto pfType = dyn_cast<PrimeFieldType>(element)) {
    return pfType.isMontgomery();
  } else if (auto f2Type = dyn_cast<QuadraticExtFieldType>(element)) {
    return f2Type.isMontgomery();
  } else {
    return false;
  }
}

} // namespace mlir::zkir::field
