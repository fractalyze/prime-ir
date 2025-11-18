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

unsigned getIntOrPrimeFieldBitWidth(Type type) {
  assert(llvm::isa<PrimeFieldType>(type) || llvm::isa<IntegerType>(type));
  if (auto pfType = dyn_cast<PrimeFieldType>(type)) {
    return pfType.getStorageBitWidth();
  }
  return cast<IntegerType>(type).getWidth();
}

llvm::TypeSize PrimeFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getStorageBitWidth());
}

uint64_t PrimeFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getStorageType());
}

llvm::TypeSize QuadraticExtFieldType::getTypeSizeInBits(
    DataLayout const &, llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getBaseField().getStorageBitWidth() * 2);
}

uint64_t QuadraticExtFieldType::getABIAlignment(
    DataLayout const &dataLayout,
    llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getBaseField().getStorageType());
}
} // namespace mlir::zkir::field
