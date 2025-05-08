#include "zkir/Utils/OpUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::zkir {

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto shapedType = llvm::dyn_cast<ShapedType>(type))
    return shapedType.cloneWith(std::nullopt, i1Type);
  if (llvm::isa<UnrankedTensorType>(type))
    return UnrankedTensorType::get(i1Type);
  return i1Type;
}

}  // namespace mlir::zkir
