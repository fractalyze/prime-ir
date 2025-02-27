#include "zkir/Dialect/Poly/IR/PolyDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/Poly/IR/PolyAttributes.h"
#include "zkir/Dialect/Poly/IR/PolyOps.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"

// Generated definitions
#include "zkir/Dialect/Poly/IR/PolyDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/Poly/IR/PolyAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/Poly/IR/PolyTypes.cpp.inc"

#define GET_OP_CLASSES
#include "zkir/Dialect/Poly/IR/PolyOps.cpp.inc"

namespace mlir {
namespace zkir {
namespace poly {

class PolyOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<PolyType>([&](auto &polyType) {
                     os << "Poly";
                     os << "_PF";
                     os << polyType.getBaseField().getModulus().getValue();
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }
};

void PolyDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkir/Dialect/Poly/IR/PolyAttributes.cpp.inc"  // NOLINT(build/include)
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "zkir/Dialect/Poly/IR/PolyTypes.cpp.inc"  // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "zkir/Dialect/Poly/IR/PolyOps.cpp.inc"  // NOLINT(build/include)
      >();

  addInterface<PolyOpAsmDialectInterface>();
}

}  // namespace poly
}  // namespace zkir
}  // namespace mlir
