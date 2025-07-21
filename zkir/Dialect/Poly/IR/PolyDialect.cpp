#include "zkir/Dialect/Poly/IR/PolyDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"

// IWYU pragma: begin_keep
// Headers needed for PolyDialect.cpp.inc
#include "zkir/Dialect/Field/IR/FieldDialect.h"
// Headers needed for PolyAttributes.cpp.inc
#include "zkir/Dialect/Poly/IR/PolyAttributes.h"
// Headers needed for PolyOps.cpp.inc
#include "zkir/Dialect/Poly/IR/PolyOps.h"
// IWYU pragma: end_keep

// Generated definitions
#include "zkir/Dialect/Poly/IR/PolyDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/Poly/IR/PolyAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/Poly/IR/PolyTypes.cpp.inc"

#define GET_OP_CLASSES
#include "zkir/Dialect/Poly/IR/PolyOps.cpp.inc"

namespace mlir::zkir::poly {

class PolyOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<PolyType>([&](auto &polyType) {
                     os << "poly_pf";
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

}  // namespace mlir::zkir::poly
