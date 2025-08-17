#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"

#include <cassert>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

// IWYU pragma: begin_keep
// Headers needed for ModArithDialect.cpp.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
// Headers needed for ModArithAttributes.cpp.inc
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
// Headers needed for ModArithOps.cpp.inc
#include "zkir/Dialect/ModArith/IR/ModArithOps.h"
#include "zkir/Utils/OpUtils.h"
// IWYU pragma: end_keep

// Generated definitions
#include "zkir/Dialect/ModArith/IR/ModArithDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithTypes.cpp.inc"

#define GET_OP_CLASSES
#include "zkir/Dialect/ModArith/IR/ModArithOps.cpp.inc"

namespace mlir::zkir::mod_arith {

class ModArithOpAsmDialectInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<ModArithType>([&](auto &modArithType) {
                     os << "z";
                     os << modArithType.getModulus().getValue();
                     os << "_";
                     os << modArithType.getModulus().getType();
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }
};

void ModArithDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.cpp.inc" // NOLINT(build/include)
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "zkir/Dialect/ModArith/IR/ModArithTypes.cpp.inc" // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "zkir/Dialect/ModArith/IR/ModArithOps.cpp.inc" // NOLINT(build/include)
      >();

  addInterface<ModArithOpAsmDialectInterface>();
}

} // namespace mlir::zkir::mod_arith
