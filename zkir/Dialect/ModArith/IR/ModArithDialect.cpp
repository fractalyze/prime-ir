/* Copyright 2025 The ZKIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"

#include <cassert>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
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

struct ModArithInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // All ModArith dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

class ModArithOpAsmDialectInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<ModArithType>([&](auto &modArithType) {
                     os << "z";
                     os << modArithType.getModulus().getValue();
                     os << "_";
                     os << modArithType.getStorageType();
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

  addInterface<ModArithInlinerInterface>();
  addInterface<ModArithOpAsmDialectInterface>();
}

} // namespace mlir::zkir::mod_arith
