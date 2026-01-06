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

#include "zkir/Dialect/Field/IR/FieldDialect.h"

#include <cassert>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Utils/KnownModulus.h"

// IWYU pragma: begin_keep
// Headers needed for FieldDialect.cpp.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OperationSupport.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"

// Headers needed for FieldAttributes.cpp.inc
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"

// Headers needed for FieldOps.cpp.inc
#include "zkir/Dialect/Field/IR/FieldOperation.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkir/Utils/OpUtils.h"
// IWYU pragma: end_keep

// Generated definitions
#include "zkir/Dialect/Field/IR/FieldDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zkir/Dialect/Field/IR/FieldAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zkir/Dialect/Field/IR/FieldTypes.cpp.inc"

#define GET_OP_CLASSES
#include "zkir/Dialect/Field/IR/FieldOps.cpp.inc"

namespace mlir::zkir::field {

struct FieldInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // All field dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

class FieldOpAsmDialectInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res =
        llvm::TypeSwitch<Type, AliasResult>(type)
            .Case<PrimeFieldType>([&](auto &pfElemType) {
              auto modulus = pfElemType.getModulus().getValue();
              std::optional<std::string> alias = getKnownModulusAlias(modulus);
              if (alias) {
                os << "pf_" << *alias;
                if (!pfElemType.isMontgomery()) {
                  os << "_std";
                }
                return AliasResult::FinalAlias;
              }
              os << "pf" << modulus << "_" << pfElemType.getStorageType();
              return AliasResult::OverridableAlias;
            })
            .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    auto res =
        llvm::TypeSwitch<Attribute, AliasResult>(attr)
            .Case<RootOfUnityAttr>([&](auto &rootOfUnityAttr) {
              os << "root_of_unity_"
                 << rootOfUnityAttr.getDegree().getValue().countTrailingZeros();
              return AliasResult::FinalAlias;
            })
            .Default([&](Attribute) { return AliasResult::NoAlias; });
    return res;
  }
};

void FieldDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "zkir/Dialect/Field/IR/FieldTypes.cpp.inc" // NOLINT(build/include)
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkir/Dialect/Field/IR/FieldAttributes.cpp.inc" // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "zkir/Dialect/Field/IR/FieldOps.cpp.inc" // NOLINT(build/include)
      >();

  addInterface<FieldInlinerInterface>();
  addInterface<FieldOpAsmDialectInterface>();
}

} // namespace mlir::zkir::field
