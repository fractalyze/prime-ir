/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"

#include <cassert>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/KnownCurves.h"

// IWYU pragma: begin_keep
// Headers needed for EllipticCurveDialect.cpp.inc
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
// Headers needed for EllipticCurveAttributes.cpp.inc
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "prime_ir/Dialect/Field/IR/FieldAttributes.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
// Headers needed for EllipticCurveOps.cpp.inc
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Utils/OpUtils.h"
// IWYU pragma: end_keep

// Generated definitions
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.cpp.inc"

#define GET_OP_CLASSES
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.cpp.inc"

namespace mlir::prime_ir::elliptic_curve {

class EllipticCurveOpAsmDialectInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<AffineType>([&](auto &point) {
                     os << "affine";
                     return AliasResult::FinalAlias;
                   })
                   .Case<JacobianType>([&](auto &point) {
                     os << "jacobian";
                     return AliasResult::FinalAlias;
                   })
                   .Case<XYZZType>([&](auto &point) {
                     os << "xyzz";
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    auto res =
        llvm::TypeSwitch<Attribute, AliasResult>(attr)
            .Case<ShortWeierstrassAttr>([&](auto &swAttr) {
              std::optional<std::string> alias = getKnownCurveAlias(swAttr);
              if (alias) {
                auto fieldType =
                    cast<field::FieldTypeInterface>(swAttr.getBaseField());
                os << *alias;
                if (!fieldType.isMontgomery()) {
                  os << "_std";
                }
                return AliasResult::FinalAlias;
              }
              os << "unknown_sw_curve";
              return AliasResult::OverridableAlias;
            })
            .Default([&](Attribute) { return AliasResult::NoAlias; });
    return res;
  }
};

void EllipticCurveDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.cpp.inc" // NOLINT(build/include)
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.cpp.inc" // NOLINT(build/include)
      >();
  addOperations<
#define GET_OP_LIST
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.cpp.inc" // NOLINT(build/include)
      >();

  addInterface<EllipticCurveOpAsmDialectInterface>();
}

} // namespace mlir::prime_ir::elliptic_curve
