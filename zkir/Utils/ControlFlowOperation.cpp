/* Copyright 2026 The ZKIR Authors.

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

#include "zkir/Utils/ControlFlowOperation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace zk_dtypes {

mlir::Value ControlFlowOperation<mlir::Value>::Equal(mlir::Value x,
                                                     mlir::Value y) {
  mlir::ImplicitLocOpBuilder *b =
      mlir::zkir::BuilderContext::GetInstance().Top();
  mlir::Type xType = getElementTypeOrSelf(x.getType());
  if (mlir::isa<mlir::zkir::field::PrimeFieldType>(xType) ||
      mlir::isa<mlir::zkir::field::ExtensionFieldTypeInterface>(xType)) {
    return b->create<mlir::zkir::field::CmpOp>(mlir::arith::CmpIPredicate::eq,
                                               x, y);
  }
  llvm_unreachable("Unsupported type for comparison");
  return nullptr;
}

mlir::Value ControlFlowOperation<mlir::Value>::NotEqual(mlir::Value x,
                                                        mlir::Value y) {
  mlir::ImplicitLocOpBuilder *b =
      mlir::zkir::BuilderContext::GetInstance().Top();
  mlir::Type xType = getElementTypeOrSelf(x.getType());
  if (mlir::isa<mlir::zkir::field::PrimeFieldType>(xType) ||
      mlir::isa<mlir::zkir::field::ExtensionFieldTypeInterface>(xType)) {
    return b->create<mlir::zkir::field::CmpOp>(mlir::arith::CmpIPredicate::ne,
                                               x, y);
  }
  llvm_unreachable("Unsupported type for comparison");
  return nullptr;
}

mlir::Value ControlFlowOperation<mlir::Value>::And(mlir::Value x,
                                                   mlir::Value y) {
  mlir::ImplicitLocOpBuilder *b =
      mlir::zkir::BuilderContext::GetInstance().Top();
  return b->create<mlir::arith::AndIOp>(x, y);
}

mlir::Value ControlFlowOperation<mlir::Value>::Or(mlir::Value x,
                                                  mlir::Value y) {
  mlir::ImplicitLocOpBuilder *b =
      mlir::zkir::BuilderContext::GetInstance().Top();
  return b->create<mlir::arith::OrIOp>(x, y);
}

mlir::Value ControlFlowOperation<mlir::Value>::Not(mlir::Value x) {
  mlir::ImplicitLocOpBuilder *b =
      mlir::zkir::BuilderContext::GetInstance().Top();
  auto zero = b->create<mlir::arith::ConstantOp>(x.getLoc(), b->getI1Type(),
                                                 b->getBoolAttr(false));
  return b->create<mlir::arith::CmpIOp>(mlir::arith::CmpIPredicate::eq, x,
                                        zero);
}

mlir::Value ControlFlowOperation<mlir::Value>::True() {
  mlir::ImplicitLocOpBuilder *b =
      mlir::zkir::BuilderContext::GetInstance().Top();
  return b->create<mlir::arith::ConstantOp>(b->getI1Type(),
                                            b->getBoolAttr(true));
}

mlir::Value ControlFlowOperation<mlir::Value>::False() {
  mlir::ImplicitLocOpBuilder *b =
      mlir::zkir::BuilderContext::GetInstance().Top();
  return b->create<mlir::arith::ConstantOp>(b->getI1Type(),
                                            b->getBoolAttr(false));
}

} // namespace zk_dtypes
