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

#ifndef PRIME_IR_DIALECT_POLY_IR_POLYATTRIBUTES_H_
#define PRIME_IR_DIALECT_POLY_IR_POLYATTRIBUTES_H_

#include <tuple>
#include <utility>

#include "prime_ir/Dialect/Field/IR/FieldAttributes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.h"

namespace mlir::prime_ir::poly::detail {

struct PrimitiveRootAttrStorage : public AttributeStorage {
  using KeyTy = std::tuple<field::RootOfUnityAttr, mod_arith::MontgomeryAttr>;
  PrimitiveRootAttrStorage(field::RootOfUnityAttr rootOfUnity, IntegerAttr root,
                           IntegerAttr invDegree, IntegerAttr invRoot,
                           DenseElementsAttr roots, DenseElementsAttr invRoots,
                           mod_arith::MontgomeryAttr montgomery)
      : rootOfUnity(std::move(rootOfUnity)), root(std::move(root)),
        invDegree(std::move(invDegree)), invRoot(std::move(invRoot)),
        roots(std::move(roots)), invRoots(std::move(invRoots)),
        montgomery(std::move(montgomery)) {}

  KeyTy getAsKey() const { return KeyTy(rootOfUnity, montgomery); }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(rootOfUnity, montgomery);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  static PrimitiveRootAttrStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy &&key);

  field::RootOfUnityAttr rootOfUnity;
  IntegerAttr root;
  IntegerAttr invDegree;
  IntegerAttr invRoot;
  DenseElementsAttr roots;
  DenseElementsAttr invRoots;
  mod_arith::MontgomeryAttr montgomery;
};

} // namespace mlir::prime_ir::poly::detail

#define GET_ATTRDEF_CLASSES
#include "prime_ir/Dialect/Poly/IR/PolyAttributes.h.inc"

#endif // PRIME_IR_DIALECT_POLY_IR_POLYATTRIBUTES_H_
