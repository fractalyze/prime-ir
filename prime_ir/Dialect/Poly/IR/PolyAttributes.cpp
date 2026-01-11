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

#include "prime_ir/Dialect/Poly/IR/PolyAttributes.h"

#include <utility>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOperation.h"

namespace mlir::prime_ir::poly {

field::RootOfUnityAttr PrimitiveRootAttr::getRootOfUnity() const {
  return getImpl()->rootOfUnity;
}

IntegerAttr PrimitiveRootAttr::getRoot() const { return getImpl()->root; }

IntegerAttr PrimitiveRootAttr::getInvRoot() const { return getImpl()->invRoot; }

IntegerAttr PrimitiveRootAttr::getDegree() const {
  return getImpl()->rootOfUnity.getDegree();
}

IntegerAttr PrimitiveRootAttr::getInvDegree() const {
  return getImpl()->invDegree;
}

DenseElementsAttr PrimitiveRootAttr::getRoots() const {
  return getImpl()->roots;
}

DenseElementsAttr PrimitiveRootAttr::getInvRoots() const {
  return getImpl()->invRoots;
}

mod_arith::MontgomeryAttr PrimitiveRootAttr::getMontgomery() const {
  return getImpl()->montgomery;
}

namespace detail {

// static
PrimitiveRootAttrStorage *
PrimitiveRootAttrStorage::construct(AttributeStorageAllocator &allocator,
                                    KeyTy &&key) {
  field::RootOfUnityAttr rootOfUnity = std::get<0>(key);
  mod_arith::MontgomeryAttr montgomery = std::get<1>(key);

  auto root = field::PrimeFieldOperation::fromUnchecked(
      rootOfUnity.getRoot(),
      cast<field::PrimeFieldType>(rootOfUnity.getType()));
  field::PrimeFieldOperation invRoot = root.inverse();
  auto degree = field::PrimeFieldOperation(rootOfUnity.getDegree().getValue(),
                                           rootOfUnity.getType());
  field::PrimeFieldOperation invDegree = degree.inverse();
  // Compute the exponent table.
  SmallVector<APInt> roots, invRoots;
  unsigned degreeInt = rootOfUnity.getDegree().getInt();
  auto computePowers = [&](const field::PrimeFieldOperation &root,
                           SmallVector<APInt> &roots) {
    roots.reserve(degreeInt);
    field::PrimeFieldOperation cur = root.getOne();
    for (unsigned i = 0; i < degreeInt; ++i) {
      roots.push_back(cur);
      cur *= root;
    }
  };
  computePowers(root, roots);
  computePowers(invRoot, invRoots);
  // Create a ranked tensor type for the exponents attribute.
  auto tensorType =
      RankedTensorType::get({degreeInt}, rootOfUnity.getRoot().getType());

  // Create the DenseIntElementsAttr from the computed exponent values.
  DenseElementsAttr rootsAttr = DenseElementsAttr::get(tensorType, roots);
  DenseElementsAttr invRootsAttr = DenseElementsAttr::get(tensorType, invRoots);
  return new (allocator.allocate<PrimitiveRootAttrStorage>())
      PrimitiveRootAttrStorage(std::move(rootOfUnity), root.getIntegerAttr(),
                               invDegree.getIntegerAttr(),
                               invRoot.getIntegerAttr(), std::move(rootsAttr),
                               std::move(invRootsAttr), std::move(montgomery));
}

} // namespace detail
} // namespace mlir::prime_ir::poly
