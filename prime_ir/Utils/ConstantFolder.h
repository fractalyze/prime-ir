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

#ifndef PRIME_IR_UTILS_CONSTANTFOLDER_H_
#define PRIME_IR_UTILS_CONSTANTFOLDER_H_

#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir::prime_ir {

template <typename Config>
class UnaryConstantFolder {
public:
  using NativeInputType = typename Config::NativeInputType;
  using NativeOutputType = typename Config::NativeOutputType;
  using ScalarAttr = typename Config::ScalarAttr;
  using TensorAttr = typename Config::TensorAttr;

  class Delegate {
  public:
    virtual ~Delegate() = default;

    virtual NativeInputType getNativeInput(ScalarAttr attr) const = 0;
    virtual NativeOutputType operate(const NativeInputType &value) const = 0;
    virtual OpFoldResult getScalarAttr(const NativeOutputType &value) const = 0;
    virtual OpFoldResult
    getTensorAttr(ShapedType type, ArrayRef<NativeOutputType> values) const = 0;

    // Extract per-prime-coeff values from a tensor constant. Field types
    // (DenseElementsAttr<tensor<...x!PF>>) walk raw bytes; storage-int
    // (DenseIntElementsAttr) iterates via getValues<APInt>. Delegates with
    // non-trivial element types (e.g. EF using SmallVector<APInt>) override.
    virtual SmallVector<NativeInputType, 0>
    extractValues(TensorAttr attr) const {
      if constexpr (std::is_same_v<NativeInputType, APInt>) {
        if (auto intAttr = llvm::dyn_cast<DenseIntElementsAttr>(attr))
          return llvm::to_vector<0>(intAttr.template getValues<APInt>());
        return {};
      } else {
        return {};
      }
    }

    virtual OpFoldResult foldScalar(ScalarAttr lhs) const {
      return getScalarAttr(operate(getNativeInput(lhs)));
    }
    virtual OpFoldResult foldTensor(TensorAttr lhs) const {
      if constexpr (std::is_same_v<ScalarAttr, TensorAttr>) {
        return {};
      } else {
        SmallVector<NativeInputType, 0> inputs = extractValues(lhs);
        if (inputs.empty())
          return {};
        auto values = llvm::map_to_vector(
            inputs, [this](const NativeInputType &v) { return operate(v); });
        return getTensorAttr(lhs.getType(), values);
      }
    }
  };

  template <typename FoldAdaptor, typename Delegate>
  static OpFoldResult fold(FoldAdaptor adaptor, Delegate *delegate) {
    if (auto inputAttr = dyn_cast_if_present<ScalarAttr>(adaptor.getInput())) {
      return delegate->foldScalar(inputAttr);
    } else if (auto inputAttr =
                   dyn_cast_if_present<TensorAttr>(adaptor.getInput())) {
      return delegate->foldTensor(inputAttr);
    }
    return {};
  }
};

template <typename Config>
class BinaryConstantFolder {
public:
  using NativeInputType = typename Config::NativeInputType;
  using NativeOutputType = typename Config::NativeOutputType;
  using ScalarAttr = typename Config::ScalarAttr;
  using TensorAttr = typename Config::TensorAttr;

  class Delegate {
  public:
    virtual ~Delegate() = default;

    virtual NativeInputType getNativeInput(ScalarAttr attr) const = 0;
    virtual NativeOutputType operate(const NativeInputType &lhs,
                                     const NativeInputType &rhs) const = 0;
    virtual OpFoldResult getScalarAttr(const NativeOutputType &value) const = 0;
    virtual OpFoldResult
    getTensorAttr(ShapedType type, ArrayRef<NativeOutputType> values) const = 0;

    // Extract per-prime-coeff values from a tensor constant. See
    // UnaryConstantFolder::Delegate::extractValues for the rationale.
    virtual SmallVector<NativeInputType, 0>
    extractValues(TensorAttr attr) const {
      if constexpr (std::is_same_v<NativeInputType, APInt>) {
        if (auto intAttr = llvm::dyn_cast<DenseIntElementsAttr>(attr))
          return llvm::to_vector<0>(intAttr.template getValues<APInt>());
        return {};
      } else {
        return {};
      }
    }

    virtual OpFoldResult foldScalar(ScalarAttr lhs, ScalarAttr rhs) const {
      return getScalarAttr(operate(getNativeInput(lhs), getNativeInput(rhs)));
    }
    virtual OpFoldResult foldScalar(ScalarAttr rhs) const { return {}; }
    virtual OpFoldResult foldTensor(TensorAttr lhs, TensorAttr rhs) const {
      if constexpr (std::is_same_v<ScalarAttr, TensorAttr>) {
        return {};
      } else {
        SmallVector<NativeInputType, 0> lhsValues = extractValues(lhs);
        SmallVector<NativeInputType, 0> rhsValues = extractValues(rhs);
        if (lhsValues.empty() || rhsValues.empty())
          return {};
        if (lhsValues.size() != rhsValues.size())
          return {};
        auto values =
            llvm::map_to_vector(llvm::zip(lhsValues, rhsValues),
                                [this](const auto &pair) -> NativeOutputType {
                                  const auto &[l, r] = pair;
                                  return operate(l, r);
                                });
        return getTensorAttr(lhs.getType(), values);
      }
    }
    virtual OpFoldResult foldTensor(TensorAttr rhs) const { return {}; }
  };

  template <typename FoldAdaptor, typename Delegate>
  static OpFoldResult fold(FoldAdaptor adaptor, Delegate *delegate) {
    if (auto rhsAttr = dyn_cast_if_present<ScalarAttr>(adaptor.getRhs())) {
      if (auto lhsAttr = dyn_cast_if_present<ScalarAttr>(adaptor.getLhs())) {
        return delegate->foldScalar(lhsAttr, rhsAttr);
      } else {
        return delegate->foldScalar(rhsAttr);
      }
    } else if (auto rhsAttr =
                   dyn_cast_if_present<TensorAttr>(adaptor.getRhs())) {
      if (auto lhsAttr = dyn_cast_if_present<TensorAttr>(adaptor.getLhs())) {
        return delegate->foldTensor(lhsAttr, rhsAttr);
      } else {
        return delegate->foldTensor(rhsAttr);
      }
    }
    return {};
  }
};

template <typename Config>
class AdditiveConstantFolderDelegate
    : public BinaryConstantFolder<Config>::Delegate {
public:
  using NativeInputType = typename Config::NativeInputType;
  using NativeOutputType = typename Config::NativeOutputType;
  using ScalarAttr = typename Config::ScalarAttr;
  using TensorAttr = typename Config::TensorAttr;

  virtual bool isZero(const NativeInputType &value) const = 0;
  virtual OpFoldResult getLhs() const = 0;

  using BinaryConstantFolder<Config>::Delegate::foldScalar;
  using BinaryConstantFolder<Config>::Delegate::foldTensor;

  OpFoldResult foldScalar(ScalarAttr rhs) const override {
    if (isZero(this->getNativeInput(rhs))) {
      // x ± 0 -> x
      return getLhs();
    }
    return {};
  }

  OpFoldResult foldTensor(TensorAttr rhs) const override {
    if constexpr (std::is_same_v<ScalarAttr, TensorAttr>) {
      return {};
    } else {
      SmallVector<NativeInputType, 0> values = this->extractValues(rhs);
      if (values.empty())
        return {};
      if (llvm::all_of(
              values, [this](const NativeInputType &v) { return isZero(v); })) {
        // x ± 0 -> x
        return getLhs();
      }
      return {};
    }
  }
};

template <typename Config>
class MultiplicativeConstantFolderDelegate
    : public BinaryConstantFolder<Config>::Delegate {
public:
  using NativeInputType = typename Config::NativeInputType;
  using NativeOutputType = typename Config::NativeOutputType;
  using ScalarAttr = typename Config::ScalarAttr;
  using TensorAttr = typename Config::TensorAttr;

  virtual bool isZero(const NativeInputType &value) const = 0;
  virtual bool isOne(const NativeInputType &value) const = 0;
  virtual OpFoldResult getLhs() const = 0;

  using BinaryConstantFolder<Config>::Delegate::foldScalar;
  using BinaryConstantFolder<Config>::Delegate::foldTensor;

  OpFoldResult foldScalar(ScalarAttr rhs) const override {
    NativeInputType rhsValue = this->getNativeInput(rhs);
    if (isZero(rhsValue)) {
      // x * 0 -> 0
      return rhs;
    } else if (isOne(rhsValue)) {
      // x * 1 -> x
      return getLhs();
    }
    return {};
  }

  OpFoldResult foldTensor(TensorAttr rhs) const override {
    if constexpr (std::is_same_v<ScalarAttr, TensorAttr>) {
      return {};
    } else {
      SmallVector<NativeInputType, 0> values = this->extractValues(rhs);
      if (values.empty())
        return {};
      if (llvm::all_of(
              values, [this](const NativeInputType &v) { return isZero(v); })) {
        // x * 0 -> 0
        return rhs;
      } else if (llvm::all_of(values, [this](const NativeInputType &v) {
                   return isOne(v);
                 })) {
        // x * 1 -> x
        return getLhs();
      }
      return {};
    }
  }
};

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_CONSTANTFOLDER_H_
