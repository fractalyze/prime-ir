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

#include <optional>

#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/IR/Attributes.h"
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

    OpFoldResult foldScalar(ScalarAttr lhs) const {
      return getScalarAttr(operate(getNativeInput(lhs)));
    }
    OpFoldResult foldTensor(TensorAttr lhs) const {
      SmallVector<NativeOutputType> values;
      if (lhs.isSplat()) {
        values = {operate(lhs.template getSplatValue<NativeInputType>())};
      } else {
        values = llvm::map_to_vector(
            lhs.template getValues<NativeInputType>(),
            [this](const NativeInputType &value) -> NativeOutputType {
              return operate(value);
            });
      }
      return getTensorAttr(lhs.getType(), values);
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

    OpFoldResult foldScalar(ScalarAttr lhs, ScalarAttr rhs) const {
      return getScalarAttr(operate(getNativeInput(lhs), getNativeInput(rhs)));
    }
    virtual OpFoldResult foldScalar(ScalarAttr rhs) const { return {}; }
    OpFoldResult foldTensor(TensorAttr lhs, TensorAttr rhs) const {
      SmallVector<NativeOutputType> values;
      if (lhs.isSplat() && rhs.isSplat()) {
        values = {operate(lhs.template getSplatValue<NativeInputType>(),
                          rhs.template getSplatValue<NativeInputType>())};
      } else {
        values = llvm::map_to_vector(
            llvm::zip(lhs.template getValues<NativeInputType>(),
                      rhs.template getValues<NativeInputType>()),
            [this](const auto &values) -> NativeOutputType {
              const auto &[lhs, rhs] = values;
              return operate(lhs, rhs);
            });
      }
      return getTensorAttr(lhs.getType(), values);
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
class AdditiveConstantFolder : public BinaryConstantFolder<Config>::Delegate {
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
    if (rhs.isSplat()) {
      if (isZero(rhs.template getSplatValue<NativeInputType>())) {
        // x ± 0 -> x
        return getLhs();
      }
    } else {
      auto rhsValues = rhs.template getValues<NativeInputType>();
      if (llvm::all_of(rhsValues, [this](const NativeInputType &value) {
            return isZero(value);
          })) {
        // x ± 0 -> x
        return getLhs();
      }
    }
    return {};
  }
};

template <typename Config>
class MultiplicativeConstantFolder
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
    if (rhs.isSplat()) {
      auto rhsValue = rhs.template getSplatValue<NativeInputType>();
      if (isZero(rhsValue)) {
        // x * 0 -> 0
        return rhs;
      } else if (isOne(rhsValue)) {
        // x * 1 -> x
        return getLhs();
      }
    } else {
      auto rhsValues = rhs.template getValues<NativeInputType>();
      if (llvm::all_of(rhsValues, [this](const NativeInputType &value) {
            return isZero(value);
          })) {
        // x * 0 -> 0
        return rhs;
      } else if (llvm::all_of(rhsValues, [this](const NativeInputType &value) {
                   return isOne(value);
                 })) {
        // x * 1 -> x
        return getLhs();
      }
    }
    return {};
  }
};

// TODO(junbeomlee): Unify with UnaryConstantFolder and BinaryConstantFolder
// after deciding how to handle Inverse(0).
// See https://github.com/fractalyze/prime-ir/issues/177
// Extension field constant folder for unary operations.
template <typename Config>
class ExtensionFieldUnaryConstantFolder {
public:
  using NativeInputType = typename Config::NativeInputType;
  using NativeOutputType = typename Config::NativeOutputType;
  using ScalarAttr = typename Config::ScalarAttr;

  class Delegate {
  public:
    virtual ~Delegate() = default;

    virtual NativeInputType getNativeInput(ScalarAttr attr) const = 0;
    virtual std::optional<NativeOutputType>
    operate(const NativeInputType &coeffs) const = 0;
    virtual OpFoldResult
    getScalarAttr(const NativeOutputType &coeffs) const = 0;

    OpFoldResult foldScalar(ScalarAttr input) const {
      auto result = operate(getNativeInput(input));
      if (!result)
        return {};
      return getScalarAttr(*result);
    }
  };

  template <typename FoldAdaptor, typename DelegateT>
  static OpFoldResult fold(FoldAdaptor adaptor, DelegateT *delegate) {
    if (auto inputAttr = dyn_cast_if_present<ScalarAttr>(adaptor.getInput())) {
      return delegate->foldScalar(inputAttr);
    }
    return {};
  }
};

// Extension field constant folder for binary operations.
template <typename Config>
class ExtensionFieldBinaryConstantFolder {
public:
  using NativeInputType = typename Config::NativeInputType;
  using NativeOutputType = typename Config::NativeOutputType;
  using ScalarAttr = typename Config::ScalarAttr;

  class Delegate {
  public:
    virtual ~Delegate() = default;

    virtual NativeInputType getNativeInput(ScalarAttr attr) const = 0;
    virtual NativeOutputType operate(const NativeInputType &lhs,
                                     const NativeInputType &rhs) const = 0;
    virtual OpFoldResult
    getScalarAttr(const NativeOutputType &coeffs) const = 0;

    OpFoldResult foldScalar(ScalarAttr lhs, ScalarAttr rhs) const {
      return getScalarAttr(operate(getNativeInput(lhs), getNativeInput(rhs)));
    }
  };

  template <typename FoldAdaptor, typename DelegateT>
  static OpFoldResult fold(FoldAdaptor adaptor, DelegateT *delegate) {
    auto lhsAttr = dyn_cast_if_present<ScalarAttr>(adaptor.getLhs());
    auto rhsAttr = dyn_cast_if_present<ScalarAttr>(adaptor.getRhs());
    if (lhsAttr && rhsAttr) {
      return delegate->foldScalar(lhsAttr, rhsAttr);
    }
    return {};
  }
};

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_CONSTANTFOLDER_H_
