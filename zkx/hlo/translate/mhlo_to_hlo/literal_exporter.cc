/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/hlo/translate/mhlo_to_hlo/literal_exporter.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/APInt.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

#include "zkx/array.h"
#include "zkx/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/shape.h"
#include "zkx/types.h"
#include "zkx/zkx_data.pb.h"

namespace mlir::mhlo {

template <typename T>
zkx::Array<T> ArrayFromDenseElementsAttr(mlir::DenseElementsAttr dense_attr) {
  constexpr zkx::PrimitiveType type =
      zkx::primitive_util::NativeToPrimitiveType<T>();
  zkx::Shape shape = zkx::TypeToShape(dense_attr.getType());
  zkx::Array<T> array(shape.dimensions());
  if constexpr (!zkx::primitive_util::IsSubByteNonPredType(type)) {
    if constexpr (type == zkx::PRED ||
                  zkx::primitive_util::IsIntegralType(type)) {
      if constexpr (zkx::is_big_int_v<T>) {
        // BigInt types need to be converted from APInt.
        auto values = dense_attr.getValues<llvm::APInt>();
        for (int i = 0; i < values.size(); ++i) {
          T big_int;
          std::copy_n(values[i].getRawData(), T::kLimbNums, big_int.limbs());
          array.data()[i] = big_int;
        }
      } else {
        array.SetValues(dense_attr.getValues<T>());
      }
    } else {
      LOG(FATAL) << "ArrayFromDenseElementsAttr is not implemented for type: "
                 << zkx::PrimitiveType_Name(type);
    }
  } else {
    // The only way to get subbyte integers from getValues() is to get them as
    // APInts.
    auto values = dense_attr.getValues<llvm::APInt>();
    for (int i = 0; i < values.size(); i++) {
      if constexpr (zkx::primitive_util::IsUnsignedIntegralType(type)) {
        array.data()[i] = T{values[i].getZExtValue()};
      } else {
        static_assert(zkx::primitive_util::IsSignedIntegralType(type));
        array.data()[i] = T{values[i].getSExtValue()};
      }
    }
  }
  return array;
}

absl::StatusOr<zkx::Literal> CreateLiteralFromAttribute(mlir::ElementsAttr attr,
                                                        zkx::Layout layout) {
  auto dense_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense_attr)
    return absl::UnimplementedError("Only dense elements attr are supported");

  zkx::Shape shape = zkx::TypeToShape(dense_attr.getType());

  return zkx::primitive_util::PrimitiveTypeSwitch<absl::StatusOr<zkx::Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<zkx::Literal> {
        if constexpr (zkx::primitive_util::IsArrayType(
                          primitive_type_constant)) {
          using cpp_type =
              zkx::primitive_util::NativeTypeOf<primitive_type_constant>;
          zkx::Array<cpp_type> source_data =
              ArrayFromDenseElementsAttr<cpp_type>(dense_attr);
          if (layout.minor_to_major().empty()) {
            return zkx::LiteralUtil::CreateFromArray(source_data);
          }
          return zkx::LiteralUtil::CreateFromArrayWithLayout(source_data,
                                                             layout);
        }
        return absl::InternalError(
            absl::StrCat("Unsupported type: ",
                         zkx::PrimitiveType_Name(shape.element_type())));
      },
      shape.element_type());
}

}  // namespace mlir::mhlo
