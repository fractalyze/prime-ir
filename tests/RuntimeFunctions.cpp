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

#include "tests/RuntimeFunctions.h"

#include "zk_dtypes/include/big_int.h"

namespace mlir::zkir {

template <typename T>
void printMemref(void *memref) {
  impl::printMemRef(*reinterpret_cast<UnrankedMemRefType<T> *>(memref));
}

#define IF_SAME_TYPE(ActualType, ...) (std::is_same_v<T, ActualType>) ||

template <int N, typename T>
void printMemref(void *memref) {
  // For elliptic curve point types, we print the underlying memref of base
  // fields.
  // R1 points (like G1) are rank N+1 (e.g., memref<Nx2x!field>), and
  // R2 points (like G2) are rank N+2 (e.g., memref<Nx2x2x!field>).
  if constexpr (ZK_DTYPES_ALL_R1_EC_POINT_TYPE_LIST(IF_SAME_TYPE) false) {
    using BaseField = typename T::BaseField;
    impl::printMemRef(
        *reinterpret_cast<StridedMemRefType<BaseField, N + 1> *>(memref));
    // NOLINTNEXTLINE(readability/braces)
  } else if constexpr (ZK_DTYPES_ALL_R2_EC_POINT_TYPE_LIST(
                           IF_SAME_TYPE) false) {
    using BaseField = typename T::BaseField;
    impl::printMemRef(
        *reinterpret_cast<StridedMemRefType<BaseField, N + 2> *>(memref));
  } else {
    impl::printMemRef(*reinterpret_cast<StridedMemRefType<T, N> *>(memref));
  }
}

#undef IF_SAME_TYPE

} // namespace mlir::zkir

extern "C" {
#define DEFINE_PRINT_MEMREF_FUNCTION(ActualType, UpperCamelCaseName, ...)      \
  void _mlir_ciface_printMemref##UpperCamelCaseName(void *memref) {            \
    mlir::zkir::printMemref<ActualType>(memref);                               \
  }                                                                            \
  void _mlir_ciface_printMemref1##D##UpperCamelCaseName(void *memref) {        \
    mlir::zkir::printMemref<1, ActualType>(memref);                            \
  }                                                                            \
  void _mlir_ciface_printMemref2##D##UpperCamelCaseName(void *memref) {        \
    mlir::zkir::printMemref<2, ActualType>(memref);                            \
  }                                                                            \
  void _mlir_ciface_printMemref3##D##UpperCamelCaseName(void *memref) {        \
    mlir::zkir::printMemref<3, ActualType>(memref);                            \
  }
ZK_DTYPES_ALL_TYPE_LIST(DEFINE_PRINT_MEMREF_FUNCTION)
#undef DEFINE_PRINT_MEMREF_FUNCTION

void _mlir_ciface_printMemrefI256(void *memref) {
  mlir::zkir::printMemref<::zk_dtypes::BigInt<4>>(memref);
}

void _mlir_ciface_printMemref1DI256(void *memref) {
  mlir::zkir::printMemref<1, ::zk_dtypes::BigInt<4>>(memref);
}

void _mlir_ciface_printMemref2DI256(void *memref) {
  mlir::zkir::printMemref<2, ::zk_dtypes::BigInt<4>>(memref);
}

void _mlir_ciface_printMemref3DI256(void *memref) {
  mlir::zkir::printMemref<3, ::zk_dtypes::BigInt<4>>(memref);
}
}
