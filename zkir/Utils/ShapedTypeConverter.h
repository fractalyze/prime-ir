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

#ifndef ZKIR_UTILS_SHAPEDTYPECONVERTER_H_
#define ZKIR_UTILS_SHAPEDTYPECONVERTER_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::zkir {

class ShapedTypeConverter : public TypeConverter {
protected:
  static Type convertShapedType(ShapedType oldType, ArrayRef<int64_t> shape,
                                Type elementType);
};

} // namespace mlir::zkir

#endif // ZKIR_UTILS_SHAPEDTYPECONVERTER_H_
