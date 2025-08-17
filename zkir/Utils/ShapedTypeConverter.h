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
