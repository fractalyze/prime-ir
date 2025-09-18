#ifndef ZKIR_UTILS_SIMPLESTRUCTBUILDER_H_
#define ZKIR_UTILS_SIMPLESTRUCTBUILDER_H_

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace mlir::zkir {

// A simple utility class for setting and getting fields of a LLVM struct.
template <size_t kNumFields>
class SimpleStructBuilder : public StructBuilder {
public:
  // Construct a helper for the given point value.
  using StructBuilder::StructBuilder;

  // Build IR creating a `poison` value of the given type.
  static SimpleStructBuilder<kNumFields> poison(OpBuilder &builder,
                                                Location loc, Type type);

  // Build IR creating a initialized value of the given type with the given
  // field values.
  static SimpleStructBuilder<kNumFields> initialized(OpBuilder &builder,
                                                     Location loc, Type type,
                                                     ValueRange fieldValues);

  SmallVector<Value> getValues(OpBuilder &builder, Location loc);
  void setValues(OpBuilder &builder, Location loc, ValueRange fieldValues);
};

extern template class SimpleStructBuilder<1>;
extern template class SimpleStructBuilder<2>;
extern template class SimpleStructBuilder<3>;
extern template class SimpleStructBuilder<4>;

} // namespace mlir::zkir

#endif // ZKIR_UTILS_SIMPLESTRUCTBUILDER_H_
