#include "zkir/Utils/SimpleStructBuilder.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace mlir::zkir {

// static
template <size_t kNumFields>
SimpleStructBuilder<kNumFields>
SimpleStructBuilder<kNumFields>::poison(OpBuilder &builder, Location loc,
                                        Type type) {
  Value poisonValue = builder.create<LLVM::PoisonOp>(loc, type);
  return SimpleStructBuilder<kNumFields>(poisonValue);
}

// static
template <size_t kNumFields>
SimpleStructBuilder<kNumFields> SimpleStructBuilder<kNumFields>::initialized(
    OpBuilder &builder, Location loc, Type type, ValueRange fieldValues) {
  SimpleStructBuilder<kNumFields> structBuilder =
      SimpleStructBuilder<kNumFields>::poison(builder, loc, type);
  structBuilder.setValues(builder, loc, fieldValues);
  return structBuilder;
}

template <size_t kNumFields>
SmallVector<Value>
SimpleStructBuilder<kNumFields>::getValues(OpBuilder &builder, Location loc) {
  SmallVector<Value> fields(kNumFields);
  for (size_t i = 0; i < kNumFields; i++) {
    fields[i] = extractPtr(builder, loc, i);
  }
  return fields;
}

template <size_t kNumFields>
void SimpleStructBuilder<kNumFields>::setValues(OpBuilder &builder,
                                                Location loc,
                                                ValueRange fieldValues) {
  assert(fieldValues.size() == kNumFields &&
         "Number of field values must match template parameter");
  for (size_t i = 0; i < kNumFields; ++i) {
    setPtr(builder, loc, i, fieldValues[i]);
  }
}

// Explicit instantiations for common sizes
template class SimpleStructBuilder<1>;
template class SimpleStructBuilder<2>;
template class SimpleStructBuilder<3>;
template class SimpleStructBuilder<4>;

} // namespace mlir::zkir
