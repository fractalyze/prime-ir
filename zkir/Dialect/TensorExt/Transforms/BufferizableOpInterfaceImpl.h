#ifndef ZKIR_DIALECT_TENSOREXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define ZKIR_DIALECT_TENSOREXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace mlir {
class DialectRegistry;

namespace zkir::tensor_ext {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
}  // namespace zkir::tensor_ext
}  // namespace mlir

#endif  // ZKIR_DIALECT_TENSOREXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
