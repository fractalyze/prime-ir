#ifndef ZKIR_UTILS_COMMONTRAITS_TD_
#define ZKIR_UTILS_COMMONTRAITS_TD_

include "mlir/IR/OpBase.td"

class VarPositiveAndBounded<string var, string upperBoundExpr>
  : PredOpTrait<"VarPositiveAndBounded",
      CPred<
        "(" # var # ") > 0 && "
        "(" # var # ") <= (" # upperBoundExpr # ")"
      >>;

// Checks that tensor input and outputs have identical shapes. This is stricker
// than the verification done in `SameOperandsAndResultShape` that allows for
// tensor dimensions to be 'compatible' (e.g., dynamic dimensions being
// compatible with static ones).
class AllSameTensorDims<list<string> types> : AllMatchSameOperatorTrait<types,
      "(::llvm::isa<::mlir::TensorType>($_self.getType()) ?"
      " ::llvm::cast<::mlir::TensorType>($_self.getType()).getShape() :"
      " ::llvm::ArrayRef<int64_t>{})", "matching static and dynamic dimensions">;

#endif  // ZKIR_UTILS_COMMONTRAITS_TD_
