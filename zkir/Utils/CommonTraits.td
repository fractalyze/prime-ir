#ifndef ZKIR_UTILS_COMMONTRAITS_TD_
#define ZKIR_UTILS_COMMONTRAITS_TD_

include "mlir/IR/OpBase.td"

class VarPositiveAndBounded<string var, string upperBoundExpr>
  : PredOpTrait<"VarPositiveAndBounded",
      CPred<
        "(" # var # ") > 0 && "
        "(" # var # ") <= (" # upperBoundExpr # ")"
      >>;

#endif  // ZKIR_UTILS_COMMONTRAITS_TD_
