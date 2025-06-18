#ifndef ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
#define ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_

include "zkir/Dialect/ModArith/IR/ModArithOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def ModArithToArith : Pass<"mod-arith-to-arith", "ModuleOp"> {
  let summary = "Lower `mod_arith` to standard `arith`.";
  let description = [{
    This pass lowers the `mod_arith` dialect to their `arith` equivalents.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::zkir::mod_arith::ModArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
  ];
}

defvar DefGE = ConstantEnumCase<Arith_CmpIPredicateAttr, "uge">;
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;

def ConvertSubIfGE : Pattern<
  (ModArith_SubIfGEOp $x, $y),
  [
    (Arith_SubIOp:$subOp $x, $y, DefOverflow),
    (Arith_CmpIOp:$cmpOp DefGE, $x, $y),
    (SelectOp $cmpOp, $subOp, $x)
  ]
>;

#endif  // ZKIR_DIALECT_MODARITH_CONVERSIONS_MODARITHTOARITH_MODARITHTOARITH_TD_
