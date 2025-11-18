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

#ifndef ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_
#define ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_

include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def EllipticCurveToField : Pass<"elliptic-curve-to-field", "ModuleOp"> {
  let summary = "Lower `elliptic_curve` to `field` dialect.";

  let description = [{
    This pass lowers the `elliptic_curve` dialect to their `field` equivalents.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "tensor::TensorDialect",
    "zkir::field::FieldDialect",
  ];
}

#endif  // ZKIR_DIALECT_ELLIPTICURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_ELLIPTICCURVETOFIELD_TD_
