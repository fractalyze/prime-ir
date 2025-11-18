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
