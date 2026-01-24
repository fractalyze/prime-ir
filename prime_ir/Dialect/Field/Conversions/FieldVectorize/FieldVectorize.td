/* Copyright 2026 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDVECTORIZE_FIELDVECTORIZE_TD_
#define PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDVECTORIZE_FIELDVECTORIZE_TD_

include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def FieldVectorize : Pass<"field-vectorize", "func::FuncOp"> {
  let summary = "Vectorize scalar field operations to packed SIMD form";
  let description = [{
    This pass transforms scalar field operations into vectorized (packed) form.

    Type transformations:
    - `!field.pf<...>` -> `vector<N x !field.pf<...>>`
    - `memref<M x !field.pf<...>>` -> `memref<M x vector<N x !field.pf<...>>>`
    - `tensor<M x !field.pf<...>>` -> `tensor<M x vector<N x !field.pf<...>>>`

    Scalar constants are broadcast using `vector.splat`.
    All field operations naturally support vectors due to the ElementwiseMappable trait.

    Example:
    ```mlir
    // Before: scalar operations
    func.func @add(%a: !field.pf<p>, %b: !field.pf<p>) -> !field.pf<p> {
      %result = field.add %a, %b : !field.pf<p>
      return %result : !field.pf<p>
    }

    // After: vectorized operations (with --vector-width=16)
    func.func @add(%a: vector<16x!field.pf<p>>, %b: vector<16x!field.pf<p>>) -> vector<16x!field.pf<p>> {
      %result = field.add %a, %b : vector<16x!field.pf<p>>
      return %result : vector<16x!field.pf<p>>
    }
    ```
  }];

  let options = [
    Option<"vectorWidth", "vector-width", "unsigned", "16",
           "Vector width for packed field operations (default: 16 for AVX-512)">
  ];

  let dependentDialects = [
    "arith::ArithDialect",
    "vector::VectorDialect",
    "func::FuncDialect",
    "memref::MemRefDialect",
    "tensor::TensorDialect",
  ];
}

#endif  // PRIME_IR_DIALECT_FIELD_CONVERSIONS_FIELDVECTORIZE_FIELDVECTORIZE_TD_
