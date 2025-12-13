// Copyright 2025 The ZKIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPERATION_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPERATION_H_

#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithOperation.h"

namespace mlir::zkir::field {

class PrimeFieldOperation : public mod_arith::ModArithOperation {
public:
  PrimeFieldOperation(APInt value, PrimeFieldType type)
      : ModArithOperation(value, convertPrimeFieldType(type)) {}
  PrimeFieldOperation(IntegerAttr attr, PrimeFieldType type)
      : ModArithOperation(attr, convertPrimeFieldType(type)) {}
};

} // namespace mlir::zkir::field

#endif // ZKIR_DIALECT_FIELD_IR_FIELDOPERATION_H_
