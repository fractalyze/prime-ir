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

#ifndef PRIME_IR_DIALECT_RING_PYTHON_RINGTYPES_H_
#define PRIME_IR_DIALECT_RING_PYTHON_RINGTYPES_H_

#include "mlir/Bindings/Python/IRAttributes.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "prime_ir/Dialect/Ring/C/RingTypes.h"

namespace mlir::prime_ir::ring::python {

class PyRqType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyRqType> {
public:
  static constexpr IsAFunctionTy isaFunction = primeIRTypeIsARq;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIRRqTypeGetTypeID;
  static constexpr const char *pyClassName = "RqType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

void populateIRTypes(nanobind::module_ &m);

} // namespace mlir::prime_ir::ring::python

#endif // PRIME_IR_DIALECT_RING_PYTHON_RINGTYPES_H_
