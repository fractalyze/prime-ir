/* Copyright 2025 The PrimeIR Authors.

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

#ifndef PRIME_IR_DIALECT_FIELD_PYTHON_FIELDTYPES_H_
#define PRIME_IR_DIALECT_FIELD_PYTHON_FIELDTYPES_H_

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "prime_ir/Dialect/Field/C/FieldTypes.h"

namespace mlir::prime_ir::field::python {

using ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType;

class PyPrimeFieldType : public PyConcreteType<PyPrimeFieldType> {
public:
  static constexpr IsAFunctionTy isaFunction = primeIRTypeIsAPrimeField;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIRPrimeFieldTypeGetTypeID;
  static constexpr const char *pyClassName = "PrimeFieldType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class PyExtensionFieldType : public PyConcreteType<PyExtensionFieldType> {
public:
  static constexpr IsAFunctionTy isaFunction = primeIRTypeIsAnExtensionField;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIRExtensionFieldTypeGetTypeID;
  static constexpr const char *pyClassName = "ExtensionFieldType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

void populateIRTypes(nanobind::module_ &m);

} // namespace mlir::prime_ir::field::python

#endif // PRIME_IR_DIALECT_FIELD_PYTHON_FIELDTYPES_H_
