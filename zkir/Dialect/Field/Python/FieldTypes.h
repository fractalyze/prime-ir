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

#ifndef ZKIR_DIALECT_FIELD_PYTHON_FIELDTYPES_H_
#define ZKIR_DIALECT_FIELD_PYTHON_FIELDTYPES_H_

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/lib/Bindings/Python/IRModule.h"
#include "zkir/Dialect/Field/C/FieldTypes.h"

namespace mlir::zkir::field::python {

class PyPrimeFieldType : public mlir::python::PyConcreteType<PyPrimeFieldType> {
public:
  static constexpr IsAFunctionTy isaFunction = zkirTypeIsAPrimeField;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      zkirPrimeFieldTypeGetTypeID;
  static constexpr const char *pyClassName = "PrimeFieldType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class PyQuadraticExtensionFieldType
    : public mlir::python::PyConcreteType<PyQuadraticExtensionFieldType> {
public:
  static constexpr IsAFunctionTy isaFunction =
      zkirTypeIsAQuadraticExtensionField;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      zkirQuadraticExtensionFieldTypeGetTypeID;
  static constexpr const char *pyClassName = "QuadraticExtensionFieldType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class PyCubicExtensionFieldType
    : public mlir::python::PyConcreteType<PyCubicExtensionFieldType> {
public:
  static constexpr IsAFunctionTy isaFunction = zkirTypeIsACubicExtensionField;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      zkirCubicExtensionFieldTypeGetTypeID;
  static constexpr const char *pyClassName = "CubicExtensionFieldType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

void populateIRTypes(nanobind::module_ &m);

} // namespace mlir::zkir::field::python

#endif // ZKIR_DIALECT_FIELD_PYTHON_FIELDTYPES_H_
