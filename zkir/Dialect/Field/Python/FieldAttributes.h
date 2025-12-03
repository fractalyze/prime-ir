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

#ifndef ZKIR_DIALECT_FIELD_PYTHON_FIELDATTRIBUTES_H_
#define ZKIR_DIALECT_FIELD_PYTHON_FIELDATTRIBUTES_H_

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/lib/Bindings/Python/IRModule.h"
#include "zkir/Dialect/Field/C/FieldAttributes.h"

namespace mlir::zkir::field::python {
class PyPrimeFieldAttr
    : public mlir::python::PyConcreteAttribute<PyPrimeFieldAttr> {
public:
  static constexpr IsAFunctionTy isaFunction = zkirAttrIsAPrimeField;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      zkirPrimeFieldAttrGetTypeID;
  static constexpr const char *pyClassName = "PrimeFieldAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);
};

void populateIRAttributes(nanobind::module_ &m);

} // namespace mlir::zkir::field::python

#endif // ZKIR_DIALECT_FIELD_PYTHON_FIELDATTRIBUTES_H_
