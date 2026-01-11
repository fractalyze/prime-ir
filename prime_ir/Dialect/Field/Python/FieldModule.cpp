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

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "prime_ir/Dialect/Field/C/FieldDialect.h"
#include "prime_ir/Dialect/Field/Python/FieldTypes.h"

namespace nb = nanobind;

NB_MODULE(_field, m) {
  m.doc() = "PrimeIR Field dialect";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__field__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);
  mlir::prime_ir::field::python::populateIRTypes(m);
}
