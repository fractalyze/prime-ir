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

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "prime_ir/Dialect/EllipticCurve/C/EllipticCurveDialect.h"
#include "prime_ir/Dialect/EllipticCurve/Python/EllipticCurveAttributes.h"
#include "prime_ir/Dialect/EllipticCurve/Python/EllipticCurveTypes.h"

namespace nb = nanobind;

NB_MODULE(_elliptic_curve, m) {
  m.doc() = "PrimeIR EllipticCurve dialect";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__elliptic_curve__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);
  mlir::prime_ir::elliptic_curve::python::populateIRAttributes(m);
  mlir::prime_ir::elliptic_curve::python::populateIRTypes(m);
}
