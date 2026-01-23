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

#include "prime_ir/Dialect/EllipticCurve/Python/EllipticCurveTypes.h"

namespace nb = nanobind;
using namespace mlir::python;

namespace mlir::prime_ir::elliptic_curve::python {

// static
void PyAffineType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyAttribute &curve, DefaultingPyMlirContext context) -> PyAffineType {
        MlirType t = primeIRAffineTypeGet(context->get(), curve);
        return PyAffineType(context->getRef(), t);
      },
      nb::arg("curve"), nb::arg("context") = nb::none(),
      "Create an affine point type with the given curve attribute");
  c.def_prop_ro(
      "curve",
      [](PyAffineType &self) -> PyAttribute {
        return PyAttribute(self.getContext(), primeIRAffineTypeGetCurve(self));
      },
      "Returns the curve attribute of the affine point type");
}

// static
void PyJacobianType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyAttribute &curve,
         DefaultingPyMlirContext context) -> PyJacobianType {
        MlirType t = primeIRJacobianTypeGet(context->get(), curve);
        return PyJacobianType(context->getRef(), t);
      },
      nb::arg("curve"), nb::arg("context") = nb::none(),
      "Create a jacobian point type with the given curve attribute");
  c.def_prop_ro(
      "curve",
      [](PyJacobianType &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRJacobianTypeGetCurve(self));
      },
      "Returns the curve attribute of the jacobian point type");
}

// static
void PyXYZZType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyAttribute &curve, DefaultingPyMlirContext context) -> PyXYZZType {
        MlirType t = primeIRXYZZTypeGet(context->get(), curve);
        return PyXYZZType(context->getRef(), t);
      },
      nb::arg("curve"), nb::arg("context") = nb::none(),
      "Create an xyzz point type with the given curve attribute");
  c.def_prop_ro(
      "curve",
      [](PyXYZZType &self) -> PyAttribute {
        return PyAttribute(self.getContext(), primeIRXYZZTypeGetCurve(self));
      },
      "Returns the curve attribute of the xyzz point type");
}

void populateIRTypes(nb::module_ &m) {
  PyAffineType::bind(m);
  PyJacobianType::bind(m);
  PyXYZZType::bind(m);
}

} // namespace mlir::prime_ir::elliptic_curve::python
