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

#include "zkir/Dialect/Field/Python/FieldTypes.h"

namespace nb = nanobind;
using namespace mlir::python;

namespace mlir::zkir::field::python {

// static
void PyPrimeFieldType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyAttribute &modulus, bool isMontgomery,
         DefaultingPyMlirContext context) -> PyPrimeFieldType {
        MlirType t =
            zkirPrimeFieldTypeGet(context->get(), modulus, isMontgomery);
        return PyPrimeFieldType(context->getRef(), t);
      },
      nb::arg("modulus"), nb::arg("is_montgomery"),
      nb::arg("context") = nb::none(), "Create a prime field type");
  c.def_prop_ro(
      "modulus",
      [](PyPrimeFieldType &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           zkirPrimeFieldTypeGetModulus(self));
      },
      "Returns the modulus of the prime field type");
  c.def_prop_ro(
      "is_montgomery",
      [](PyPrimeFieldType &self) -> bool {
        return zkirPrimeFieldTypeIsMontgomery(self);
      },
      "Returns whether this is a montgomery form");
}

// static
void PyQuadraticExtensionFieldType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyPrimeFieldType &baseField, PyAttribute &nonResidue,
         DefaultingPyMlirContext context) -> PyQuadraticExtensionFieldType {
        MlirType t = zkirQuadraticExtensionFieldTypeGet(context->get(),
                                                        baseField, nonResidue);
        return PyQuadraticExtensionFieldType(context->getRef(), t);
      },
      nb::arg("base_field"), nb::arg("non_residue"),
      nb::arg("context") = nb::none(),
      "Create a quadratic extension field type");
  c.def_prop_ro(
      "base_field",
      [](PyQuadraticExtensionFieldType &self) -> PyPrimeFieldType {
        return PyPrimeFieldType(
            self.getContext(),
            zkirQuadraticExtensionFieldTypeGetBaseField(self));
      },
      "Returns the base field of the quadratic extension field type");
  c.def_prop_ro(
      "non_residue",
      [](PyQuadraticExtensionFieldType &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           zkirQuadraticExtensionFieldTypeGetNonResidue(self));
      },
      "Returns the non-residue of the quadratic extension field type");
  c.def_prop_ro(
      "is_montgomery",
      [](PyQuadraticExtensionFieldType &self) -> bool {
        return zkirQuadraticExtensionFieldTypeIsMontgomery(self);
      },
      "Returns whether this is a montgomery form");
}

// static
void PyCubicExtensionFieldType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyPrimeFieldType &baseField, PyAttribute &nonResidue,
         DefaultingPyMlirContext context) -> PyCubicExtensionFieldType {
        MlirType t = zkirCubicExtensionFieldTypeGet(context->get(), baseField,
                                                    nonResidue);
        return PyCubicExtensionFieldType(context->getRef(), t);
      },
      nb::arg("base_field"), nb::arg("non_residue"),
      nb::arg("context") = nb::none(), "Create a cubic extension field type");
  c.def_prop_ro(
      "base_field",
      [](PyCubicExtensionFieldType &self) -> PyPrimeFieldType {
        return PyPrimeFieldType(self.getContext(),
                                zkirCubicExtensionFieldTypeGetBaseField(self));
      },
      "Returns the base field of the cubic extension field type");
  c.def_prop_ro(
      "non_residue",
      [](PyCubicExtensionFieldType &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           zkirCubicExtensionFieldTypeGetNonResidue(self));
      },
      "Returns the non-residue of the cubic extension field type");
  c.def_prop_ro(
      "is_montgomery",
      [](PyCubicExtensionFieldType &self) -> bool {
        return zkirCubicExtensionFieldTypeIsMontgomery(self);
      },
      "Returns whether this is a montgomery form");
}

void populateIRTypes(nb::module_ &m) {
  PyPrimeFieldType::bind(m);
  PyQuadraticExtensionFieldType::bind(m);
  PyCubicExtensionFieldType::bind(m);
}

} // namespace mlir::zkir::field::python
