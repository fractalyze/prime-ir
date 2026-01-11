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

#include "prime_ir/Dialect/Field/Python/FieldTypes.h"

namespace nb = nanobind;
using namespace mlir::python;

namespace mlir::prime_ir::field::python {

// static
void PyPrimeFieldType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyAttribute &modulus, bool isMontgomery,
         DefaultingPyMlirContext context) -> PyPrimeFieldType {
        MlirType t =
            primeIRPrimeFieldTypeGet(context->get(), modulus, isMontgomery);
        return PyPrimeFieldType(context->getRef(), t);
      },
      nb::arg("modulus"), nb::arg("is_montgomery"),
      nb::arg("context") = nb::none(), "Create a prime field type");
  c.def_prop_ro(
      "modulus",
      [](PyPrimeFieldType &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRPrimeFieldTypeGetModulus(self));
      },
      "Returns the modulus of the prime field type");
  c.def_prop_ro(
      "is_montgomery",
      [](PyPrimeFieldType &self) -> bool {
        return primeIRPrimeFieldTypeIsMontgomery(self);
      },
      "Returns whether this is a montgomery form");
}

// static
void PyQuadraticExtensionFieldType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyPrimeFieldType &baseField, PyAttribute &nonResidue,
         DefaultingPyMlirContext context) -> PyQuadraticExtensionFieldType {
        MlirType t = primeIRQuadraticExtensionFieldTypeGet(
            context->get(), baseField, nonResidue);
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
            primeIRQuadraticExtensionFieldTypeGetBaseField(self));
      },
      "Returns the base field of the quadratic extension field type");
  c.def_prop_ro(
      "non_residue",
      [](PyQuadraticExtensionFieldType &self) -> PyAttribute {
        return PyAttribute(
            self.getContext(),
            primeIRQuadraticExtensionFieldTypeGetNonResidue(self));
      },
      "Returns the non-residue of the quadratic extension field type");
  c.def_prop_ro(
      "is_montgomery",
      [](PyQuadraticExtensionFieldType &self) -> bool {
        return primeIRQuadraticExtensionFieldTypeIsMontgomery(self);
      },
      "Returns whether this is a montgomery form");
}

// static
void PyCubicExtensionFieldType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyPrimeFieldType &baseField, PyAttribute &nonResidue,
         DefaultingPyMlirContext context) -> PyCubicExtensionFieldType {
        MlirType t = primeIRCubicExtensionFieldTypeGet(context->get(),
                                                       baseField, nonResidue);
        return PyCubicExtensionFieldType(context->getRef(), t);
      },
      nb::arg("base_field"), nb::arg("non_residue"),
      nb::arg("context") = nb::none(), "Create a cubic extension field type");
  c.def_prop_ro(
      "base_field",
      [](PyCubicExtensionFieldType &self) -> PyPrimeFieldType {
        return PyPrimeFieldType(
            self.getContext(),
            primeIRCubicExtensionFieldTypeGetBaseField(self));
      },
      "Returns the base field of the cubic extension field type");
  c.def_prop_ro(
      "non_residue",
      [](PyCubicExtensionFieldType &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRCubicExtensionFieldTypeGetNonResidue(self));
      },
      "Returns the non-residue of the cubic extension field type");
  c.def_prop_ro(
      "is_montgomery",
      [](PyCubicExtensionFieldType &self) -> bool {
        return primeIRCubicExtensionFieldTypeIsMontgomery(self);
      },
      "Returns whether this is a montgomery form");
}

// static
void PyExtensionFieldType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](unsigned degree, PyPrimeFieldType &baseField, PyAttribute &nonResidue,
         DefaultingPyMlirContext context) -> PyExtensionFieldType {
        MlirType t = primeIRExtensionFieldTypeGet(context->get(), degree,
                                                  baseField, nonResidue);
        return PyExtensionFieldType(context->getRef(), t);
      },
      nb::arg("degree"), nb::arg("base_field"), nb::arg("non_residue"),
      nb::arg("context") = nb::none(), "Create an extension field type");
  c.def_prop_ro(
      "degree",
      [](PyExtensionFieldType &self) -> unsigned {
        return primeIRExtensionFieldTypeGetDegree(self);
      },
      "Returns the degree of the extension field type");
  c.def_prop_ro(
      "base_field",
      [](PyExtensionFieldType &self) -> PyPrimeFieldType {
        return PyPrimeFieldType(self.getContext(),
                                primeIRExtensionFieldTypeGetBaseField(self));
      },
      "Returns the base field of the extension field type");
  c.def_prop_ro(
      "non_residue",
      [](PyExtensionFieldType &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRExtensionFieldTypeGetNonResidue(self));
      },
      "Returns the non-residue of the extension field type");
  c.def_prop_ro(
      "is_montgomery",
      [](PyExtensionFieldType &self) -> bool {
        return primeIRExtensionFieldTypeIsMontgomery(self);
      },
      "Returns whether this is a montgomery form");
}

void populateIRTypes(nb::module_ &m) {
  PyPrimeFieldType::bind(m);
  PyQuadraticExtensionFieldType::bind(m);
  PyCubicExtensionFieldType::bind(m);
  PyExtensionFieldType::bind(m);
}

} // namespace mlir::prime_ir::field::python
