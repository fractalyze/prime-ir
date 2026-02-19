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
      [](PyPrimeFieldType &self) -> PyIntegerAttribute {
        return PyIntegerAttribute(self.getContext(),
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
void PyExtensionFieldType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](unsigned degree, PyType &baseField, PyAttribute &nonResidue,
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
      "Returns the immediate extension degree");
  c.def_prop_ro(
      "degree_over_prime",
      [](PyExtensionFieldType &self) -> unsigned {
        return primeIRExtensionFieldTypeGetDegreeOverPrime(self);
      },
      "Returns the total degree over the base prime field");
  c.def_prop_ro(
      "base_field",
      [](PyExtensionFieldType &self) -> nb::object {
        MlirType baseField = primeIRExtensionFieldTypeGetBaseField(self);
        if (primeIRTypeIsAnExtensionField(baseField)) {
          return nb::cast(PyExtensionFieldType(self.getContext(), baseField));
        }
        return nb::cast(PyPrimeFieldType(self.getContext(), baseField));
      },
      "Returns the base field of the extension field type");
  c.def_prop_ro(
      "base_prime_field",
      [](PyExtensionFieldType &self) -> PyPrimeFieldType {
        return PyPrimeFieldType(
            self.getContext(),
            primeIRExtensionFieldTypeGetBasePrimeField(self));
      },
      "Returns the underlying prime field at the base of the tower");
  c.def_prop_ro(
      "non_residue",
      [](PyExtensionFieldType &self) -> PyIntegerAttribute {
        return PyIntegerAttribute(self.getContext(),
                                  primeIRExtensionFieldTypeGetNonResidue(self));
      },
      "Returns the non-residue of the extension field type");
  c.def_prop_ro(
      "is_montgomery",
      [](PyExtensionFieldType &self) -> bool {
        return primeIRExtensionFieldTypeIsMontgomery(self);
      },
      "Returns whether this is a montgomery form");
  c.def_prop_ro(
      "is_tower",
      [](PyExtensionFieldType &self) -> bool {
        return primeIRExtensionFieldTypeIsTower(self);
      },
      "Returns whether this is a tower extension");
}

void populateIRTypes(nb::module_ &m) {
  PyPrimeFieldType::bind(m);
  PyExtensionFieldType::bind(m);
}

} // namespace mlir::prime_ir::field::python
