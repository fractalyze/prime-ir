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

#include "zkir/Dialect/Field/Python/FieldAttributes.h"

#include "zkir/Dialect/Field/Python/FieldTypes.h"

namespace nb = nanobind;
using namespace mlir::python;

namespace mlir::zkir::field::python {

// static
void PyPrimeFieldAttr::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyPrimeFieldType &type, PyAttribute &value) -> PyPrimeFieldAttr {
        MlirAttribute attr = zkirPrimeFieldAttrGet(type, value);
        return PyPrimeFieldAttr(type.getContext(), attr);
      },
      nb::arg("type"), nb::arg("value"), "Create a prime field attribute");
  c.def_prop_ro(
      "value",
      [](PyPrimeFieldAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(), zkirPrimeFieldAttrGetValue(self));
      },
      "Returns the value of the prime field attribute");
}

void populateIRAttributes(nb::module_ &m) { PyPrimeFieldAttr::bind(m); }

} // namespace mlir::zkir::field::python
