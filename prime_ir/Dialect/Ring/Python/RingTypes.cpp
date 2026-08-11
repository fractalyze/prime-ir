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
#include "prime_ir/Dialect/Ring/Python/RingTypes.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace mlir::prime_ir::ring::python {

// static
void PyRqType::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](const std::vector<int64_t> &moduli, PyAttribute &ringDegree,
         bool isEval, DefaultingPyMlirContext context) -> PyRqType {
        MlirType t = primeIRRqTypeGet(
            context->get(), static_cast<intptr_t>(moduli.size()), moduli.data(),
            ringDegree,
            isEval ? PRIME_IR_RING_DOMAIN_EVAL : PRIME_IR_RING_DOMAIN_COEFF);
        return PyRqType(context->getRef(), t);
      },
      nb::arg("moduli"), nb::arg("ring_degree"), nb::arg("is_eval") = false,
      nb::arg("context") = nb::none(),
      "Create the RNS ring Z_Q[X]/(X^N+1), Q the product of the moduli");
  c.def_prop_ro(
      "moduli",
      [](PyRqType &self) -> std::vector<int64_t> {
        intptr_t n = primeIRRqTypeGetNumModuli(self);
        std::vector<int64_t> moduli;
        moduli.reserve(n);
        for (intptr_t i = 0; i < n; ++i) {
          moduli.push_back(primeIRRqTypeGetModulus(self, i));
        }
        return moduli;
      },
      "Returns the RNS moduli, one per limb");
  c.def_prop_ro(
      "ring_degree",
      [](PyRqType &self) -> PyIntegerAttribute {
        return PyIntegerAttribute(self.getContext(),
                                  primeIRRqTypeGetRingDegree(self));
      },
      "Returns the ring degree N");
  c.def_prop_ro(
      "is_eval",
      [](PyRqType &self) -> bool {
        return primeIRRqTypeGetDomain(self) == PRIME_IR_RING_DOMAIN_EVAL;
      },
      "Returns whether the element is written in the evaluation basis");
}

void populateIRTypes(nb::module_ &m) { PyRqType::bind(m); }

} // namespace mlir::prime_ir::ring::python
