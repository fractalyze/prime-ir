# Copyright 2026 The PrimeIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl.testing import absltest

from prime_ir.mlir.ir import *
from prime_ir.mlir.dialects import ring


# Both admit a 2N-th root of unity at N = 8, so the ring splits and the
# evaluation basis exists.
MODULI = [12289, 40961]
DEGREE = 8


def _degree_attr():
  return IntegerAttr.get(IntegerType.get_signless(32), DEGREE)


class RingTest(absltest.TestCase):

  def testCoeffType(self):
    with Context() as ctx, Location.unknown():
      ring.register_dialect(ctx)
      rq = ring.RqType.get(MODULI, _degree_attr())
      self.assertEqual(str(rq), "!ring.rq<[12289, 40961], 8 : i32>")
      self.assertEqual(rq.moduli, MODULI)
      self.assertEqual(int(rq.ring_degree), DEGREE)
      self.assertFalse(rq.is_eval)

  def testEvalTypePrintsItsBasis(self):
    with Context() as ctx, Location.unknown():
      ring.register_dialect(ctx)
      rq = ring.RqType.get(MODULI, _degree_attr(), is_eval=True)
      # The coefficient basis is the default spelling, so only `eval` is named.
      self.assertEqual(str(rq), "!ring.rq<[12289, 40961], 8 : i32, eval>")
      self.assertTrue(rq.is_eval)

  def testTypeRoundTripsThroughTheParser(self):
    with Context() as ctx, Location.unknown():
      ring.register_dialect(ctx)
      for is_eval in (False, True):
        built = ring.RqType.get(MODULI, _degree_attr(), is_eval=is_eval)
        self.assertEqual(Type.parse(str(built)), built)

  def _mul(self, is_eval):
    """Builds `ring.mul` on two ring-typed values and returns the module."""
    rq = ring.RqType.get(MODULI, _degree_attr(), is_eval=is_eval)
    module = Module.create()
    with InsertionPoint(module.body):
      # An unrealized cast is the cheapest way to source a value of an
      # arbitrary type; only the ring op under test needs a real builder.
      lhs = Operation.create(
          "builtin.unrealized_conversion_cast", results=[rq]
      ).results[0]
      rhs = Operation.create(
          "builtin.unrealized_conversion_cast", results=[rq]
      ).results[0]
      ring.MulOp(lhs, rhs)
    return module

  def testMulBuildsOnTheEvalBasis(self):
    with Context() as ctx, Location.unknown():
      ring.register_dialect(ctx)
      module = self._mul(is_eval=True)
      module.operation.verify()
      self.assertIn("ring.mul", str(module))

  def testMulRejectsTheCoeffBasis(self):
    with Context() as ctx, Location.unknown():
      ring.register_dialect(ctx)
      module = self._mul(is_eval=False)
      # The coefficient-basis product is a negacyclic convolution, which needs
      # the transform that lives above this dialect.
      with self.assertRaises(MLIRError):
        module.operation.verify()


if __name__ == "__main__":
  absltest.main()
