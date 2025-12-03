from absl.testing import absltest

from zkir.mlir.ir import *
from zkir.mlir.dialects import field


BABYBEAR_MODULUS = 2**31 - 2**27 + 1


def _createBabybearType(ctx):
  i32 = IntegerType.get_signless(32)
  modulus = IntegerAttr.get(i32, BABYBEAR_MODULUS)
  return field.PrimeFieldType.get(modulus, True, ctx)


def _createBabybearAttribute(ctx, value):
  pf = _createBabybearType(ctx)
  i32 = IntegerType.get_signless(32)
  return field.PrimeFieldAttr.get(pf, IntegerAttr.get(i32, value))


class FieldTest(absltest.TestCase):

  def testTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)
      self.assertEqual(str(pf), "!field.pf<2013265921 : i32, true>")
      i32 = IntegerType.get_signless(32)
      # TODO(chokobole): The direct comparison to a native Python 'int' fails here:
      # self.assertEqual(int(pf.modulus), BABYBEAR_MODULUS)
      # See https://github.com/fractalyze/zkir/issues/110
      self.assertEqual(pf.modulus, IntegerAttr.get(i32, BABYBEAR_MODULUS))
      self.assertTrue(pf.is_montgomery)

  def testAttributes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)
      i32 = IntegerType.get_signless(32)
      value_attr = IntegerAttr.get(i32, 7)
      pf_attr = field.PrimeFieldAttr.get(pf, value_attr)
      self.assertEqual(
          str(pf_attr),
          "#field.pf.elem<7 : i32> : <2013265921 : i32, true> :"
          " !field.pf<2013265921 : i32, true>",
      )
      self.assertEqual(pf_attr.value, value_attr)

  def testQuadraticExtensionTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)
      non_residue_attr = _createBabybearAttribute(ctx, BABYBEAR_MODULUS - 1)
      qe_type = field.QuadraticExtensionFieldType.get(pf, non_residue_attr)
      self.assertEqual(
          str(qe_type),
          "!field.f2<<2013265921 : i32, true>, <2013265920 : i32> : <2013265921"
          " : i32, true>>",
      )
      self.assertEqual(qe_type.base_field, pf)
      self.assertEqual(qe_type.non_residue, non_residue_attr)
      self.assertTrue(qe_type.is_montgomery)

  def testCubicExtensionTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)
      non_residue_attr = _createBabybearAttribute(ctx, 2)
      ce_type = field.CubicExtensionFieldType.get(pf, non_residue_attr)
      self.assertEqual(
          str(ce_type),
          "!field.f3<<2013265921 : i32, true>, <2 : i32> : <2013265921 : i32,"
          " true>>",
      )
      self.assertEqual(ce_type.base_field, pf)
      self.assertEqual(ce_type.non_residue, non_residue_attr)
      self.assertTrue(ce_type.is_montgomery)


if __name__ == "__main__":
  absltest.main()
