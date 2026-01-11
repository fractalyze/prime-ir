from absl.testing import absltest

from prime_ir.mlir.ir import *
from prime_ir.mlir.dialects import field


BABYBEAR_MODULUS = 2**31 - 2**27 + 1
BABYBEAR_R = 2**32


def _createBabybearType(ctx):
  i32 = IntegerType.get_signless(32)
  modulus = IntegerAttr.get(i32, BABYBEAR_MODULUS)
  return field.PrimeFieldType.get(modulus, True, ctx)


def _createBabybearAttribute(value):
  i32 = IntegerType.get_signless(32)
  return IntegerAttr.get(i32, value)


class FieldTest(absltest.TestCase):

  def testTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)
      self.assertEqual(str(pf), "!field.pf<2013265921 : i32, true>")
      i32 = IntegerType.get_signless(32)
      # TODO(chokobole): The direct comparison to a native Python 'int' fails here:
      # self.assertEqual(int(pf.modulus), BABYBEAR_MODULUS)
      # See https://github.com/fractalyze/prime-ir/issues/110
      self.assertEqual(pf.modulus, IntegerAttr.get(i32, BABYBEAR_MODULUS))
      self.assertTrue(pf.is_montgomery)

  def testQuadraticExtensionTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)
      non_residue = ((BABYBEAR_MODULUS - 1) * BABYBEAR_R) % BABYBEAR_MODULUS
      non_residue_attr = _createBabybearAttribute(non_residue)
      qe_type = field.QuadraticExtensionFieldType.get(pf, non_residue_attr)
      self.assertEqual(
          str(qe_type),
          "!field.f2<!field.pf<2013265921 : i32, true>, 2013265920 : i32>",
      )
      self.assertEqual(qe_type.base_field, pf)
      self.assertEqual(qe_type.non_residue, non_residue_attr)
      self.assertTrue(qe_type.is_montgomery)

  def testCubicExtensionTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)
      non_residue = (2 * BABYBEAR_R) % BABYBEAR_MODULUS
      non_residue_attr = _createBabybearAttribute(non_residue)
      ce_type = field.CubicExtensionFieldType.get(pf, non_residue_attr)
      self.assertEqual(
          str(ce_type),
          "!field.f3<!field.pf<2013265921 : i32, true>, 2 : i32>",
      )
      self.assertEqual(ce_type.base_field, pf)
      self.assertEqual(ce_type.non_residue, non_residue_attr)
      self.assertTrue(ce_type.is_montgomery)

  def testExtensionFieldTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)

      # Test degree 2 (11 is a valid quadratic non-residue)
      non_residue_2 = (11 * BABYBEAR_R) % BABYBEAR_MODULUS
      non_residue_attr_2 = _createBabybearAttribute(non_residue_2)
      ef2_type = field.ExtensionFieldType.get(2, pf, non_residue_attr_2)
      self.assertEqual(
          str(ef2_type),
          "!field.ef<2x!field.pf<2013265921 : i32, true>, 11 : i32>",
      )
      self.assertEqual(ef2_type.degree, 2)
      self.assertEqual(ef2_type.base_field, pf)
      self.assertEqual(ef2_type.non_residue, non_residue_attr_2)
      self.assertTrue(ef2_type.is_montgomery)

      # Test degree 3 (2 is a valid cubic non-residue)
      non_residue_3 = (2 * BABYBEAR_R) % BABYBEAR_MODULUS
      non_residue_attr_3 = _createBabybearAttribute(non_residue_3)
      ef3_type = field.ExtensionFieldType.get(3, pf, non_residue_attr_3)
      self.assertEqual(
          str(ef3_type),
          "!field.ef<3x!field.pf<2013265921 : i32, true>, 2 : i32>",
      )
      self.assertEqual(ef3_type.degree, 3)
      self.assertEqual(ef3_type.base_field, pf)
      self.assertEqual(ef3_type.non_residue, non_residue_attr_3)
      self.assertTrue(ef3_type.is_montgomery)

      # Test degree 4 (11 is a valid quartic non-residue)
      non_residue_4 = (11 * BABYBEAR_R) % BABYBEAR_MODULUS
      non_residue_attr_4 = _createBabybearAttribute(non_residue_4)
      ef4_type = field.ExtensionFieldType.get(4, pf, non_residue_attr_4)
      self.assertEqual(
          str(ef4_type),
          "!field.ef<4x!field.pf<2013265921 : i32, true>, 11 : i32>",
      )
      self.assertEqual(ef4_type.degree, 4)
      self.assertEqual(ef4_type.base_field, pf)
      self.assertEqual(ef4_type.non_residue, non_residue_attr_4)
      self.assertTrue(ef4_type.is_montgomery)


if __name__ == "__main__":
  absltest.main()
