#ifndef ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEATTRIBUTES_TD_
#define ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEATTRIBUTES_TD_

include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.td"
include "zkir/Dialect/Field/IR/FieldAttributes.td"

// Base class for elliptic curve attributes.
class EllipticCurve_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<EllipticCurve_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def EllipticCurve_ShortWeierstrassAttr : EllipticCurve_Attr<"ShortWeierstrass", "sw", [SameTypeOperands]> {
  let summary = "An attribute representing a short Weierstrass elliptic curve";
  let description = [{
    An elliptic curve in short weierstrass form is defined by the equation y² = x³ + ax + b
    (where a and b are finite field elements) and a generator point G in the form (x, y).

    Example:
    ```
    !PF = !field.pf<35:i32>
    #3 = #field.pf_elem<3> : !PF

    #curve = #elliptic_curve.sw<#3, #3, (#3, #3)>
    ```
  }];
  let parameters = (ins Field_PrimeFieldAttr:$a, Field_PrimeFieldAttr:$b, Field_PrimeFieldAttr:$Gx, Field_PrimeFieldAttr:$Gy);
  // TODO(ashjeong): add `Field_PrimeFieldType: $BaseField` for `getBaseField()`
  let assemblyFormat = "`<` $a`,`$b`,``(`$Gx`,`$Gy`)``>`";
}
#endif  // ZKIR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEATTRIBUTES_TD_
