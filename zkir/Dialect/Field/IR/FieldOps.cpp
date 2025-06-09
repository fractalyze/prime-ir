#include "zkir/Dialect/Field/IR/FieldOps.h"

#include "zkir/Utils/APIntUtils.h"

namespace mlir::zkir::field {

PrimeFieldAttr getAttrAsStandardForm(PrimeFieldAttr attr) {
  assert(attr.getType().isMontgomery() &&
         "Expected Montgomery form for PrimeFieldAttr");

  auto standardType =
      PrimeFieldType::get(attr.getContext(), attr.getType().getModulus());
  APInt value = attr.getValue().getValue();
  APInt modulus = attr.getType().getModulus().getValue();
  auto modArithType = mod_arith::ModArithType::get(attr.getContext(),
                                                   attr.getType().getModulus());
  auto montAttr =
      mod_arith::MontgomeryAttr::get(attr.getContext(), modArithType);
  value = mulMod(value, montAttr.getRInv().getValue(), modulus);

  return PrimeFieldAttr::get(standardType, value);
}

PrimeFieldAttr getAttrAsMontgomeryForm(PrimeFieldAttr attr) {
  assert(!attr.getType().isMontgomery() &&
         "Expected standard form for PrimeFieldAttr");

  auto montType =
      PrimeFieldType::get(attr.getContext(), attr.getType().getModulus(), true);
  APInt value = attr.getValue().getValue();
  APInt modulus = attr.getType().getModulus().getValue();
  auto modArithType = mod_arith::ModArithType::get(attr.getContext(),
                                                   attr.getType().getModulus());
  auto montAttr =
      mod_arith::MontgomeryAttr::get(attr.getContext(), modArithType);
  value = mulMod(value, montAttr.getR().getValue(), modulus);

  return PrimeFieldAttr::get(montType, value);
}

Type getStandardFormType(Type type) {
  Type standardType = type;
  if (auto pfType = dyn_cast<PrimeFieldType>(getElementTypeOrSelf(type))) {
    if (pfType.isMontgomery()) {
      standardType =
          PrimeFieldType::get(type.getContext(), pfType.getModulus());
    }
  } else if (auto f2Type =
                 dyn_cast<QuadraticExtFieldType>(getElementTypeOrSelf(type))) {
    if (f2Type.getBaseField().isMontgomery()) {
      auto pfType = PrimeFieldType::get(type.getContext(),
                                        f2Type.getBaseField().getModulus());
      standardType = QuadraticExtFieldType::get(
          type.getContext(), pfType, getAttrAsStandardForm(f2Type.getBeta()));
    }
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::get(memrefType.getShape(), standardType);
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.cloneWith(shapedType.getShape(), standardType);
  } else {
    return standardType;
  }
}

Type getMontgomeryFormType(Type type) {
  Type montType = type;
  if (auto pfType = dyn_cast<PrimeFieldType>(getElementTypeOrSelf(type))) {
    if (!pfType.isMontgomery()) {
      montType =
          PrimeFieldType::get(type.getContext(), pfType.getModulus(), true);
    }
  } else if (auto f2Type =
                 dyn_cast<QuadraticExtFieldType>(getElementTypeOrSelf(type))) {
    if (!f2Type.isMontgomery()) {
      auto pfType = PrimeFieldType::get(
          type.getContext(), f2Type.getBaseField().getModulus(), true);
      montType = QuadraticExtFieldType::get(
          type.getContext(), pfType, getAttrAsMontgomeryForm(f2Type.getBeta()));
    }
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::get(memrefType.getShape(), montType);
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.cloneWith(shapedType.getShape(), montType);
  } else {
    return montType;
  }
}

}  // namespace mlir::zkir::field
