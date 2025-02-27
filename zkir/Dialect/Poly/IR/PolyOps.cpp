#include "zkir/Dialect/Poly/IR/PolyOps.h"

#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/Poly/IR/PolyAttributes.h"
#include "zkir/Dialect/Poly/IR/PolyTypes.h"

namespace mlir {
namespace zkir {
namespace poly {

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute attr = polynomial::IntPolynomialAttr::parse(parser, nullptr);
  PolyType type;
  if (attr) {
    if (parser.parseColon() || parser.parseType(type)) return failure();
    polynomial::IntPolynomialAttr intPolyAttr =
        mlir::cast<polynomial::IntPolynomialAttr>(attr);

    result.addAttribute("value", UnivariatePolyAttr::get(parser.getContext(),
                                                         type, intPolyAttr));
    result.addTypes(type);
    return success();
  } else {
    return failure();
  }
}

void ConstantOp::print(OpAsmPrinter &p) {
  getValueAttr().getValue().print(p);
  p << " : ";
  p.printType(getOutput().getType());
}

}  // namespace poly
}  // namespace zkir
}  // namespace mlir
