/* Copyright 2022 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "zkx/mlir_hlo/mhlo/transforms/map_stablehlo_to_hlo_op.h"

namespace mlir::stablehlo {
namespace {

#define RETURN_CONVERTED_ENUM_ATTR(Name)                                       \
  auto stablehloValue = stringify##Name(attr.getValue());                      \
  auto hloValue = mhlo::symbolize##Name(stablehloValue);                       \
  if (!hloValue.has_value())                                                   \
    return {};                                                                 \
  return mhlo::Name##Attr::get(attr.getContext(), hloValue.value())

Attribute convertAttr(Attribute stablehloAttr) {
  // StableHLO uses DenseArray for some attributes, MHLO is in the process
  // of integrating this change. In the meantime, convert DenseArray to
  // DenseElementsAttr.
  if (auto attr = mlir::dyn_cast<DenseI64ArrayAttr>(stablehloAttr)) {
    return DenseIntElementsAttr::get(
        RankedTensorType::get(attr.getSize(), attr.getElementType()),
        attr.asArrayRef());
  }
  if (auto attr = mlir::dyn_cast<DenseBoolArrayAttr>(stablehloAttr)) {
    return DenseIntElementsAttr::get(
        RankedTensorType::get(attr.getSize(), attr.getElementType()),
        attr.asArrayRef());
  }

  // Handle StableHLO attributes.
  // The logic that handles attributes from other dialects (e.g. builtin
  // attributes) lives below.
  // TODO(chokobole): Uncomment this. Dependency: ChannelHandleAttr
  // if (auto attr = mlir::dyn_cast<ChannelHandleAttr>(stablehloAttr)) {
  //   return mhlo::ChannelHandleAttr::get(attr.getContext(), attr.getHandle(),
  //                                       attr.getType());
  // }
  if (auto attr = mlir::dyn_cast<ComparisonDirectionAttr>(stablehloAttr)) {
    RETURN_CONVERTED_ENUM_ATTR(ComparisonDirection);
  }
  // TODO(chokobole): Uncomment this. Dependency: CustomCallApiVersionAttr
  // if (auto attr = mlir::dyn_cast<CustomCallApiVersionAttr>(stablehloAttr)) {
  //   RETURN_CONVERTED_ENUM_ATTR(CustomCallApiVersion);
  // }
  if (auto attr = mlir::dyn_cast<GatherDimensionNumbersAttr>(stablehloAttr)) {
    return mhlo::GatherDimensionNumbersAttr::get(
        attr.getContext(), attr.getOffsetDims(), attr.getCollapsedSliceDims(),
        attr.getOperandBatchingDims(), attr.getStartIndicesBatchingDims(),
        attr.getStartIndexMap(), attr.getIndexVectorDim());
  }
  // TODO(chokobole): Uncomment this. Dependency: OutputOperandAliasAttr
  // if (auto attr = mlir::dyn_cast<OutputOperandAliasAttr>(stablehloAttr)) {
  //   return mhlo::OutputOperandAliasAttr::get(
  //       attr.getContext(), attr.getOutputTupleIndices(),
  //       attr.getOperandIndex(), attr.getOperandTupleIndices());
  // }
  if (auto attr = mlir::dyn_cast<ScatterDimensionNumbersAttr>(stablehloAttr)) {
    return mhlo::ScatterDimensionNumbersAttr::get(
        attr.getContext(), attr.getUpdateWindowDims(),
        attr.getInsertedWindowDims(), attr.getInputBatchingDims(),
        attr.getScatterIndicesBatchingDims(),
        attr.getScatterDimsToOperandDims(), attr.getIndexVectorDim());
  }
  if (stablehloAttr.getDialect().getNamespace() ==
      StablehloDialect::getDialectNamespace()) {
    // Our guiding principle is to support all StableHLO functionality in MHLO.
    // This check is here only for exceptional situations, e.g. when we added
    // a new StableHLO attribute and forgot to update the code above.
    return {};
  }

  // Handle non-StableHLO attributes.
  // If an attribute is not defined in StableHLO, then it is unchanged,
  // with the exception of ArrayAttr which is converted recursively.
  if (auto attrs = mlir::dyn_cast<ArrayAttr>(stablehloAttr)) {
    SmallVector<Attribute> hloAttrs;
    for (auto attr : attrs) {
      auto hloAttr = convertAttr(attr);
      if (!hloAttr)
        return {};
      hloAttrs.push_back(hloAttr);
    }
    return ArrayAttr::get(attrs.getContext(), hloAttrs);
  }
  return stablehloAttr;
}

#undef RETURN_CONVERTED_ENUM_ATTR

template <typename StablehloOpTy>
class StablehloToHloOpConverter : public OpConversionPattern<StablehloOpTy> {
public:
  using OpConversionPattern<StablehloOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StablehloOpTy stablehloOp,
                  typename StablehloOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Convert StableHLO types to HLO equivalents.
    // If a type is not defined in StableHLO, then it is unchanged,
    // with the exception of RankedTensorType and TupleType which are
    // converted recursively.
    // See `StablehloToHloTypeConverter` for more information on when this
    // conversion will succeed or fail.
    SmallVector<Type> hloTypes;
    auto typeConverter = this->getTypeConverter();
    if (failed(typeConverter->convertTypes(stablehloOp->getResultTypes(),
                                           hloTypes)))
      return failure();

    // These operands have already been converted to MHLO by
    // the dialect conversion infrastructure.
    ValueRange hloOperands = adaptor.getOperands();

    // Extensibility protocol for public MHLO features that are not yet
    // supported in StableHLO. See hlo_legalize_to_stablehlo.cc for details.
    // TODO(chokobole): Uncomment this. Dependency: CustomCallOp
    // if constexpr (std::is_same<StablehloOpTy, CustomCallOp>::value) {
    //   if (stablehloOp.getCallTargetName().starts_with("mhlo.")) {
    //     return rewriteCustomCallAsMhloOp(stablehloOp, rewriter,
    //     typeConverter,
    //                                      hloTypes, hloOperands);
    //   }
    // }

    // Convert StableHLO attributes to MHLO equivalents.
    // If an attribute is not defined in StableHLO, then it is unchanged,
    // with the exception of ArrayAttr which is converted recursively.
    SmallVector<NamedAttribute> hloAttrs;
    for (NamedAttribute stablehloAttr : stablehloOp->getAttrs()) {
      // TODO(chokobole): Uncomment this. Dependency: CustomCallOp
      // if constexpr (std::is_same<StablehloOpTy, CustomCallOp>::value) {
      //   if (stablehloAttr.getName() == "mhlo.backend_config") continue;
      // }
      Attribute hloAttr = convertAttr(stablehloAttr.getValue());
      if (!hloAttr)
        return failure();
      hloAttrs.push_back({stablehloAttr.getName(), hloAttr});
    }

    // Convert the StableHLO operation to a MHLO equivalent.
    // This can almost be done in a generic fashion, except for mhlo.case
    // that uses a variadic number of regions which means an additional argument
    // for the generic builder.
    StablehloToHloOp<StablehloOpTy> hloOp;
    if constexpr (std::is_same<StablehloOpTy, CaseOp>::value) {
      hloOp = rewriter.create<mhlo::CaseOp>(stablehloOp.getLoc(), hloTypes,
                                            hloOperands, hloAttrs,
                                            stablehloOp.getBranches().size());
    } else {
      hloOp = rewriter.create<StablehloToHloOp<StablehloOpTy>>(
          stablehloOp.getLoc(), hloTypes, hloOperands, hloAttrs);
    }

    // For backward compatibility, fix custom call with mhlo.backend_config
    // TODO(chokobole): Uncomment this. Dependency: CustomCallOp
    // if constexpr (std::is_same<StablehloOpTy, CustomCallOp>::value) {
    //   if (failed(fixupMhloBackendConfig(stablehloOp, hloOp))) return
    //   failure();
    // }

    // Finally, populate the regions while converting argument types
    // and nested operations.
    for (auto [stablehloRegion, hloRegion] :
         llvm::zip(stablehloOp->getRegions(), hloOp->getRegions())) {
      rewriter.inlineRegionBefore(stablehloRegion, hloRegion, hloRegion.end());
      if (failed(rewriter.convertRegionTypes(&hloRegion, *typeConverter,
                                             /*entryConversion=*/nullptr)))
        return failure();
    }

    rewriter.replaceOp(stablehloOp, hloOp);
    return success();
  }
};

// TODO(jeong0982): Remove this specialization once DotGeneralOp is implemented.
template <>
class StablehloToHloOpConverter<DotGeneralOp>
    : public OpConversionPattern<DotGeneralOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(DotGeneralOp op, DotGeneralOp::Adaptor,
                                ConversionPatternRewriter &) const final {
    return op.emitError("DotGeneralOp is not yet implemented in ZKX MHLO");
  }
};

template <typename... StablehloOpTypes>
void populateStablehloToHloPatterns(RewritePatternSet *patterns,
                                    TypeConverter *converter,
                                    MLIRContext *context) {
  patterns->add<StablehloToHloOpConverter<StablehloOpTypes>...>(*converter,
                                                                context);
}

} // namespace

void populateStablehloToHloPatterns(RewritePatternSet *patterns,
                                    TypeConverter *converter,
                                    MLIRContext *context) {
  // Populate conversion patterns for all StableHLO ops.
  // Our guiding principle is to support all StableHLO functionality in MHLO.
  populateStablehloToHloPatterns<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >(patterns, converter, context);
}

} // namespace mlir::stablehlo
