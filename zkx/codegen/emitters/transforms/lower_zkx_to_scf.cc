/* Copyright 2024 The OpenXLA Authors.
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
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.h"
#include "zkx/codegen/emitters/elemental_hlo_to_mlir.h"
#include "zkx/codegen/emitters/ir/zkx_ops.h"
#include "zkx/hlo/analysis/indexing_map.h"
#include "zkx/util.h"

namespace zkx::emitters {
namespace {

#define GEN_PASS_DECL_LOWERZKXTOSCFPASS
#define GEN_PASS_DEF_LOWERZKXTOSCFPASS
#define GEN_PASS_DEF_LOWERZKXLOOPSTOSCFPASS
#include "zkx/codegen/emitters/transforms/passes.h.inc"

using mlir::ImplicitLocOpBuilder;
using mlir::Location;
using mlir::OpBuilder;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::ValueRange;
using mlir::scf::IfOp;

struct RewritePredicatedInsert : mlir::OpRewritePattern<PredicatedInsertOp> {
  RewritePredicatedInsert(mlir::MLIRContext* context,
                          const LowerZkxToScfPassOptions& /*options*/)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      PredicatedInsertOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<IfOp>(
        op, op.getCondition(),
        [&](OpBuilder& b, Location loc) {
          b.create<mlir::scf::YieldOp>(
              loc, b.create<mlir::tensor::InsertOp>(
                        loc, op.getValue(), op.getDest(), op.getIndices())
                       .getResult());
        },
        [&](OpBuilder& b, Location loc) {
          b.create<mlir::scf::YieldOp>(loc, op.getDest());
        });
    return success();
  }
};

struct RewritePredicatedExtract : mlir::OpRewritePattern<PredicatedExtractOp> {
  RewritePredicatedExtract(mlir::MLIRContext* context,
                           const LowerZkxToScfPassOptions& /*options*/)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      PredicatedExtractOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<IfOp>(
        op, op.getCondition(),
        [&](OpBuilder& b, Location loc) {
          b.create<mlir::scf::YieldOp>(
              loc, b.create<mlir::tensor::ExtractOp>(loc, op.getSrc(),
                                                     op.getIndices())
                       .getResult());
        },
        [&](OpBuilder& b, Location loc) {
          b.create<mlir::scf::YieldOp>(loc, op.getFallback());
        });
    return success();
  }
};

struct RewriteShuffleReduce : mlir::OpRewritePattern<gpu::ShuffleReduceOp> {
  const int64_t warp_size;

  RewriteShuffleReduce(mlir::MLIRContext* context,
                       const LowerZkxToScfPassOptions& options)
      : OpRewritePattern(context), warp_size(options.warp_size) {}

  mlir::LogicalResult matchAndRewrite(
      gpu::ShuffleReduceOp op, mlir::PatternRewriter& rewriter) const override {
    int max_distance =
        mlir::cast<mlir::IntegerAttr>(op->getAttr("max_distance")).getInt();
    // TODO(jreiffers): Do this in a verifier.
    if (max_distance & (max_distance - 1) || max_distance >= warp_size) {
      return op->emitOpError("max_distance must be a power of 2 < warp_size");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ValueRange values = op.getOperands();
    for (int distance = max_distance; distance > 0; distance /= 2) {
      namespace ml = mlir::LLVM;
      auto shuffle_32 = [&](Value v) {
        return b
            .create<mlir::gpu::ShuffleOp>(v, distance, warp_size,
                                          mlir::gpu::ShuffleMode::DOWN)
            .getShuffleResult();
      };

      auto shuffle_int = [&](Value value) -> Value {
        auto ty = mlir::cast<mlir::IntegerType>(value.getType());
        int bit_width = ty.getWidth();
        if (bit_width == 32) {
          return shuffle_32(value);
        }
        int n_shuffles = CeilOfRatio(bit_width, 32);
        mlir::IntegerType int_ty = b.getIntegerType(bit_width);
        mlir::IntegerType padded_int_ty = b.getIntegerType(n_shuffles * 32);
        value = b.create<mlir::arith::BitcastOp>(int_ty, value);
        value = b.create<mlir::arith::ExtUIOp>(padded_int_ty, value);
        if (n_shuffles > 1) {
          // Don't generate vectors if the size is 1.
          auto vector_type = ml::getVectorType(b.getI32Type(), n_shuffles);
          value = b.create<ml::BitcastOp>(vector_type, value);
          Value result_vec = b.create<ml::UndefOp>(vector_type);
          for (int i = 0; i < n_shuffles; ++i) {
            auto idx = b.create<mlir::arith::ConstantIntOp>(i, 32);
            result_vec = b.create<ml::InsertElementOp>(
                result_vec,
                shuffle_32(b.create<ml::ExtractElementOp>(value, idx)), idx);
          }
          value = b.create<ml::BitcastOp>(padded_int_ty, result_vec);
        } else {
          value = shuffle_32(value);
        }
        value = b.create<mlir::arith::TruncIOp>(int_ty, value);
        value = b.create<mlir::arith::BitcastOp>(ty, value);
        return value;
      };

      auto shuffle = [&](Value value) -> Value {
        if (!mlir::isa<mlir::IntegerType>(value.getType())) {
          auto orig_ty = value.getType();
          unsigned bits =
              mlir::DataLayout::closest(b.getInsertionBlock()->getParentOp())
                  .getTypeSizeInBits(orig_ty);
          auto int_ty = b.getIntegerType(bits);
          // NOTE(jeong0982): Replace with a TypeInterface-based approach once
          // an IntegerBitcastable TypeInterface is added to prime-ir, so that
          // EC and other future types can provide their own bitcast logic.
          value = b.create<mlir::prime_ir::field::BitcastOp>(int_ty, value);
          value = shuffle_int(value);
          value = b.create<mlir::prime_ir::field::BitcastOp>(orig_ty, value);
          return value;
        }
        if (value.getType().isUnsignedInteger()) {
          auto ty = mlir::cast<mlir::IntegerType>(value.getType());
          mlir::IntegerType signless_ty = b.getIntegerType(ty.getWidth());
          value = b.create<mlir::UnrealizedConversionCastOp>(
                       mlir::TypeRange{signless_ty}, value)
                      .getResult(0);
          value = shuffle_int(value);
          value = b.create<mlir::UnrealizedConversionCastOp>(
                       mlir::TypeRange{ty}, value)
                      .getResult(0);
          return value;
        }
        return shuffle_int(value);
      };

      SmallVector<Value> args = values;
      for (auto value : values) {
        args.push_back(shuffle(value));
      }
      values = b.create<PureCallOp>(op.getResultTypes(),
                                    op.getCombinerAttr().getAttr(), args)
                   .getResults();
    }
    rewriter.replaceOp(op, values);
    return success();
  }
};

struct RewriteZkxLoop : mlir::OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      LoopOp op, mlir::PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    IndexingMap indexing_map = op.getIndexingMap();
    SmallVector<Value, 4> lbs, ubs, steps;
    emitters::GetLoopBoundsFromIndexingMap(b, indexing_map, lbs, ubs, steps);
    mlir::scf::LoopNest loop_nest = mlir::scf::buildLoopNest(
        b, loc, lbs, ubs, steps, op.getInits(),
        [&](OpBuilder& nested_builder, Location loc, ValueRange symbol_values,
            ValueRange iter_args) -> mlir::scf::ValueVector {
          ImplicitLocOpBuilder nested_b(loc, nested_builder);
          Value is_in_bounds = emitters::CheckConstraints(
              indexing_map, op.getDims(), symbol_values, nested_b);
          auto if_op = nested_b.create<IfOp>(
              is_in_bounds,
              [&](OpBuilder& then_builder, Location then_loc) -> void {
                ImplicitLocOpBuilder then_b(then_loc, then_builder);
                mlir::IRMapping mapping;
                mapping.map(op.getInductionVars(), symbol_values);
                mapping.map(op.getIndexingMapResults(),
                            emitters::ApplyIndexing(indexing_map, op.getDims(),
                                                    symbol_values, then_b));
                mapping.map(op.getRegionIterArgs(), iter_args);
                mlir::Block* old_block = op.getBody();
                for (auto& old_op : old_block->without_terminator()) {
                  then_b.clone(old_op, mapping);
                }
                SmallVector<Value, 4> then_results;
                for (auto result : old_block->getTerminator()->getOperands()) {
                  then_results.push_back(mapping.lookupOrDefault(result));
                }
                then_b.create<mlir::scf::YieldOp>(then_results);
              },
              [&](OpBuilder& else_b, Location else_loc) {
                else_b.create<mlir::scf::YieldOp>(loc, iter_args);
              });
          return if_op.getResults();
        });
    rewriter.replaceOp(op, loop_nest.results);
    return success();
  }
};

mlir::VectorType GetThreadLevelVectorType(
    gpu::IndexedVectorType indexed_vector) {
  mlir::Type data_type = indexed_vector.getElementType();
  SmallVector<int64_t> vector_dims;
  IndexingMap map = indexed_vector.getIndexingMapAttr().getIndexingMap();
  for (Interval bound : map.GetSymbolBounds()) {
    vector_dims.push_back(bound.GetLoopTripCount());
  }
  return mlir::VectorType::get(vector_dims, data_type);
}

struct RewriteMaterialize : mlir::OpRewritePattern<gpu::MaterializeOp> {
  RewriteMaterialize(mlir::MLIRContext* context,
                     const LowerZkxToScfPassOptions& options)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      gpu::MaterializeOp op, mlir::PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    mlir::VectorType vec_type =
        GetThreadLevelVectorType(op.getResult().getType());
    mlir::Type element_type = op.getResult().getType().getElementType();
    auto zero_attr = b.getZeroAttr(vec_type);
    auto& dialect = vec_type.getElementType().getDialect();
    auto* materialize_op =
        dialect.materializeConstant(b, zero_attr, vec_type, b.getLoc());
    Value init_vec =
        materialize_op
            ? materialize_op->getResult(0)
            : b.create<mlir::arith::ConstantOp>(zero_attr).getResult();

    auto loop = b.create<LoopOp>(
        op.getMapAttr(), op.getIndices(), ValueRange{init_vec},
        [&](OpBuilder&, Location, ValueRange ivs, ValueRange map_results,
            ValueRange iter_args) {
          SmallVector<Value, 4> args(op.getInput());
          args.insert(args.end(), map_results.begin(), map_results.end());
          SmallVector<mlir::Type, 1> types{element_type};
          Value call_result =
              b.create<PureCallOp>(op.getCalleeAttr(), ValueRange{args}, types)
                  .getResult(0);
          SmallVector<mlir::OpFoldResult> offset(ivs);
          Value old_vec = iter_args.back();
          Value new_vec =
              b.create<mlir::vector::InsertOp>(call_result, old_vec, offset);
          b.create<YieldOp>(new_vec);
        });
    Value convert = b.create<mlir::UnrealizedConversionCastOp>(
                         op.getResult().getType(), loop->getResults())
                        .getResult(0);
    rewriter.replaceOp(op, convert);
    return success();
  }
};

struct RewriteInsert : mlir::OpRewritePattern<gpu::InsertOp> {
  RewriteInsert(mlir::MLIRContext* context,
                const LowerZkxToScfPassOptions& options)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      gpu::InsertOp op, mlir::PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value convert =
        b.create<mlir::UnrealizedConversionCastOp>(
             GetThreadLevelVectorType(op.getSource().getType()), op.getSource())
            .getResult(0);
    // InsertOp's map attribute (op.getMap()) is a mapping from
    //    indexed_vector index -> tensor index.
    // We get indexed_vector index by using its encoding map (source_map).
    // So we loop over indexed_vector encoding map and use the results as the
    // dimensions for InsertOp's map in order to get the final tensor index.
    IndexingMapAttr source_map = op.getSource().getType().getIndexingMapAttr();
    auto loop = b.create<LoopOp>(
        source_map, op.getIndices(), ValueRange{op.getDest()},
        [&](OpBuilder&, Location, ValueRange ivs, ValueRange map_results,
            ValueRange iter_args) {
          SmallVector<mlir::OpFoldResult> vector_offset(ivs);
          Value scalar =
              b.create<mlir::vector::ExtractOp>(convert, vector_offset)
                  .getResult();
          auto tensor_indices = b.create<ApplyIndexingOp>(
              map_results, ValueRange(), op.getMap().getIndexingMap());
          Value new_tensor = b.create<mlir::tensor::InsertOp>(
              scalar, iter_args.back(), tensor_indices.getResults());
          b.create<YieldOp>(new_tensor);
        });
    rewriter.replaceOp(op, loop->getResults());

    return success();
  }
};

class LowerZkxToScfPass
    : public impl::LowerZkxToScfPassBase<LowerZkxToScfPass> {
 public:
  explicit LowerZkxToScfPass(const LowerZkxToScfPassOptions& options)
      : options_(options) {}

  void runOnOperation() override {
    mlir::MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewritePredicatedInsert, RewritePredicatedExtract,
                 RewriteShuffleReduce, RewriteMaterialize, RewriteInsert>(
        ctx, options_);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  const LowerZkxToScfPassOptions options_;
};

class LowerZkxLoopsToScfPass
    : public impl::LowerZkxLoopsToScfPassBase<LowerZkxLoopsToScfPass> {
 public:
  void runOnOperation() override {
    mlir::MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteZkxLoop>(ctx);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerZkxToScfPass(const int64_t warp_size) {
  LowerZkxToScfPassOptions options;
  options.warp_size = warp_size;
  return std::make_unique<LowerZkxToScfPass>(options);
}

std::unique_ptr<::mlir::Pass> CreateLowerZkxLoopsToScfPass() {
  return std::make_unique<LowerZkxLoopsToScfPass>();
}

}  // namespace zkx::emitters
