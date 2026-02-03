/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#include "zkx/service/elemental_ir_emitter.h"

#include <algorithm>
#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/mlir/codegen_utils.h"
#include "zkx/mlir/mlir_utils.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"

namespace zkx {

// static
bool ElementalIrEmitter::OpInvalidatesCache(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kConcatenate:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitUnaryOp(
    const HloInstruction* instr, mlir::Value value) {
  Shape shape = instr->operand(0)->shape();
  PrimitiveType operand_type = shape.element_type();
  if (ShapeUtil::ElementIsIntegral(shape)) {
    if (instr->opcode() == HloOpcode::kConvert) {
      Shape output_shape = instr->shape();
      if (ShapeUtil::ElementIsIntegral(output_shape)) {
        return EmitIntegerUnaryOp(
            instr, value, primitive_util::IsSignedIntegralType(operand_type));
      } else if (ShapeUtil::ElementIsField(output_shape)) {
        return EmitFieldUnaryOp(instr, value);
      } else {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported output type for integer convert operation: %s",
            output_shape.ToString()));
      }
    }
    return EmitIntegerUnaryOp(
        instr, value, primitive_util::IsSignedIntegralType(operand_type));
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldUnaryOp(instr, value);
  } else if (ShapeUtil::ElementIsEcPoint(shape)) {
    return EmitEcPointUnaryOp(instr, value);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unhandled primitive type: %s",
      primitive_util::LowercasePrimitiveTypeName(operand_type)));
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitIntegerUnaryOp(
    const HloInstruction* instr, mlir::Value value, bool is_signed) {
  switch (instr->opcode()) {
    case HloOpcode::kAbs:
      return b_->create<mlir::math::AbsIOp>(value);
    case HloOpcode::kBitcastConvert: {
      mlir::Type ret_type =
          mlir_utils::ShapeToMlirTensorType(instr->shape(), context_);
      return b_->create<mlir::arith::BitcastOp>(ret_type, value);
    }
    case HloOpcode::kClz:
      return b_->create<mlir::math::CountLeadingZerosOp>(value);
    case HloOpcode::kConvert: {
      mlir::Type ret_type =
          mlir_utils::ShapeToMlirTensorType(instr->shape(), context_);
      return mlir_utils::ConvertInteger(*b_, {ret_type}, value.getType(),
                                        ret_type, {value}, is_signed);
    }
    case HloOpcode::kNegate:
      return b_->create<mlir::arith::SubIOp>(
          mlir_utils::GetConstantOrSplat(
              *b_, value.getType(),
              b_->getZeroAttr(getElementTypeOrSelf(value.getType()))),
          value);
    case HloOpcode::kNot:
      return b_->create<mlir::arith::XOrIOp>(
          value,
          mlir_utils::GetConstantOrSplat(
              *b_, value.getType(),
              b_->getIntegerAttr(getElementTypeOrSelf(value.getType()), -1)));
    case HloOpcode::kPopulationCount:
      return b_->create<mlir::math::CtPopOp>(value);
    case HloOpcode::kSign:
      return mlir_utils::SignInteger(*b_, value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary integer op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitFieldUnaryOp(
    const HloInstruction* instr, mlir::Value value) {
  switch (instr->opcode()) {
    case HloOpcode::kConvert: {
      mlir::Type ret_type =
          mlir_utils::ShapeToMlirTensorType(instr->shape(), context_);
      return mlir_utils::ConvertField(*b_, {ret_type}, value.getType(),
                                      ret_type, {value});
    }
    case HloOpcode::kInverse:
      return b_->create<mlir::prime_ir::field::InverseOp>(value);
    case HloOpcode::kNegate:
      return b_->create<mlir::prime_ir::field::NegateOp>(value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitEcPointUnaryOp(
    const HloInstruction* instr, mlir::Value value) {
  switch (instr->opcode()) {
    case HloOpcode::kConvert: {
      mlir::Type ret_type =
          mlir_utils::ShapeToMlirTensorType(instr->shape(), context_);
      return mlir_utils::ConvertEcPoint(*b_, {ret_type}, value.getType(),
                                        ret_type, {value});
    }
    case HloOpcode::kNegate:
      return b_->create<mlir::prime_ir::elliptic_curve::NegateOp>(value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary ec point op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitBinaryOp(
    const HloInstruction* instr, mlir::Value lhs_value, mlir::Value rhs_value) {
  Shape shape = instr->operand(0)->shape();
  PrimitiveType operand_type = shape.element_type();
  if (ShapeUtil::ElementIsIntegral(shape)) {
    return EmitIntegerBinaryOp(
        instr, lhs_value, rhs_value,
        primitive_util::IsSignedIntegralType(operand_type));
  } else if (ShapeUtil::ElementIsEcPoint(shape) ||
             ShapeUtil::ElementIsEcPoint(instr->operand(1)->shape())) {
    return EmitEcPointBinaryOp(instr, lhs_value, rhs_value);
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldBinaryOp(instr, lhs_value, rhs_value);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unhandled primitive type: %s",
      primitive_util::LowercasePrimitiveTypeName(operand_type)));
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitIntegerBinaryOp(
    const HloInstruction* instr, mlir::Value lhs_value, mlir::Value rhs_value,
    bool is_signed) {
  switch (instr->opcode()) {
    // TODO(jingyue): add the "nsw" attribute for signed types.
    case HloOpcode::kAdd:
      return b_->create<mlir::arith::AddIOp>(lhs_value, rhs_value);
    case HloOpcode::kAnd:
      return b_->create<mlir::arith::AndIOp>(lhs_value, rhs_value);
    case HloOpcode::kCompare:
      return b_->create<mlir::arith::CmpIOp>(
          mlir_utils::CreateMlirArithCmpIPredicate(
              instr->comparison_direction(), is_signed),
          lhs_value, rhs_value);
    case HloOpcode::kDivide:
      return EmitIntegerDivide(lhs_value, rhs_value, is_signed);
    case HloOpcode::kMaximum:
      return EmitIntegerMax(lhs_value, rhs_value, is_signed);
    case HloOpcode::kMinimum:
      return EmitIntegerMin(lhs_value, rhs_value, is_signed);
    case HloOpcode::kMultiply:
      return b_->create<mlir::arith::MulIOp>(lhs_value, rhs_value);
    case HloOpcode::kOr:
      return b_->create<mlir::arith::OrIOp>(lhs_value, rhs_value);
    case HloOpcode::kPower: {
      return b_->create<mlir::math::IPowIOp>(lhs_value, rhs_value);
    }
    case HloOpcode::kRemainder:
      return EmitIntegerRemainder(lhs_value, rhs_value, is_signed);
    case HloOpcode::kShiftLeft:
      return b_->create<mlir::arith::ShLIOp>(lhs_value, rhs_value);
    case HloOpcode::kShiftRightArithmetic:
      return b_->create<mlir::arith::ShRSIOp>(lhs_value, rhs_value);
    case HloOpcode::kShiftRightLogical:
      return b_->create<mlir::arith::ShRUIOp>(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b_->create<mlir::arith::SubIOp>(lhs_value, rhs_value);
    case HloOpcode::kXor:
      return b_->create<mlir::arith::XOrIOp>(lhs_value, rhs_value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary integer op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitFieldBinaryOp(
    const HloInstruction* instr, mlir::Value lhs_value, mlir::Value rhs_value) {
  switch (instr->opcode()) {
    case HloOpcode::kAdd:
      return b_->create<mlir::prime_ir::field::AddOp>(lhs_value, rhs_value);
    case HloOpcode::kCompare:
      return EmitFieldCompare(instr->comparison_direction(), lhs_value,
                              rhs_value);
    case HloOpcode::kDivide: {
      auto inv = b_->create<mlir::prime_ir::field::InverseOp>(rhs_value);
      return b_->create<mlir::prime_ir::field::MulOp>(lhs_value, inv);
    }
    case HloOpcode::kMaximum:
      return EmitFieldMax(lhs_value, rhs_value);
    case HloOpcode::kMinimum:
      return EmitFieldMin(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return b_->create<mlir::prime_ir::field::MulOp>(lhs_value, rhs_value);
    case HloOpcode::kPower: {
      const PrimitiveType exponent_type =
          instr->operand(1)->shape().element_type();
      if (!primitive_util::IsUnsignedIntegralType(exponent_type)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "The exponent for a power operation on a field must be an "
            "unsigned integer, but got %s",
            primitive_util::LowercasePrimitiveTypeName(exponent_type)));
      }
      return b_->create<mlir::prime_ir::field::PowUIOp>(lhs_value.getType(),
                                                        lhs_value, rhs_value);
    }
    case HloOpcode::kSubtract:
      return b_->create<mlir::prime_ir::field::SubOp>(lhs_value, rhs_value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitEcPointBinaryOp(
    const HloInstruction* instr, mlir::Value lhs_value, mlir::Value rhs_value) {
  const Shape& shape = instr->shape();
  mlir::Type ret_type =
      mlir_utils::ShapeToMlirTensorType(shape, b_->getContext());

  switch (instr->opcode()) {
    case HloOpcode::kAdd:
      return b_->create<mlir::prime_ir::elliptic_curve::AddOp>(
          ret_type, lhs_value, rhs_value);
    case HloOpcode::kCompare:
      if (instr->comparison_direction() != ComparisonDirection::kEq &&
          instr->comparison_direction() != ComparisonDirection::kNe) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported comparison direction for EC points: %s",
            ComparisonDirectionToString(instr->comparison_direction())));
      }
      return b_->create<mlir::prime_ir::elliptic_curve::CmpOp>(
          mlir_utils::CreateMlirArithCmpIPredicate(
              instr->comparison_direction(), false),
          lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return b_->create<mlir::prime_ir::elliptic_curve::ScalarMulOp>(
          ret_type, lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b_->create<mlir::prime_ir::elliptic_curve::SubOp>(
          ret_type, lhs_value, rhs_value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary ec point op %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitTernaryOp(
    const HloInstruction* instr, mlir::Value value1, mlir::Value value2,
    mlir::Value value3) {
  Shape shape =
      instr->operand(instr->opcode() == HloOpcode::kSelect ? 1 : 0)->shape();
  PrimitiveType operand_type = shape.element_type();
  if (ShapeUtil::ElementIsIntegral(shape)) {
    return EmitIntegerTernaryOp(
        instr, value1, value2, value3,
        primitive_util::IsSignedIntegralType(operand_type));
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldTernaryOp(instr, value1, value2, value3);
  } else if (ShapeUtil::ElementIsEcPoint(shape)) {
    return EmitEcPointTernaryOp(instr, value1, value2, value3);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unhandled primitive type: %s",
      primitive_util::LowercasePrimitiveTypeName(operand_type)));
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitIntegerTernaryOp(
    const HloInstruction* instr, mlir::Value value1, mlir::Value value2,
    mlir::Value value3, bool is_signed) {
  switch (instr->opcode()) {
    case HloOpcode::kClamp:
      return EmitIntegerMin(value1, EmitIntegerMax(value1, value2, is_signed),
                            is_signed);
    case HloOpcode::kSelect:
      return b_->create<mlir::arith::SelectOp>(value1, value2, value3);
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled ternary integer op: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitFieldTernaryOp(
    const HloInstruction* instr, mlir::Value value1, mlir::Value value2,
    mlir::Value value3) {
  switch (instr->opcode()) {
    case HloOpcode::kClamp:
      return EmitFieldMin(EmitFieldMax(value1, value2), value3);
    case HloOpcode::kSelect:
      return b_->create<mlir::arith::SelectOp>(value1, value2, value3);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled ternary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitEcPointTernaryOp(
    const HloInstruction* instr, mlir::Value value1, mlir::Value value2,
    mlir::Value value3) {
  switch (instr->opcode()) {
    case HloOpcode::kSelect:
      return b_->create<mlir::arith::SelectOp>(value1, value2, value3);
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled ternary ec point op: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

mlir::Value ElementalIrEmitter::EmitIntegerDivide(mlir::Value lhs,
                                                  mlir::Value rhs,
                                                  bool is_signed) {
  if (is_signed) {
    return b_->create<mlir::arith::DivSIOp>(lhs, rhs);
  } else {
    return b_->create<mlir::arith::DivUIOp>(lhs, rhs);
  }
}

mlir::Value ElementalIrEmitter::EmitIntegerRemainder(mlir::Value lhs,
                                                     mlir::Value rhs,
                                                     bool is_signed) {
  if (is_signed) {
    return b_->create<mlir::arith::RemSIOp>(lhs, rhs);
  } else {
    return b_->create<mlir::arith::RemUIOp>(lhs, rhs);
  }
}

mlir::Value ElementalIrEmitter::EmitIntegerMax(mlir::Value lhs, mlir::Value rhs,
                                               bool is_signed) {
  if (is_signed) {
    return b_->create<mlir::arith::MaxSIOp>(lhs, rhs);
  } else {
    return b_->create<mlir::arith::MaxUIOp>(lhs, rhs);
  }
}

mlir::Value ElementalIrEmitter::EmitIntegerMin(mlir::Value lhs, mlir::Value rhs,
                                               bool is_signed) {
  if (is_signed) {
    return b_->create<mlir::arith::MinSIOp>(lhs, rhs);
  } else {
    return b_->create<mlir::arith::MinUIOp>(lhs, rhs);
  }
}

mlir::Value ElementalIrEmitter::EmitFieldCompare(ComparisonDirection direction,
                                                 mlir::Value lhs,
                                                 mlir::Value rhs) {
  return b_->create<mlir::prime_ir::field::CmpOp>(
      mlir_utils::CreateMlirArithCmpIPredicate(direction, false), lhs, rhs);
}

mlir::Value ElementalIrEmitter::EmitFieldMax(mlir::Value lhs, mlir::Value rhs) {
  auto ge = EmitFieldCompare(ComparisonDirection::kGe, lhs, rhs);
  return b_->create<mlir::arith::SelectOp>(ge, lhs, rhs);
}

mlir::Value ElementalIrEmitter::EmitFieldMin(mlir::Value lhs, mlir::Value rhs) {
  auto le = EmitFieldCompare(ComparisonDirection::kLe, lhs, rhs);
  return b_->create<mlir::arith::SelectOp>(le, lhs, rhs);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalConcatenate(
    const HloInstruction* instr, mlir::ValueRange inputs) {
  pass_flag_.enable_expand_strided_metadata = true;

  return b_->create<mlir::tensor::ConcatOp>(instr->concatenate_dimension(),
                                            inputs);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalBroadcast(
    const HloInstruction* instr, mlir::Value input,
    absl::Span<const int64_t> source_dimensions) {
  pass_flag_.enable_linalg_to_parallel_loops = true;

  int64_t rank = instr->shape().rank();
  llvm::SmallVector<int64_t> target_dimensions;
  target_dimensions.reserve(rank - source_dimensions.size());
  absl::flat_hash_set<int64_t> source_set(source_dimensions.begin(),
                                          source_dimensions.end());
  for (int64_t i = 0; i < rank; ++i) {
    if (source_set.find(i) == source_set.end()) {
      target_dimensions.push_back(i);
    }
  }

  auto init = b_->create<mlir::tensor::EmptyOp>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b_->getContext()),
      mlir::ValueRange{});

  auto broadcast =
      b_->create<mlir::linalg::BroadcastOp>(input, init, target_dimensions);
  return broadcast.getResult()[0];
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalMatvec(
    const HloInstruction* instr, mlir::Value lhs, mlir::Value rhs) {
  if (LayoutUtil::IsDenseArray(instr->operand(0)->shape())) {
    return absl::UnimplementedError(
        "Dense matrix vector multiplication is not supported");
  }
  if (!LayoutUtil::IsCSRArray(instr->operand(0)->shape())) {
    return absl::UnimplementedError(
        "Only CSR matrix vector multiplication is supported");
  }
  pass_flag_.enable_linalg_to_parallel_loops = true;

  mlir::MLIRContext* ctx = lhs.getContext();
  auto result_type = mlir::cast<mlir::RankedTensorType>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), ctx));
  llvm::SmallVector<int64_t> shapes;
  for (int64_t i = 0; i < instr->shape().dimensions_size(); ++i) {
    shapes.push_back(instr->shape().dimensions(i));
  }
  auto result =
      b_->create<mlir::tensor::EmptyOp>(shapes, result_type.getElementType());

  auto d0 = b_->getAffineDimExpr(0);
  auto d1 = b_->getAffineDimExpr(1);

  auto generic_op = b_->create<mlir::linalg::GenericOp>(
      /*resultTensorTypes=*/mlir::TypeRange{result_type},
      /*inputs=*/mlir::ValueRange{lhs, rhs},
      /*outputs=*/mlir::ValueRange{result},
      /*indexingMaps=*/
      llvm::SmallVector<mlir::AffineMap>{
          mlir::AffineMap::get(2, 0, {d0, d1}, ctx),
          mlir::AffineMap::get(2, 0, {d1}, ctx),
          mlir::AffineMap::get(2, 0, {d0}, ctx),
      },
      /*iteratorTypes=*/
      llvm::SmallVector<mlir::utils::IteratorType>{
          mlir::utils::IteratorType::parallel,
          mlir::utils::IteratorType::reduction,
      },
      /*doc=*/"matrix vector multiplication",
      /*libraryCall=*/mlir::StringRef(),
      [](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) {
        mlir::ImplicitLocOpBuilder b(loc, builder);
        auto x = args[0];
        auto y = args[1];
        auto z = args[2];

        auto mul_op =
            b.create<mlir::sparse_tensor::BinaryOp>(x.getType(), x, y);
        {
          mlir::Region& overlap_region = mul_op.getOverlapRegion();
          mlir::Block* block = b.createBlock(&overlap_region);
          block->addArguments({x.getType(), y.getType()}, {loc, loc});
          b.setInsertionPointToStart(block);
          auto mul = b.create<mlir::prime_ir::field::MulOp>(
              block->getArgument(0), block->getArgument(1));
          b.create<mlir::sparse_tensor::YieldOp>(mul);
        }

        b.setInsertionPointAfter(mul_op);
        auto reduce_op = b.create<mlir::sparse_tensor::ReduceOp>(
            mul_op.getType(), mul_op, z,
            /*identity=*/
            b.create<mlir::prime_ir::field::ConstantOp>(mul_op.getType(), 0));
        {
          mlir::Region& reduce_region = reduce_op.getRegion();
          mlir::Block* block = b.createBlock(&reduce_region);
          block->addArguments({mul_op.getType(), z.getType()}, {loc, loc});
          b.setInsertionPointToStart(block);
          auto add = b.create<mlir::prime_ir::field::AddOp>(
              block->getArgument(0), block->getArgument(1));
          b.create<mlir::sparse_tensor::YieldOp>(add);
        }

        b.setInsertionPointAfter(reduce_op);
        b.create<mlir::linalg::YieldOp>(
            mlir::ValueRange{reduce_op.getOutput()});
      });
  return generic_op.getResult(0);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalDot(
    const HloInstruction* instr, mlir::Value lhs, mlir::Value rhs) {
  int64_t rank0 = instr->operand(0)->shape().rank();
  int64_t rank1 = instr->operand(1)->shape().rank();

  if (rank0 == 1) {
    if (rank1 == 1) {
      return absl::UnimplementedError(
          "Dot op with vector and vector is not supported");
    } else if (rank1 == 2) {
      return absl::UnimplementedError(
          "Dot op with vector and matrix is not supported");
    }
  } else if (rank0 == 2) {
    if (rank1 == 1) {
      return EmitElementalMatvec(instr, lhs, rhs);
    } else if (rank1 == 2) {
      return absl::UnimplementedError(
          "Dot op with matrix and matrix is not supported");
    }
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Dot op with rank %d and rank %d is not supported", rank0, rank1));
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalDynamicSlice(
    const HloInstruction* instr, mlir::Value input,
    mlir::ValueRange start_indices) {
  pass_flag_.enable_expand_strided_metadata = true;

  llvm::SmallVector<int64_t> static_offsets;
  llvm::SmallVector<int64_t> static_strides;
  for (size_t i = 0; i < instr->shape().rank(); ++i) {
    static_offsets.push_back(mlir::ShapedType::kDynamic);
    static_strides.push_back(1);
  }

  return b_->create<mlir::tensor::ExtractSliceOp>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b_->getContext()),
      input, start_indices, /*sizes=*/mlir::ValueRange{},
      /*strides=*/mlir::ValueRange{},
      /*static_offsets=*/static_offsets, instr->dynamic_slice_sizes(),
      static_strides);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalDynamicUpdateSlice(
    const HloInstruction* instr, mlir::Value dest, mlir::Value update,
    mlir::ValueRange start_indices) {
  pass_flag_.enable_expand_strided_metadata = true;

  llvm::SmallVector<mlir::Value> sizes;
  llvm::SmallVector<int64_t> static_offsets;
  llvm::SmallVector<int64_t> static_sizes;
  llvm::SmallVector<int64_t> static_strides;
  for (size_t i = 0; i < instr->shape().rank(); ++i) {
    sizes.push_back(b_->create<mlir::tensor::DimOp>(update, i));
    static_offsets.push_back(mlir::ShapedType::kDynamic);
    static_sizes.push_back(mlir::ShapedType::kDynamic);
    static_strides.push_back(1);
  }

  return b_->create<mlir::tensor::InsertSliceOp>(
      update, dest, start_indices, sizes,
      /*strides=*/mlir::ValueRange{},
      /*static_offsets=*/static_offsets, static_sizes, static_strides);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalIota(
    const HloInstruction* instr) {
  pass_flag_.enable_linalg_to_parallel_loops = true;
  pass_flag_.enable_lower_affine = true;

  PrimitiveType element_type = instr->shape().element_type();
  if (!(primitive_util::IsIntegralType(element_type) ||
        primitive_util::IsFieldType(element_type))) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unhandled primitive type: %s",
        primitive_util::LowercasePrimitiveTypeName(element_type)));
  }

  auto output_type =
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b_->getContext());
  bool is_signed = primitive_util::IsSignedIntegralType(element_type);

  int64_t iota_dimension = instr->iota_dimension();
  auto iota_op = b_->create<mlir::tensor::GenerateOp>(
      output_type, /*dynamicExtents=*/mlir::ValueRange{},
      [&](mlir::OpBuilder& nested_b, mlir::Location loc,
          mlir::ValueRange indices) {
        mlir::ImplicitLocOpBuilder b(loc, nested_b);

        mlir::Value value_as_index = indices[iota_dimension];
        mlir::Type element_type =
            mlir::cast<mlir::RankedTensorType>(output_type).getElementType();

        mlir::Value value;
        if (element_type.isInteger()) {
          if (is_signed) {
            value = b.create<mlir::arith::IndexCastOp>(element_type,
                                                       value_as_index);
          } else {
            value = b.create<mlir::arith::IndexCastUIOp>(element_type,
                                                         value_as_index);
          }
        } else if (auto prime_field_type =
                       mlir::dyn_cast<mlir::prime_ir::field::PrimeFieldType>(
                           element_type)) {
          value = b.create<mlir::arith::IndexCastUIOp>(
              prime_field_type.getStorageType(), value_as_index);

          if (prime_field_type.isMontgomery()) {
            value = b.create<mlir::prime_ir::field::BitcastOp>(
                mlir::prime_ir::field::getStandardFormType(element_type),
                value);
            value =
                b.create<mlir::prime_ir::field::ToMontOp>(element_type, value);
          } else {
            value =
                b.create<mlir::prime_ir::field::BitcastOp>(element_type, value);
          }
        }

        b.create<mlir::tensor::YieldOp>(value);
      });

  return iota_op.getResult();
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalPad(
    const HloInstruction* instr, mlir::Value input, mlir::Value padding_value) {
  pass_flag_.enable_tensor_to_linalg = true;
  pass_flag_.enable_expand_strided_metadata = true;

  llvm::SmallVector<mlir::OpFoldResult> lower_edge_padding_low;
  llvm::SmallVector<mlir::OpFoldResult> lower_edge_padding_high;
  const PaddingConfig& padding_config = instr->padding_config();
  for (const PaddingConfig::PaddingConfigDimension& dimension :
       padding_config.dimensions()) {
    lower_edge_padding_low.push_back(
        b_->getIndexAttr(dimension.edge_padding_low()));
    lower_edge_padding_high.push_back(
        b_->getIndexAttr(dimension.edge_padding_high()));
  }

  mlir::Value padding_value_scalar =
      b_->create<mlir::tensor::ExtractOp>(padding_value);
  return b_->create<mlir::tensor::PadOp>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b_->getContext()),
      input, lower_edge_padding_low, lower_edge_padding_high,
      padding_value_scalar);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalReshape(
    const HloInstruction* instr, mlir::Value input) {
  auto output_type =
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b_->getContext());
  mlir::Value shape = b_->create<mlir::arith::ConstantOp>(
      b_->getIndexTensorAttr(instr->shape().dimensions()));
  return b_->create<mlir::tensor::ReshapeOp>(output_type, input, shape);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalReverse(
    const HloInstruction* instr, mlir::Value input) {
  pass_flag_.enable_linalg_to_parallel_loops = true;
  pass_flag_.enable_lower_affine = true;

  CHECK(mlir::cast<mlir::RankedTensorType>(input.getType()).hasStaticShape());

  auto output = b_->create<mlir::tensor::EmptyOp>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b_->getContext()),
      mlir::ValueRange{});

  llvm::SmallVector<mlir::AffineExpr, 3> output_exprs;
  for (int64_t i = 0; i < instr->shape().rank(); ++i) {
    auto it =
        std::find(instr->dimensions().begin(), instr->dimensions().end(), i);
    if (it != instr->dimensions().end()) {
      output_exprs.push_back(
          b_->getAffineConstantExpr(instr->shape().dimensions(i) - 1) -
          b_->getAffineDimExpr(i));
    } else {
      output_exprs.push_back(b_->getAffineDimExpr(i));
    }
  }

  llvm::SmallVector<mlir::AffineMap, 2> indexing_maps;
  indexing_maps.push_back(mlir::AffineMap::getMultiDimIdentityMap(
      instr->shape().rank(), b_->getContext()));
  indexing_maps.push_back(mlir::AffineMap::get(instr->shape().rank(), 0,
                                               output_exprs, b_->getContext()));

  llvm::SmallVector<mlir::utils::IteratorType, 3> iterator_types;
  for (int64_t i = 0; i < instr->shape().rank(); ++i) {
    iterator_types.push_back(mlir::utils::IteratorType::parallel);
  }

  return b_
      ->create<mlir::linalg::GenericOp>(
          mlir::TypeRange{output.getType()}, mlir::ValueRange{input},
          mlir::ValueRange{output}, indexing_maps, iterator_types,
          /*doc=*/"reverse",
          /*libraryCall=*/mlir::StringRef(),
          [](mlir::OpBuilder& builder, mlir::Location loc,
             mlir::ValueRange args) {
            mlir::ImplicitLocOpBuilder b(loc, builder);
            b.create<mlir::linalg::YieldOp>(mlir::ValueRange{args[0]});
          })
      .getResult(0);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalSlice(
    const HloInstruction* instr, mlir::Value value) {
  pass_flag_.enable_expand_strided_metadata = true;

  const Shape& shape = instr->shape();

  auto result_type = mlir::cast<mlir::RankedTensorType>(
      mlir_utils::ShapeToMlirTensorType(shape, b_->getContext()));

  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;

  absl::Span<const int64_t> slices_starts = instr->slice_starts();
  absl::Span<const int64_t> slices_limits = instr->slice_limits();
  absl::Span<const int64_t> slices_strides = instr->slice_strides();

  for (int64_t i = 0; i < shape.rank(); ++i) {
    offsets.push_back(b_->getIndexAttr(slices_starts[i]));
    sizes.push_back(b_->getIndexAttr(slices_limits[i] - slices_starts[i]));
    strides.push_back(b_->getIndexAttr(slices_strides[i]));
  }

  return b_->create<mlir::tensor::ExtractSliceOp>(result_type, value, offsets,
                                                  sizes, strides);
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitElementalTranspose(
    const HloInstruction* instr, mlir::Value input) {
  pass_flag_.enable_linalg_to_parallel_loops = true;

  auto output = b_->create<mlir::tensor::EmptyOp>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b_->getContext()),
      mlir::ValueRange{});

  return b_
      ->create<mlir::linalg::TransposeOp>(input, output, instr->dimensions())
      ->getResult(0);
}

absl::StatusOr<llvm::SmallVector<mlir::Value>>
ElementalIrEmitter::EmitThreadLocalCall(const HloComputation& /*callee*/,
                                        mlir::ValueRange /*parameters*/,
                                        std::string_view /*name*/,
                                        bool /*is_reducer*/) {
  return absl::UnimplementedError("EmitThreadLocalCall is not implemented");
}

}  // namespace zkx
