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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/IntrinsicFunctionGenerator.h"

#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/ConversionUtils.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BitSerialAlgorithm.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::elliptic_curve {

IntrinsicFunctionGenerator::IntrinsicFunctionGenerator(ModuleOp module)
    : module(module), symbolTable(module) {}

bool IntrinsicFunctionGenerator::shouldUseIntrinsic(Type pointType,
                                                    LoweringMode mode) {
  if (mode == LoweringMode::Inline)
    return false;

  // For Intrinsic mode, always use intrinsic for scalar_mul
  if (mode == LoweringMode::Intrinsic)
    return true;

  // Auto mode: use intrinsic for points over extension fields (G2, etc.)
  auto pointTypeInterface = dyn_cast<PointTypeInterface>(pointType);
  if (!pointTypeInterface)
    return false;

  Type baseFieldType = pointTypeInterface.getBaseFieldType();
  if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType)) {
    // Use intrinsic for extension fields (G2 uses Fp2)
    return true;
  }

  // Prime field points (G1) - inline is usually fine
  return false;
}

std::string IntrinsicFunctionGenerator::mangleFunctionName(StringRef baseName,
                                                           Type pointType,
                                                           Type scalarType) {
  auto pointTypeInterface = cast<PointTypeInterface>(pointType);
  Type baseFieldType = pointTypeInterface.getBaseFieldType();
  PointKind kind = pointTypeInterface.getPointKind();

  // Build name: __prime_ir_ec_<baseName>_<pointKind>_<fieldInfo>
  std::string kindStr;
  switch (kind) {
  case PointKind::kAffine:
    kindStr = "affine";
    break;
  case PointKind::kJacobian:
    kindStr = "jacobian";
    break;
  case PointKind::kXYZZ:
    kindStr = "xyzz";
    break;
  }

  uint64_t fieldHash = 0;
  if (auto pfType = dyn_cast<field::PrimeFieldType>(baseFieldType)) {
    fieldHash = pfType.getModulus().getValue().getLimitedValue();
  } else if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType)) {
    auto baseField = cast<field::PrimeFieldType>(efType.getBaseField());
    fieldHash = baseField.getModulus().getValue().getLimitedValue();
    fieldHash ^= efType.getDegree() << 16;
  }

  return ("__prime_ir_ec_" + baseName + "_" + kindStr + "_" + Twine(fieldHash))
      .str();
}

unsigned IntrinsicFunctionGenerator::getNumBaseFieldElements(Type pointType) {
  auto pointTypeInterface = cast<PointTypeInterface>(pointType);
  Type baseFieldType = pointTypeInterface.getBaseFieldType();
  PointKind kind = pointTypeInterface.getPointKind();

  unsigned numCoords = static_cast<unsigned>(kind) + 2; // 2, 3, or 4

  unsigned fieldDegree = 1;
  if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType)) {
    fieldDegree = efType.getDegreeOverPrime();
  }

  return numCoords * fieldDegree;
}

MemRefType IntrinsicFunctionGenerator::getCoordMemRefType(Type pointType) {
  auto pointTypeInterface = cast<PointTypeInterface>(pointType);
  Type baseFieldType = pointTypeInterface.getBaseFieldType();

  // Get the ultimate base field type
  Type elementType = baseFieldType;
  if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType)) {
    elementType = efType.getBaseField();
  }

  unsigned numElements = getNumBaseFieldElements(pointType);
  return MemRefType::get({static_cast<int64_t>(numElements)}, elementType);
}

func::FuncOp
IntrinsicFunctionGenerator::getOrCreateScalarMulFunction(Type pointType,
                                                         Type scalarType) {
  std::string funcName =
      mangleFunctionName("scalar_mul", pointType, scalarType);

  // Check if the function already exists
  if (auto existingFunc = symbolTable.lookup<func::FuncOp>(funcName)) {
    return existingFunc;
  }

  // Get scalar integer type
  auto scalarFieldType = cast<field::PrimeFieldType>(scalarType);
  Type scalarIntType = scalarFieldType.getStorageType();

  // Create the function signature with out-parameters
  // func @__prime_ir_ec_scalar_mul_...(memref<N x FieldType>, IntType,
  //                                     memref<N x FieldType>)
  MemRefType coordMemRefType = getCoordMemRefType(pointType);
  SmallVector<Type, 3> inputTypes = {coordMemRefType, scalarIntType,
                                     coordMemRefType};
  auto funcType = FunctionType::get(module.getContext(), inputTypes, {});

  // Create function at module level
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto func = builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
  func.setPrivate();

  // Generate function body
  generateScalarMulBody(func, pointType, scalarType);

  // Add to symbol table
  symbolTable.insert(func);
  return func;
}

namespace {

// Helper to load point coordinates from a memref.
SmallVector<Value> loadPointCoords(ImplicitLocOpBuilder &b, Value memRef,
                                   unsigned numElements) {
  SmallVector<Value> coords;
  for (unsigned i = 0; i < numElements; ++i) {
    Value idx = b.create<arith::ConstantIndexOp>(i);
    coords.push_back(b.create<memref::LoadOp>(memRef, idx));
  }
  return coords;
}

// Helper to store point coordinates to a memref.
void storePointCoords(ImplicitLocOpBuilder &b, ArrayRef<Value> coords,
                      Value outMemRef) {
  for (unsigned i = 0; i < coords.size(); ++i) {
    Value idx = b.create<arith::ConstantIndexOp>(i);
    b.create<memref::StoreOp>(coords[i], outMemRef, idx);
  }
}

// Convert flat base field elements to point value.
Value flatCoordsToPoint(ImplicitLocOpBuilder &b, Type pointType,
                        ArrayRef<Value> flatCoords) {
  auto pointTypeInterface = cast<PointTypeInterface>(pointType);
  Type baseFieldType = pointTypeInterface.getBaseFieldType();
  PointKind kind = pointTypeInterface.getPointKind();
  unsigned numCoords = static_cast<unsigned>(kind) + 2;

  SmallVector<Value> coords;

  if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType)) {
    // Extension field: group flat coords into extension field elements
    unsigned degree = efType.getDegreeOverPrime();
    for (unsigned i = 0; i < numCoords; ++i) {
      SmallVector<Value> coeffs;
      for (unsigned j = 0; j < degree; ++j) {
        coeffs.push_back(flatCoords[i * degree + j]);
      }
      coords.push_back(b.create<field::ExtFromCoeffsOp>(baseFieldType, coeffs));
    }
  } else {
    // Prime field: direct mapping
    coords.assign(flatCoords.begin(), flatCoords.end());
  }

  return fromCoords(b, pointType, coords);
}

// Convert point value to flat base field elements.
SmallVector<Value> pointToFlatCoords(ImplicitLocOpBuilder &b, Value point) {
  auto pointTypeInterface = cast<PointTypeInterface>(point.getType());
  Type baseFieldType = pointTypeInterface.getBaseFieldType();

  Operation::result_range pointCoords = toCoords(b, point);

  SmallVector<Value> flatCoords;
  if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType)) {
    // Extension field: flatten to base field coefficients
    // Compute result types for ExtToCoeffsOp
    Type primeFieldType = efType.getBaseField();
    unsigned degree = efType.getDegree();
    SmallVector<Type> coeffTypes(degree, primeFieldType);

    for (Value coord : pointCoords) {
      auto coeffsOp = b.create<field::ExtToCoeffsOp>(coeffTypes, coord);
      for (Value coeff : coeffsOp.getResults()) {
        flatCoords.push_back(coeff);
      }
    }
  } else {
    // Prime field: direct mapping
    flatCoords.assign(pointCoords.begin(), pointCoords.end());
  }

  return flatCoords;
}

} // namespace

void IntrinsicFunctionGenerator::generateScalarMulBody(func::FuncOp func,
                                                       Type pointType,
                                                       Type scalarType) {
  Block *entryBlock = func.addEntryBlock();
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(entryBlock);
  ImplicitLocOpBuilder b(func.getLoc(), builder);
  ScopedBuilderContext scopedBuilderContext(&b);

  Value pointInMemRef = entryBlock->getArgument(0);
  Value scalarInt = entryBlock->getArgument(1);
  Value pointOutMemRef = entryBlock->getArgument(2);

  unsigned numElements = getNumBaseFieldElements(pointType);

  // Load input point
  SmallVector<Value> flatInputCoords =
      loadPointCoords(b, pointInMemRef, numElements);
  Value inputPoint = flatCoordsToPoint(b, pointType, flatInputCoords);

  // Create zero point for accumulator
  Value zeroPoint = createZeroPoint(b, pointType);

  // Implement double-and-add algorithm using shared utility
  Value finalResult = generateBitSerialLoop(
      b, scalarInt, inputPoint, zeroPoint,
      [pointType](ImplicitLocOpBuilder &b, Value v) {
        return b.create<DoubleOp>(pointType, v);
      },
      [pointType](ImplicitLocOpBuilder &b, Value acc, Value v) {
        return b.create<AddOp>(pointType, acc, v);
      });

  // Store result to output memref
  SmallVector<Value> flatOutputCoords = pointToFlatCoords(b, finalResult);
  storePointCoords(b, flatOutputCoords, pointOutMemRef);

  b.create<func::ReturnOp>();
}

Value IntrinsicFunctionGenerator::emitScalarMulCall(OpBuilder &builder,
                                                    Location loc,
                                                    Type pointType,
                                                    Type outputType,
                                                    Value point, Value scalar) {
  ImplicitLocOpBuilder b(loc, builder);

  // Get scalar field type
  Type scalarType = scalar.getType();
  auto scalarFieldType = cast<field::PrimeFieldType>(scalarType);
  Type scalarIntType = scalarFieldType.getStorageType();

  // Get or create the intrinsic function
  func::FuncOp func = getOrCreateScalarMulFunction(outputType, scalarType);
  MemRefType memRefType = getCoordMemRefType(outputType);

  // Allocate temporary memrefs on stack
  Value pointInMemRef = b.create<memref::AllocaOp>(memRefType);
  Value pointOutMemRef = b.create<memref::AllocaOp>(memRefType);

  // Convert point to output type if needed and flatten to coords
  Value convertedPoint = point;
  if (point.getType() != outputType) {
    convertedPoint = b.create<ConvertPointTypeOp>(outputType, point);
  }

  SmallVector<Value> flatInputCoords = pointToFlatCoords(b, convertedPoint);
  storePointCoords(b, flatInputCoords, pointInMemRef);

  // Convert scalar to integer if needed
  Value scalarReduced =
      scalarFieldType.isMontgomery()
          ? b.create<field::FromMontOp>(
                field::getStandardFormType(scalarFieldType), scalar)
          : scalar;
  Value scalarInt =
      b.create<field::BitcastOp>(TypeRange{scalarIntType}, scalarReduced);

  // Call the intrinsic function
  b.create<func::CallOp>(func,
                         ValueRange{pointInMemRef, scalarInt, pointOutMemRef});

  // Load result and reconstruct point
  unsigned numElements = getNumBaseFieldElements(outputType);
  SmallVector<Value> flatOutputCoords =
      loadPointCoords(b, pointOutMemRef, numElements);
  return flatCoordsToPoint(b, outputType, flatOutputCoords);
}

} // namespace mlir::prime_ir::elliptic_curve
