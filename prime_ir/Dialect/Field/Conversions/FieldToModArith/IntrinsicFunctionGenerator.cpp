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

#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/IntrinsicFunctionGenerator.h"

#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/ExtensionFieldCodeGen.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldCodeGen.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/PrimeFieldCodeGen.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Utils/BuilderContext.h"

namespace mlir::prime_ir::field {

IntrinsicFunctionGenerator::IntrinsicFunctionGenerator(
    ModuleOp module, const TypeConverter *converter)
    : module(module), converter(converter), symbolTable(module) {}

bool IntrinsicFunctionGenerator::shouldUseIntrinsic(Type fieldType,
                                                    LoweringMode mode) {
  if (mode == LoweringMode::Inline)
    return false;

  auto efType = dyn_cast<ExtensionFieldType>(fieldType);
  if (!efType)
    return false;

  // Both Intrinsic and Auto modes use intrinsic for quartic (degree >= 4)
  return efType.getDegree() >= 4;
}

std::string
IntrinsicFunctionGenerator::mangleFunctionName(StringRef baseName,
                                               ExtensionFieldType type) {
  // Create a unique name based on the type
  // Format: __prime_ir_<baseName>_<degree>_<modulus_hash>
  auto baseField = cast<PrimeFieldType>(type.getBaseField());
  APInt modulus = baseField.getModulus().getValue();

  // Use a simple hash of the modulus to keep names manageable
  uint64_t modulusHash = modulus.getLimitedValue();
  if (modulus.getBitWidth() > 64) {
    // For larger moduli, use a combination of high and low bits
    modulusHash ^= modulus.extractBits(64, 64).getLimitedValue();
  }

  return ("__prime_ir_" + baseName + "_" + Twine(type.getDegree()) + "_" +
          Twine(modulusHash))
      .str();
}

MemRefType
IntrinsicFunctionGenerator::getCoeffMemRefType(ExtensionFieldType type) {
  auto baseField = cast<PrimeFieldType>(type.getBaseField());
  Type convertedBaseType = converter->convertType(baseField);
  return MemRefType::get({static_cast<int64_t>(type.getDegree())},
                         convertedBaseType);
}

func::FuncOp IntrinsicFunctionGenerator::getOrCreateQuarticMulFunction(
    ExtensionFieldType type) {
  std::string funcName = mangleFunctionName("ext4_mul", type);

  // Check if the function already exists
  if (auto existingFunc = symbolTable.lookup<func::FuncOp>(funcName)) {
    return existingFunc;
  }

  // Create the function signature with out-parameters
  // func @__prime_ir_ext4_mul(...)(memref<4xT>, memref<4xT>, memref<4xT>)
  MemRefType coeffMemRefType = getCoeffMemRefType(type);
  SmallVector<Type, 3> inputTypes = {coeffMemRefType, coeffMemRefType,
                                     coeffMemRefType};
  auto funcType = FunctionType::get(module.getContext(), inputTypes, {});

  // Create function at module level
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto func = builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
  func.setPrivate();

  // Generate function body
  generateQuarticMulBody(func, type);

  // Add to symbol table
  symbolTable.insert(func);
  return func;
}

func::FuncOp IntrinsicFunctionGenerator::getOrCreateQuarticSquareFunction(
    ExtensionFieldType type) {
  std::string funcName = mangleFunctionName("ext4_square", type);

  if (auto existingFunc = symbolTable.lookup<func::FuncOp>(funcName)) {
    return existingFunc;
  }

  // func @__prime_ir_ext4_square(...)(memref<4xT>, memref<4xT>)
  MemRefType coeffMemRefType = getCoeffMemRefType(type);
  SmallVector<Type, 2> inputTypes = {coeffMemRefType, coeffMemRefType};
  auto funcType = FunctionType::get(module.getContext(), inputTypes, {});

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto func = builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
  func.setPrivate();

  generateQuarticSquareBody(func, type);

  symbolTable.insert(func);
  return func;
}

func::FuncOp IntrinsicFunctionGenerator::getOrCreateQuarticInverseFunction(
    ExtensionFieldType type) {
  std::string funcName = mangleFunctionName("ext4_inverse", type);

  if (auto existingFunc = symbolTable.lookup<func::FuncOp>(funcName)) {
    return existingFunc;
  }

  // func @__prime_ir_ext4_inverse(...)(memref<4xT>, memref<4xT>)
  MemRefType coeffMemRefType = getCoeffMemRefType(type);
  SmallVector<Type, 2> inputTypes = {coeffMemRefType, coeffMemRefType};
  auto funcType = FunctionType::get(module.getContext(), inputTypes, {});

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto func = builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
  func.setPrivate();

  generateQuarticInverseBody(func, type);

  symbolTable.insert(func);
  return func;
}

namespace {

// Helper to load quartic coefficients from a memref.
SmallVector<Value, 4> loadQuarticCoeffs(ImplicitLocOpBuilder &b, Value memRef) {
  SmallVector<Value, 4> coeffs;
  for (unsigned i = 0; i < 4; ++i) {
    Value idx = b.create<arith::ConstantIndexOp>(i);
    coeffs.push_back(b.create<memref::LoadOp>(memRef, idx));
  }
  return coeffs;
}

// Helper to store quartic coefficients to a memref.
void storeQuarticCoeffs(ImplicitLocOpBuilder &b, ResultRange coeffs,
                        Value outMemRef) {
  for (unsigned i = 0; i < 4; ++i) {
    Value idx = b.create<arith::ConstantIndexOp>(i);
    b.create<memref::StoreOp>(coeffs[i], outMemRef, idx);
  }
}

// Overload for SmallVector.
void storeQuarticCoeffs(ImplicitLocOpBuilder &b, ArrayRef<Value> coeffs,
                        Value outMemRef) {
  for (unsigned i = 0; i < 4; ++i) {
    Value idx = b.create<arith::ConstantIndexOp>(i);
    b.create<memref::StoreOp>(coeffs[i], outMemRef, idx);
  }
}

// Helper to get the non-residue constant for an extension field.
Value getNonResidue(ImplicitLocOpBuilder &b, ExtensionFieldType type) {
  auto baseField = cast<PrimeFieldType>(type.getBaseField());
  APInt nrValue = cast<IntegerAttr>(type.getNonResidue()).getValue();
  return createConst(b, baseField, nrValue.getSExtValue());
}

} // namespace

Value IntrinsicFunctionGenerator::emitQuarticMulCall(OpBuilder &builder,
                                                     Location loc,
                                                     ExtensionFieldType type,
                                                     Value lhs, Value rhs) {
  ImplicitLocOpBuilder b(loc, builder);

  // Get the function
  func::FuncOp func = getOrCreateQuarticMulFunction(type);
  MemRefType memRefType = getCoeffMemRefType(type);

  // Allocate temporary memrefs on stack
  Value aMemRef = b.create<memref::AllocaOp>(memRefType);
  Value bMemRef = b.create<memref::AllocaOp>(memRefType);
  Value outMemRef = b.create<memref::AllocaOp>(memRefType);

  // Store input coefficients
  auto lhsCoeffs = toCoeffs(b, lhs);
  auto rhsCoeffs = toCoeffs(b, rhs);
  storeQuarticCoeffs(b, SmallVector<Value>(lhsCoeffs.begin(), lhsCoeffs.end()),
                     aMemRef);
  storeQuarticCoeffs(b, SmallVector<Value>(rhsCoeffs.begin(), rhsCoeffs.end()),
                     bMemRef);

  // Call the intrinsic function
  b.create<func::CallOp>(func, ValueRange{aMemRef, bMemRef, outMemRef});

  // Load and reconstruct result
  SmallVector<Value, 4> resultCoeffs = loadQuarticCoeffs(b, outMemRef);
  return fromCoeffs(b, type, resultCoeffs);
}

Value IntrinsicFunctionGenerator::emitQuarticSquareCall(OpBuilder &builder,
                                                        Location loc,
                                                        ExtensionFieldType type,
                                                        Value input) {
  ImplicitLocOpBuilder b(loc, builder);

  func::FuncOp func = getOrCreateQuarticSquareFunction(type);
  MemRefType memRefType = getCoeffMemRefType(type);

  Value inputMemRef = b.create<memref::AllocaOp>(memRefType);
  Value outMemRef = b.create<memref::AllocaOp>(memRefType);

  auto inputCoeffs = toCoeffs(b, input);
  storeQuarticCoeffs(b,
                     SmallVector<Value>(inputCoeffs.begin(), inputCoeffs.end()),
                     inputMemRef);

  b.create<func::CallOp>(func, ValueRange{inputMemRef, outMemRef});

  SmallVector<Value, 4> resultCoeffs = loadQuarticCoeffs(b, outMemRef);
  return fromCoeffs(b, type, resultCoeffs);
}

Value IntrinsicFunctionGenerator::emitQuarticInverseCall(
    OpBuilder &builder, Location loc, ExtensionFieldType type, Value input) {
  ImplicitLocOpBuilder b(loc, builder);

  func::FuncOp func = getOrCreateQuarticInverseFunction(type);
  MemRefType memRefType = getCoeffMemRefType(type);

  Value inputMemRef = b.create<memref::AllocaOp>(memRefType);
  Value outMemRef = b.create<memref::AllocaOp>(memRefType);

  auto inputCoeffs = toCoeffs(b, input);
  storeQuarticCoeffs(b,
                     SmallVector<Value>(inputCoeffs.begin(), inputCoeffs.end()),
                     inputMemRef);

  b.create<func::CallOp>(func, ValueRange{inputMemRef, outMemRef});

  SmallVector<Value, 4> resultCoeffs = loadQuarticCoeffs(b, outMemRef);
  return fromCoeffs(b, type, resultCoeffs);
}

void IntrinsicFunctionGenerator::generateQuarticMulBody(
    func::FuncOp func, ExtensionFieldType type) {
  Block *entryBlock = func.addEntryBlock();
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(entryBlock);
  ImplicitLocOpBuilder b(func.getLoc(), builder);
  ScopedBuilderContext scopedBuilderContext(&b);

  Value aMemRef = entryBlock->getArgument(0);
  Value bMemRef = entryBlock->getArgument(1);
  Value outMemRef = entryBlock->getArgument(2);

  Value aValue = fromCoeffs(b, type, loadQuarticCoeffs(b, aMemRef));
  Value bValue = fromCoeffs(b, type, loadQuarticCoeffs(b, bMemRef));
  Value nonResidue = getNonResidue(b, type);

  QuarticExtensionFieldCodeGen aCodeGen(aValue, nonResidue);
  QuarticExtensionFieldCodeGen bCodeGen(bValue, nonResidue);
  Value result = aCodeGen * bCodeGen;

  storeQuarticCoeffs(b, toCoeffs(b, result), outMemRef);
  b.create<func::ReturnOp>();
}

void IntrinsicFunctionGenerator::generateQuarticSquareBody(
    func::FuncOp func, ExtensionFieldType type) {
  Block *entryBlock = func.addEntryBlock();
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(entryBlock);
  ImplicitLocOpBuilder b(func.getLoc(), builder);
  ScopedBuilderContext scopedBuilderContext(&b);

  Value inputMemRef = entryBlock->getArgument(0);
  Value outMemRef = entryBlock->getArgument(1);

  Value inputValue = fromCoeffs(b, type, loadQuarticCoeffs(b, inputMemRef));
  Value nonResidue = getNonResidue(b, type);

  QuarticExtensionFieldCodeGen codeGen(inputValue, nonResidue);
  Value result = codeGen.Square();

  storeQuarticCoeffs(b, toCoeffs(b, result), outMemRef);
  b.create<func::ReturnOp>();
}

void IntrinsicFunctionGenerator::generateQuarticInverseBody(
    func::FuncOp func, ExtensionFieldType type) {
  Block *entryBlock = func.addEntryBlock();
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(entryBlock);
  ImplicitLocOpBuilder b(func.getLoc(), builder);
  ScopedBuilderContext scopedBuilderContext(&b);

  Value inputMemRef = entryBlock->getArgument(0);
  Value outMemRef = entryBlock->getArgument(1);

  Value inputValue = fromCoeffs(b, type, loadQuarticCoeffs(b, inputMemRef));
  Value nonResidue = getNonResidue(b, type);

  QuarticExtensionFieldCodeGen codeGen(inputValue, nonResidue);
  Value result = codeGen.Inverse();

  storeQuarticCoeffs(b, toCoeffs(b, result), outMemRef);
  b.create<func::ReturnOp>();
}

} // namespace mlir::prime_ir::field
