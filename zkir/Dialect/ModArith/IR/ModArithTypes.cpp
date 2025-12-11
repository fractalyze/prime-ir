/* Copyright 2025 The ZKIR Authors.

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

#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"

#include "zkir/Utils/AssemblyFormatUtils.h"

namespace mlir::zkir::mod_arith {
Type ModArithType::parse(AsmParser &parser) {
  return parseModulus<ModArithType>(parser);
}

void ModArithType::print(AsmPrinter &printer) const {
  printModulus(printer, getModulus().getValue(), getStorageType(),
               isMontgomery());
}

llvm::TypeSize ModArithType::getTypeSizeInBits(
    mlir::DataLayout const &,
    llvm::ArrayRef<mlir::DataLayoutEntryInterface>) const {
  return llvm::TypeSize::getFixed(getStorageBitWidth());
}

uint64_t
ModArithType::getABIAlignment(DataLayout const &dataLayout,
                              llvm::ArrayRef<DataLayoutEntryInterface>) const {
  return dataLayout.getTypeABIAlignment(getStorageType());
}
} // namespace mlir::zkir::mod_arith
