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

#include <ostream>
#include <string>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

namespace {
// Build APInt from 4x u64 words (little-endian) and stream it to std::ostream.
void printI256Elem(std::ostream &os, const uint64_t *words4) {
  llvm::APInt v(256,
                llvm::ArrayRef<uint64_t>(words4, 4)); // little-endian words
  std::string s;
  llvm::raw_string_ostream rso(s);
  rso << v; // uses llvm::raw_ostream<<APInt
  rso.flush();
  os << s;
}

void printRec(std::ostream &os,
              const uint64_t *baseWords, // points to raw words
              int64_t dim, int64_t rank,
              int64_t offsetElems, // element offset (not bytes)
              const int64_t *sizes, const int64_t *strides) {
  constexpr int kWordsPerElem = 256 / 64;

  if (dim == 0) {
    const uint64_t *elemWords = baseWords + offsetElems * kWordsPerElem;
    printI256Elem(os, elemWords);
    return;
  }

  // First
  os << "[";
  printRec(os, baseWords, dim - 1, rank, offsetElems, sizes + 1, strides + 1);
  if (sizes[0] <= 1) {
    os << "]";
    return;
  }
  os << ", ";
  if (dim > 1)
    os << "\n";

  // Middles
  for (int64_t i = 1; i + 1 < sizes[0]; ++i) {
    impl::printSpace(os, rank - dim + 1);
    printRec(os, baseWords, dim - 1, rank, offsetElems + i * strides[0],
             sizes + 1, strides + 1);
    os << ", ";
    if (dim > 1)
      os << "\n";
  }

  // Last
  impl::printSpace(os, rank - dim + 1);
  printRec(os, baseWords, dim - 1, rank,
           offsetElems + (sizes[0] - 1) * strides[0], sizes + 1, strides + 1);
  os << "]";
}
} // namespace

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_printMemrefI256(UnrankedMemRefType<void> *mVoid) {
  // Reinterpret the unranked descriptor as if it were for i256 elements.
  UnrankedMemRefType<char> *m =
      reinterpret_cast<UnrankedMemRefType<char> *>(mVoid);

  DynamicMemRefType<char> dm(*m);
  printMemRefMetaData(std::cout, dm);
  std::cout << " data = \n";
  if (dm.rank == 0)
    std::cout << "[";

  // Base as words
  const uint64_t *baseWords = reinterpret_cast<const uint64_t *>(dm.data);
  printRec(std::cout, baseWords, dm.rank, dm.rank, dm.offset, dm.sizes,
           dm.strides);

  if (dm.rank == 0)
    std::cout << "]";
  std::cout << std::endl;
}

extern "C" MLIR_RUNNERUTILS_EXPORT void printMemrefI256(int64_t rank,
                                                        void *ptr) {
  // The mlir c-interface passes us an unranked memref descriptor; we forward
  // it.
  UnrankedMemRefType<void> desc{rank, ptr};
  _mlir_ciface_printMemrefI256(&desc);
}
