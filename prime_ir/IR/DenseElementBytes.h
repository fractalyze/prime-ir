/* Copyright 2025 The PrimeIR Authors.

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

// Pure APInt <-> byte primitives backing the DenseElementTypeInterface
// (de)serialization for prime_ir field / mod_arith / elliptic-curve types.
// Header-only and dependency-free beyond LLVM ADT + MLIR builtin types so the
// base ModArith dialect can share them without depending on the field layer.
// All helpers assume a little-endian host, matching MLIR's
// DenseElementsAttr::getRawData layout.

#ifndef PRIME_IR_IR_DENSEELEMENTBYTES_H_
#define PRIME_IR_IR_DENSEELEMENTBYTES_H_

#include <cstring>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::prime_ir {

// Low `numBytes` of `v.getRawData()` as a char view. LE-host only.
inline llvm::ArrayRef<char> apintLowBytes(const llvm::APInt &v,
                                          unsigned numBytes) {
  static_assert(llvm::endianness::native == llvm::endianness::little,
                "prime_ir's APInt byte view assumes LE host");
  return {reinterpret_cast<const char *>(v.getRawData()), numBytes};
}

// Pack APInts as a LE byte buffer (`bytesPerInt` each).
inline llvm::SmallVector<char> packAPIntsLE(llvm::ArrayRef<llvm::APInt> coeffs,
                                            unsigned bytesPerInt) {
  llvm::SmallVector<char> bytes;
  bytes.reserve(coeffs.size() * bytesPerInt);
  for (const llvm::APInt &v : coeffs) {
    auto chunk = apintLowBytes(v, bytesPerInt);
    bytes.append(chunk.begin(), chunk.end());
  }
  return bytes;
}

// Decode `raw` as `total` `primeBits`-wide APInts (LE, `primeBytes` each).
// `available` is the number of source coeffs; values past index `available`
// wrap modulo it, so callers can pass an already-replicated buffer or use
// the splat-aware DenseElementsAttr overload below.
inline llvm::SmallVector<llvm::APInt, 0>
coeffsFromRawBytes(llvm::ArrayRef<char> raw, size_t available, size_t total,
                   unsigned primeBits) {
  unsigned primeBytes = (primeBits + 7) / 8;
  llvm::SmallVector<llvm::APInt, 0> coeffs;
  coeffs.reserve(total);
  if (primeBits <= 64) {
    // Single-word fast path — skips per-coeff SmallVector<uint64_t>. Hot path
    // for Babybear / Goldilocks / Mersenne31.
    for (size_t i = 0; i < total; ++i) {
      uint64_t word = 0;
      std::memcpy(&word, raw.data() + (i % available) * primeBytes, primeBytes);
      coeffs.emplace_back(primeBits, word);
    }
    return coeffs;
  }
  unsigned numWords = (primeBits + 63) / 64;
  for (size_t i = 0; i < total; ++i) {
    llvm::SmallVector<uint64_t, 4> words(numWords, 0);
    std::memcpy(words.data(), raw.data() + (i % available) * primeBytes,
                primeBytes);
    coeffs.emplace_back(primeBits, words);
  }
  return coeffs;
}

// Splat-aware DenseElementsAttr overload. `expectedTotal < 0` means
// "use whatever fits".
inline llvm::SmallVector<llvm::APInt, 0>
coeffsFromRawBytes(DenseElementsAttr dense, unsigned primeBits,
                   int64_t expectedTotal = -1) {
  unsigned primeBytes = (primeBits + 7) / 8;
  llvm::ArrayRef<char> raw = dense.getRawData();
  size_t available = raw.size() / primeBytes;
  size_t total =
      expectedTotal > 0 ? static_cast<size_t>(expectedTotal) : available;
  return coeffsFromRawBytes(raw, available, total, primeBits);
}

// Replicate a byte buffer `fanout` times end-to-end.
inline llvm::SmallVector<char> replicateRawBytes(llvm::ArrayRef<char> raw,
                                                 size_t fanout) {
  llvm::SmallVector<char> out;
  out.reserve(fanout * raw.size());
  for (size_t i = 0; i < fanout; ++i)
    out.append(raw.begin(), raw.end());
  return out;
}

// Single-storage-int element types (prime field / binary field / mod_arith)
// all (de)serialize one element as one storage-int's worth of LE bytes.
inline Attribute scalarIntConvertToAttribute(IntegerType storageType,
                                             llvm::ArrayRef<char> rawData) {
  unsigned bitWidth = storageType.getWidth();
  if (rawData.size() != (bitWidth + 7) / 8)
    return Attribute{};
  auto coeffs =
      coeffsFromRawBytes(rawData, /*available=*/1, /*total=*/1, bitWidth);
  return IntegerAttr::get(storageType, coeffs[0]);
}

inline llvm::LogicalResult
scalarIntConvertFromAttribute(unsigned bitWidth, Attribute attr,
                              llvm::SmallVectorImpl<char> &result) {
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return llvm::failure();
  auto bytes = apintLowBytes(intAttr.getValue(), (bitWidth + 7) / 8);
  result.append(bytes.begin(), bytes.end());
  return llvm::success();
}

} // namespace mlir::prime_ir

#endif // PRIME_IR_IR_DENSEELEMENTBYTES_H_
