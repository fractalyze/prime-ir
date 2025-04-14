#ifndef BENCHMARK_BENCHMARKUTILS_H_
#define BENCHMARK_BENCHMARKUTILS_H_

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define HOST_IS_LITTLE_ENDIAN 1
#define HOST_IS_BIG_ENDIAN 0
#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define HOST_IS_LITTLE_ENDIAN 0
#define HOST_IS_BIG_ENDIAN 1
#elif defined(_WIN32)  // Check Windows after standard macros
#define HOST_IS_LITTLE_ENDIAN 1
#define HOST_IS_BIG_ENDIAN 0
#else
#warning \
    "Cannot determine host endianness at compile time. Assuming little-endian."
#define HOST_IS_LITTLE_ENDIAN 1
#define HOST_IS_BIG_ENDIAN 0
#endif

namespace zkir {
namespace benchmark {

// For reference, see
// https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
template <typename T>
class Memref {
 public:
  Memref(size_t h, size_t w) {
    allocatedPtr = reinterpret_cast<T *>(malloc(sizeof(T) * w * h));
    alignedPtr = allocatedPtr;

    offset = 0;
    sizes[0] = h;
    sizes[1] = w;
    strides[0] = w;
    strides[1] = 1;
  }

  T *pget(size_t i, size_t j) const {
    return &alignedPtr[offset + i * strides[0] + j * strides[1]];
  }

  T get(size_t i, size_t j) const { return *pget(i, j); }

 private:
  T *allocatedPtr;
  T *alignedPtr;
  size_t offset;
  size_t sizes[2];
  size_t strides[2];
};

namespace {
// Helper to parse a single hex character (case-insensitive)
// Throws std::invalid_argument if the character is not a valid hex digit.
inline uint8_t parseHexDigit(char c) {
  if (c >= '0' && c <= '9') {
    return static_cast<uint8_t>(c - '0');
  }
  if (c >= 'a' && c <= 'f') {
    return static_cast<uint8_t>(c - 'a' + 10);
  }
  if (c >= 'A' && c <= 'F') {
    return static_cast<uint8_t>(c - 'A' + 10);
  }
  throw std::invalid_argument(
      "Invalid hexadecimal character encountered in string.");
}
}  // namespace

// Represents a large unsigned integer using an array of 64-bit limbs.
// Uses the platform's native endianness for limb storage and operations,
template <size_t kLimbCount>
struct BigInt {
  static_assert(kLimbCount > 0, "BigInt must have at least one limb.");
  uint64_t limbs[kLimbCount];

  static BigInt fromHexString(std::string_view hexStr) {
    BigInt value;
    // Prepare string view - remove optional prefix "0x" or "0X"
    if (hexStr.length() >= 2 && hexStr[0] == '0' &&
        (hexStr[1] == 'x' || hexStr[1] == 'X')) {
      hexStr.remove_prefix(2);
    }

    // Remove leading zeros
    size_t firstDigit = hexStr.find_first_not_of('0');
    if (firstDigit == std::string_view::npos) {
      // Value is 0
      value.clear();
      return value;
    }
    // Create view of the relevant digits
    std::string_view digitsView = hexStr.substr(firstDigit);
    const size_t numDigits = digitsView.length();

    // Check length against capacity
    const size_t maxDigits = kLimbCount * 16;
    if (numDigits > maxDigits) {
      throw std::overflow_error("Hex string value exceeds BigInt capacity (" +
                                std::to_string(numDigits) + " digits > " +
                                std::to_string(maxDigits) + " max).");
    }

    // Parse right-to-left, placing limbs based on host endianness
    uint64_t currentLimbValue = 0;
    int bitsInCurrentLimb = 0;
    size_t currentLimbWriteIndex = 0;

    // Determine the starting index in the limbs array based on platform
#if HOST_IS_LITTLE_ENDIAN
    // Start writing to limbs[0] (least significant limb)
    currentLimbWriteIndex = 0;
#else
    // Calculate how many limbs will be needed based on actual digits
    // and start writing to the array index corresponding to the
    // most significant limb that will be filled.
    size_t numLimbsToFill = (numDigits + 15) / 16;  // Ceiling division
    assert(numLimbsToFill <= kLimbCount &&
           "Logic error: numLimbsToFill exceeds kLimbCount");
    currentLimbWriteIndex = kLimbCount - numLimbsToFill;
#endif

    // Iterate through the relevant digits from right to left
    for (size_t i = 0; i < numDigits; ++i) {
      // Process string from right (least significant hex digit) to left
      char c = digitsView[numDigits - 1 - i];
      // parseHexDigit throws std::invalid_argument on error
      uint8_t digitValue = parseHexDigit(c);

      // Add the 4 bits of the digit to the current limb value at the correct
      // bit position
      currentLimbValue |=
          (static_cast<uint64_t>(digitValue) << bitsInCurrentLimb);
      bitsInCurrentLimb += 4;

      // If limb is full (64 bits = 16 hex digits) or it's the last digit of the
      // string
      if (bitsInCurrentLimb == 64 || i == numDigits - 1) {
        // Write the completed or final partial limb
        value.limbs[currentLimbWriteIndex] = currentLimbValue;

        // Move to the next limb index slot (index increases for both LE/BE
        // write sequences)
        currentLimbWriteIndex++;

        // Reset for next limb
        currentLimbValue = 0;
        bitsInCurrentLimb = 0;
      }
    }
    return value;
  }

  static BigInt randomLT(const BigInt &upper_bound, std::mt19937_64 &rng,
                         std::uniform_int_distribution<uint64_t> &dist) {
    // Generate a random number less than the given upper bound.
    BigInt candidate;
    for (size_t j = 0; j < kLimbCount;) {
      candidate.limbs[j] = dist(rng);
      if (candidate.limbs[j] < upper_bound.limbs[j]) {
        j++;
      }
    }
    return candidate;
  }

  static constexpr size_t getLimbCount() { return kLimbCount; }

  bool operator<(const BigInt &other) const {
#if HOST_IS_LITTLE_ENDIAN
    // Little-Endian: Compare from MOST significant limb (highest index) down
    for (int i = kLimbCount - 1; i >= 0; --i) {
      if (limbs[i] < other.limbs[i]) {
        return true;
      }
      if (limbs[i] > other.limbs[i]) {
        return false;
      }
    }
#else  // HOST_IS_BIG_ENDIAN
    // Big-Endian: Compare from MOST significant limb (lowest index) up
    for (size_t i = 0; i < kLimbCount; ++i) {
      if (limbs[i] < other.limbs[i]) {
        return true;
      }
      if (limbs[i] > other.limbs[i]) {
        return false;
      }
    }
#endif
    // Numbers are equal.
    return false;
  }

  bool operator==(const BigInt &other) const {
    for (size_t i = 0; i < kLimbCount; ++i) {
      if (limbs[i] != other.limbs[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const BigInt &other) const { return !(*this == other); }
  bool operator>(const BigInt &other) const { return other < *this; }
  bool operator<=(const BigInt &other) const { return !(other < *this); }
  bool operator>=(const BigInt &other) const { return !(*this < other); }

  void clear() { std::fill(limbs, limbs + kLimbCount, 0); }
  bool isZero() const {
    for (size_t i = 0; i < kLimbCount; ++i) {
      if (limbs[i] != 0) {
        return false;
      }
    }
    return true;
  }

  bool isOne() const {
    for (size_t i = 1; i < kLimbCount - 1; ++i) {
      if (limbs[i] != 0) {
        return false;
      }
    }
#if HOST_IS_LITTLE_ENDIAN
    return limbs[0] == 1 && limbs[kLimbCount - 1] == 0;
#else
    return limbs[0] == 0 && limbs[kLimbCount - 1] == 1;
#endif
  }

  std::string printHex() const {
    std::stringstream s;
    s << "0x";
    bool leadingZeros = true;

#if HOST_IS_BIG_ENDIAN
    for (size_t i = 0; i < kLimbCount; ++i) {
      if (leadingZeros && limbs[i] == 0 && i < kLimbCount - 1) continue;
      leadingZeros = false;
      s << std::hex << std::setw(16) << std::setfill('0') << limbs[i];
    }
#else  // HOST_IS_LITTLE_ENDIAN
    for (int i = kLimbCount - 1; i >= 0; --i) {
      if (leadingZeros && limbs[i] == 0 && i > 0) continue;
      leadingZeros = false;
      s << std::hex << std::setw(16) << std::setfill('0') << limbs[i];
    }
#endif
    // Handle case where value is exactly 0
    if (leadingZeros) s << "0";
    return s.str();
  }
};

}  // namespace benchmark
}  // namespace zkir

#endif  // BENCHMARK_BENCHMARKUTILS_H_
