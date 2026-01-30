/* Copyright 2023 The OpenXLA Authors.
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

#ifndef ZKX_FFI_API_FFI_H_
#define ZKX_FFI_API_FFI_H_

#ifdef ZKX_FFI_FFI_H_
#error Two different ZKX FFI implementations cannot be included together. \
       See README.md for more details.
#endif  // ZKX_FFI_FFI_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>  // NOLINT
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "zk_dtypes/include/all_types.h"
#include "zkx/ffi/api/c_api.h"

// IWYU pragma: begin_exports
#include "zkx/ffi/api/api.h"
// IWYU pragma: end_exports

namespace zkx::ffi {

// All user data types that are passed via the execution context or state must
// be registered with the ZKX FFI ahead of time to get unique type id.
using TypeId = ZKX_FFI_TypeId;  // NOLINT

enum class DataType : uint8_t {
  INVALID = ZKX_FFI_DataType_INVALID,
  PRED = ZKX_FFI_DataType_PRED,
  S8 = ZKX_FFI_DataType_S8,
  S16 = ZKX_FFI_DataType_S16,
  S32 = ZKX_FFI_DataType_S32,
  S64 = ZKX_FFI_DataType_S64,
  U8 = ZKX_FFI_DataType_U8,
  U16 = ZKX_FFI_DataType_U16,
  U32 = ZKX_FFI_DataType_U32,
  U64 = ZKX_FFI_DataType_U64,
  TOKEN = ZKX_FFI_DataType_TOKEN,
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) \
  enum = ZKX_FFI_DataType_##enum,
  ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
};

// Create aliases in ::zkx::ffi namespace for all DataTypes, for consistency
// with zkx that defines PrimitiveType enums in ::zkx namespace.
inline constexpr DataType PRED = DataType::PRED;
inline constexpr DataType S8 = DataType::S8;
inline constexpr DataType S16 = DataType::S16;
inline constexpr DataType S32 = DataType::S32;
inline constexpr DataType S64 = DataType::S64;
inline constexpr DataType U8 = DataType::U8;
inline constexpr DataType U16 = DataType::U16;
inline constexpr DataType U32 = DataType::U32;
inline constexpr DataType U64 = DataType::U64;
inline constexpr DataType TOKEN = DataType::TOKEN;
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) \
  inline constexpr DataType enum = DataType::enum;
ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE

inline std::ostream& operator<<(std::ostream& os, const DataType dtype) {
  return os << static_cast<ZKX_FFI_DataType>(dtype);
}

constexpr size_t ByteWidth(DataType dtype) {
  switch (dtype) {
    case DataType::INVALID:
    case DataType::TOKEN:
      return 0;
    case DataType::PRED:
      return 1;
    case DataType::S8:
    case DataType::U8:
      return 1;
    case DataType::S16:
    case DataType::U16:
      return 2;
    case DataType::S32:
    case DataType::U32:
      return 4;
    case DataType::S64:
    case DataType::U64:
      return 8;
#define ZK_DTYPES_CASE(cpp_type, unused2, enum, unused3) \
  case DataType::enum:                                   \
    return cpp_type::kByteWidth;
      ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
  }
}

//===----------------------------------------------------------------------===//
// Span is non-owning view into contiguous values of type `T`.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Replace with `std::span` when C++20 is available.
template <typename T>
class Span {
 public:
  constexpr Span() : data_(nullptr), size_(0) {}

  Span(T* data, size_t size) : data_(data), size_(size) {}
  Span(const std::vector<std::remove_const_t<T>>& vec)  // NOLINT
      : Span(vec.data(), vec.size()) {}

  T& operator[](size_t index) const { return data_[index]; }

  bool operator==(const Span<T>& other) const {
    return size() == other.size() && std::equal(begin(), end(), other.begin());
  }

  T& front() const { return data_[0]; }
  T& back() const { return data_[size_ - 1]; }
  Span<T> first(size_t n) const { return Span<T>(data_, n); }
  Span<T> last(size_t n) const { return Span<T>(data_ + size_ - n, n); }
  size_t size() const { return size_; }

  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

 private:
  T* data_;
  size_t size_;
};

//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//

enum class ErrorCode : uint8_t {
  kOk = ZKX_FFI_Error_Code_OK,
  kCancelled = ZKX_FFI_Error_Code_CANCELLED,
  kUnknown = ZKX_FFI_Error_Code_UNKNOWN,
  kInvalidArgument = ZKX_FFI_Error_Code_INVALID_ARGUMENT,
  kDeadlineExceeded = ZKX_FFI_Error_Code_DEADLINE_EXCEEDED,
  kNotFound = ZKX_FFI_Error_Code_NOT_FOUND,
  kAlreadyExists = ZKX_FFI_Error_Code_ALREADY_EXISTS,
  kPermissionDenied = ZKX_FFI_Error_Code_PERMISSION_DENIED,
  kResourceExhausted = ZKX_FFI_Error_Code_RESOURCE_EXHAUSTED,
  kFailedPrecondition = ZKX_FFI_Error_Code_FAILED_PRECONDITION,
  kAborted = ZKX_FFI_Error_Code_ABORTED,
  kOutOfRange = ZKX_FFI_Error_Code_OUT_OF_RANGE,
  kUnimplemented = ZKX_FFI_Error_Code_UNIMPLEMENTED,
  kInternal = ZKX_FFI_Error_Code_INTERNAL,
  kUnavailable = ZKX_FFI_Error_Code_UNAVAILABLE,
  kDataLoss = ZKX_FFI_Error_Code_DATA_LOSS,
  kUnauthenticated = ZKX_FFI_Error_Code_UNAUTHENTICATED,
};

class Error {
 public:
  Error() = default;

  Error(ErrorCode errc, std::string message)
      : errc_(errc), message_(std::move(message)) {}

  Error(ZKX_FFI_Error_Code errc, std::string message)
      : Error(static_cast<ErrorCode>(errc), std::move(message)) {}

  bool success() const { return errc_ == ErrorCode::kOk; }
  bool failure() const { return !success(); }

  std::optional<ErrorCode> errc() const { return errc_; }
  const std::string& message() const { return message_; }

  static Error Success() { return Error(); }

  static Error Internal(std::string message) {
    return Error(ErrorCode::kInternal, std::move(message));
  }

  static Error InvalidArgument(std::string message) {
    return Error(ErrorCode::kInvalidArgument, std::move(message));
  }

 private:
  ErrorCode errc_ = ErrorCode::kOk;
  std::string message_;
};

//===----------------------------------------------------------------------===//
// Expected<T, E> and ErrorOr<T>
//===----------------------------------------------------------------------===//

// Forward declare.
template <typename E>
class Unexpected;

// TODO(slebedev): Replace with `std::expected` when C++23 is available.
template <typename T, typename E>
class Expected {
 public:
  constexpr Expected(T value) : data_(std::move(value)) {}  // NOLINT
  constexpr Expected(Unexpected<E> u);                      // NOLINT

  constexpr operator bool() const {  // NOLINT
    return has_value();
  }

  constexpr T& operator*() & { return value(); }
  constexpr const T& operator*() const& { return value(); }
  constexpr T&& operator*() && { return std::move(value()); }
  constexpr const T& operator*() const&& { return std::move(value()); }

  constexpr T* operator->() { return &value(); }
  constexpr const T* operator->() const { return &value(); }

  constexpr bool has_value() const { return std::holds_alternative<T>(data_); }
  constexpr bool has_error() const { return std::holds_alternative<E>(data_); }

  constexpr T& value() & { return std::get<T>(data_); }
  constexpr const T& value() const& { return std::get<T>(data_); }
  constexpr T&& value() && { return std::get<T>(std::move(data_)); }
  constexpr const T& value() const&& { return std::get<T>(std::move(data_)); }

  constexpr E& error() & { return std::get<E>(data_); }
  constexpr const E& error() const& { return std::get<E>(data_); }
  constexpr E&& error() && { return std::get<E>(std::move(data_)); }
  constexpr const E&& error() const&& { return std::get<E>(std::move(data_)); }

 private:
  std::variant<T, E> data_;
};

template <typename E>
class Unexpected {
 public:
  constexpr Unexpected(E error) : error_(std::move(error)) {}  // NOLINT

 private:
  template <typename, typename>
  friend class Expected;

  E error_;
};

Unexpected(const char*) -> Unexpected<std::string>;

template <typename T, typename E>
constexpr Expected<T, E>::Expected(Unexpected<E> u)
    : data_(std::move(u.error_)) {}

template <typename T>
class ErrorOr : public Expected<T, Error> {
 public:
  using Expected<T, Error>::Expected;
};

//===----------------------------------------------------------------------===//
// Future
//===----------------------------------------------------------------------===//

// A Promise and a Future are loosely based on `std::promise` and `std::future`,
// with an API similar to `tsl::AsyncValue`. Implementation is based on a
// simplified version of an AsyncValue with at most one waiter.

// A promise to complete execution with a success or an error.
class Promise;

// A promise that completes when a specific number of count downs have occurred.
class CountDownPromise;

// A future that becomes available when a corresponding promise is completed.
class Future {
 public:
  explicit Future(const Promise& promise);
  explicit Future(const CountDownPromise& promise);

  Future(Future&&) = default;
  Future& operator=(Future&&) = default;

  template <typename F>
  void OnReady(F&& f);

 private:
  friend class Promise;

  using Waiter = std::function<void(const std::optional<Error>& error)>;

  enum class State : uint8_t { kPending, kAvailable, kError };

  struct WaiterAndState {
    static_assert(alignof(std::max_align_t) >= 8 && sizeof(Waiter*) == 8);

    static constexpr uint64_t kStateMask = (1ull << 2) - 1;
    static constexpr uint64_t kPointerMask = ~kStateMask;

    WaiterAndState(Waiter* ptr, State state) {
      value = (reinterpret_cast<uintptr_t>(ptr) & kPointerMask) |
              (static_cast<uintptr_t>(state) & kStateMask);
    }

    WaiterAndState() : WaiterAndState(nullptr, State::kPending) {}

    State state() const { return static_cast<State>(value & kStateMask); }

    Waiter* waiter() const {
      return reinterpret_cast<Waiter*>(value & kPointerMask);
    }

    uintptr_t value;
  };

  static_assert(std::atomic<WaiterAndState>::is_always_lock_free,
                "WaiterAndState atomic must be lock-free");

  struct Data {
    std::atomic<WaiterAndState> waiter_and_state = WaiterAndState();
    std::optional<Error> error;
  };

  std::shared_ptr<Data> data_;
};

class Promise {
 public:
  Promise() : data_(std::make_shared<Future::Data>()) {}

  Promise(const Promise&) = default;
  Promise& operator=(const Promise&) = default;

  Promise(Promise&&) = default;
  Promise& operator=(Promise&&) = default;

  void SetAvailable();
  void SetError(Error error);

 private:
  friend class Future;

  void SetCompleted(Future::State state);

  std::shared_ptr<Future::Data> data_;
};

// A simple implementation of `tsl::CountDownAsyncValueRef` that is compatible
// with `ffi::Future`.
class CountDownPromise {
 public:
  CountDownPromise() = default;

  CountDownPromise(Promise promise, int64_t count)
      : state_(std::make_shared<State>(std::move(promise), count)) {
    assert(count > 0 && "Count must be positive");
  }

  explicit CountDownPromise(int64_t count)
      : CountDownPromise(Promise(), count) {}

  // Drops the count by `count` and returns true if the underlying promise
  // became available.
  bool CountDown(size_t count, const Error& error = Error::Success()) {
    assert(state_->count.load() >= count && "Invalid count down value");

    if (ZKX_FFI_PREDICT_FALSE(!error.success())) {
      const std::lock_guard<std::mutex> lock(state_->mutex);
      state_->is_error.store(true, std::memory_order_release);
      state_->error = error;
    }

    bool is_complete =
        state_->count.fetch_sub(count, std::memory_order_acq_rel) == count;
    if (ZKX_FFI_PREDICT_FALSE(is_complete)) {
      bool is_error = state_->is_error.load(std::memory_order_acquire);
      if (ZKX_FFI_PREDICT_FALSE(is_error)) {
        auto take_error = [&] {
          const std::lock_guard<std::mutex> lock(state_->mutex);
          return state_->error;
        };
        state_->promise.SetError(take_error());
        return true;
      } else {
        state_->promise.SetAvailable();
        return true;
      }
    }

    return false;
  }

  // Drops the count by `1` and returns true if the underlying promise became
  // available.
  bool CountDown(Error error = Error::Success()) { return CountDown(1, error); }

 private:
  friend class Future;

  struct State {
    State(Promise promise, int64_t count)
        : promise(std::move(promise)), count(count), is_error(false) {}

    Promise promise;
    std::atomic<int64_t> count;
    std::atomic<bool> is_error;

    std::mutex mutex;
    Error error;
  };

  std::shared_ptr<State> state_;

  const Promise& AsPromise() const { return state_->promise; }
};

inline Future::Future(const Promise& promise) : data_(promise.data_) {
  assert(data_.use_count() == 2 &&
         "Promise can be used to create at most one Future");
}

inline Future::Future(const CountDownPromise& promise)
    : Future(promise.AsPromise()) {}

template <typename F>
void Future::OnReady(F&& f) {
  static_assert(std::is_invocable_v<F, const std::optional<Error>&>,
                "F must be compatible with Waiter signature");

  WaiterAndState old_value =
      data_->waiter_and_state.load(std::memory_order_acquire);

  // If future is already completed, just run the waiter.
  if (old_value.state() != State::kPending) {
    f(data_->error);
    return;
  }

  // Otherwise, add the waiter to the future.
  auto* waiter = new Waiter(std::forward<F>(f));
  auto new_value = WaiterAndState(waiter, State::kPending);

  while (!data_->waiter_and_state.compare_exchange_weak(
      old_value, new_value, std::memory_order_acq_rel,
      std::memory_order_acquire)) {
    // Another thread completed the future, just run the waiter.
    if (old_value.state() != State::kPending) {
      assert(old_value.waiter() == nullptr);
      (*waiter)(data_->error);
      delete waiter;
      return;
    }
  }

  // If CAS succeeded the future must be in the pending state.
  assert(old_value.state() == State::kPending);
}

inline void Promise::SetAvailable() { SetCompleted(Future::State::kAvailable); }

inline void Promise::SetError(Error error) {
  assert(error.errc() != ErrorCode::kOk);
  assert(data_->error == std::nullopt);
  data_->error = std::move(error);

  SetCompleted(Future::State::kError);
}

inline void Promise::SetCompleted(Future::State state) {
  Future::WaiterAndState old_value = data_->waiter_and_state.exchange(
      {nullptr, state}, std::memory_order_acq_rel);
  assert(old_value.state() == Future::State::kPending);

  if (Future::Waiter* waiter = old_value.waiter()) {
    (*waiter)(data_->error);
    delete waiter;
  }
}

//===----------------------------------------------------------------------===//
// Arguments
//===----------------------------------------------------------------------===//

// Dynamically-typed buffer.
//
// No checks are done at decoding time. Any dtype and rank combination is
// accepted.
class AnyBuffer {
 public:
  using Dimensions = Span<const int64_t>;

  explicit AnyBuffer(const ZKX_FFI_Buffer* buf) : buf_(buf) {
    assert(buf != nullptr && "ZKX_FFI_Buffer must be non-null");
  }

  DataType element_type() const { return DataType(buf_->dtype); }

  Dimensions dimensions() const { return Dimensions(buf_->dims, buf_->rank); }

  ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE size_t size_bytes() const {
    return ByteWidth(element_type()) * element_count();
  }

  ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE size_t element_count() const {
    Dimensions dims = dimensions();
    return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                           std::multiplies<>());
  }

  void* untyped_data() const { return buf_->data; }

  template <typename T>
  T* typed_data() const {
    assert(internal::NativeTypeToCApiDataType<T>() == buf_->dtype &&
           "Template type must match the underlying buffer dtype");
    return reinterpret_cast<T*>(buf_->data);
  }

  template <typename T>
  T* reinterpret_data() const {
    assert(sizeof(T) == ByteWidth(element_type()) &&
           !(reinterpret_cast<std::uintptr_t>(buf_->data) % alignof(T)) &&
           "Requested type must have the same byte width and alignment as the "
           "underlying buffer type");
    return reinterpret_cast<T*>(buf_->data);
  }

 private:
  const ZKX_FFI_Buffer* buf_;
};

namespace internal {

// A workaround for the fact that a static_assertion can be evaluated
// whether or not the template is instantiated
template <DataType dtype>
struct always_false : std::false_type {};

template <DataType dtype>
struct DataTypeToNative {
  static_assert(always_false<dtype>::value, "unsupported data type");
};

#define ZKX_FFI_REGISTER_DATATYPE_MAPPING(data_type_value, actual_type) \
  template <>                                                           \
  struct DataTypeToNative<data_type_value> {                            \
    using type = actual_type;                                           \
  };

ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::PRED, bool);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::U8, uint8_t);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::U16, uint16_t);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::U32, uint32_t);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::U64, uint64_t);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::S8, int8_t);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::S16, int16_t);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::S32, int32_t);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::S64, int64_t);
ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::TOKEN, void);
#define ZK_DTYPES_CASE(cpp_type, unused, enum, unused2) \
  ZKX_FFI_REGISTER_DATATYPE_MAPPING(DataType::enum, cpp_type);
ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE

#undef ZKX_FFI_REGISTER_DATATYPE_MAPPING

inline constexpr size_t kDynamicRank = std::numeric_limits<size_t>::max();

}  // namespace internal

template <DataType dtype>
using NativeType = typename internal::DataTypeToNative<dtype>::type;

// Buffer with a statically-known dtype and rank.
//
// The dtype and rank are checked at decoding time. If rank is not specified,
// any rank is accepted.
template <DataType dtype, size_t rank = internal::kDynamicRank>
class Buffer {
 public:
  using Dimensions = AnyBuffer::Dimensions;

  explicit Buffer(const ZKX_FFI_Buffer* buf) : buf_(buf) {
    assert(buf_ != nullptr && "ZKX_FFI_Buffer must be non-null");
  }

  DataType element_type() const { return dtype; }

  Dimensions dimensions() const {
    return Dimensions(buf_->dims,
                      rank == internal::kDynamicRank ? buf_->rank : rank);
  }

  ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE size_t size_bytes() const {
    return ByteWidth(dtype) * element_count();
  }

  ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE size_t element_count() const {
    Dimensions dims = dimensions();
    return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                           std::multiplies<>());
  }

  void* untyped_data() const { return buf_->data; }

  NativeType<dtype>* typed_data() const {
    return reinterpret_cast<NativeType<dtype>*>(untyped_data());
  }

 private:
  const ZKX_FFI_Buffer* buf_;
};

// clang-format off
template <DataType dtype> using BufferR0 = Buffer<dtype, 0>;
template <DataType dtype> using BufferR1 = Buffer<dtype, 1>;
template <DataType dtype> using BufferR2 = Buffer<dtype, 2>;
template <DataType dtype> using BufferR3 = Buffer<dtype, 3>;
template <DataType dtype> using BufferR4 = Buffer<dtype, 4>;
// clang-format on

using Token = BufferR0<DataType::TOKEN>;  // NOLINT

namespace internal {

template <DataType dtype, size_t rank>
ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE std::optional<Buffer<dtype, rank>> DecodeBuffer(
    ZKX_FFI_Buffer* buf, DiagnosticEngine& diagnostic) {
  if (auto buf_dtype = static_cast<DataType>(buf->dtype);
      ZKX_FFI_PREDICT_FALSE(buf_dtype != dtype)) {
    return diagnostic.Emit("Wrong buffer dtype: expected ")
           << dtype << " but got " << buf_dtype;
  }

  if constexpr (rank != internal::kDynamicRank) {
    if (ZKX_FFI_PREDICT_FALSE(buf->rank != rank)) {
      return diagnostic.Emit("Wrong buffer rank: expected ")
             << rank << " but got " << buf->rank;
    }
  }

  return Buffer<dtype, rank>(buf);
}

}  // namespace internal

template <DataType dtype, size_t rank = internal::kDynamicRank>
using ResultBuffer = Result<Buffer<dtype, rank>>;

// clang-format off
template <DataType dtype> using ResultBufferR0 = ResultBuffer<dtype, 0>;
template <DataType dtype> using ResultBufferR1 = ResultBuffer<dtype, 1>;
template <DataType dtype> using ResultBufferR2 = ResultBuffer<dtype, 2>;
template <DataType dtype> using ResultBufferR3 = ResultBuffer<dtype, 3>;
template <DataType dtype> using ResultBufferR4 = ResultBuffer<dtype, 4>;
// clang-format on

//===----------------------------------------------------------------------===//
// Arguments binding
//===----------------------------------------------------------------------===//

template <>
struct ArgBinding<AnyBuffer> {
  using Arg = AnyBuffer;
};

template <DataType dtype, size_t rank>
struct ArgBinding<Buffer<dtype, rank>> {
  using Arg = Buffer<dtype, rank>;
};

//===----------------------------------------------------------------------===//
// Results binding
//===----------------------------------------------------------------------===//

template <>
struct RetBinding<Result<AnyBuffer>> {
  using Ret = AnyBuffer;
};

template <DataType dtype, size_t rank>
struct RetBinding<Result<Buffer<dtype, rank>>> {
  using Ret = Buffer<dtype, rank>;
};

//===----------------------------------------------------------------------===//
// Arguments decoding
//===----------------------------------------------------------------------===//

inline std::ostream& operator<<(std::ostream& os, const ZKX_FFI_ArgType type) {
  switch (type) {
    case ZKX_FFI_ArgType_BUFFER:
      return os << "buffer";
  }
}

template <>
struct ArgDecoding<AnyBuffer> {
  ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<AnyBuffer> Decode(ZKX_FFI_ArgType type, void* arg,
                                         DiagnosticEngine& diagnostic) {
    if (ZKX_FFI_PREDICT_FALSE(type != ZKX_FFI_ArgType_BUFFER)) {
      return diagnostic.Emit("Wrong argument type: expected ")
             << ZKX_FFI_ArgType_BUFFER << " but got " << type;
    }
    return AnyBuffer(reinterpret_cast<ZKX_FFI_Buffer*>(arg));
  }
};

template <DataType dtype, size_t rank>
struct ArgDecoding<Buffer<dtype, rank>> {
  ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Buffer<dtype, rank>> Decode(
      ZKX_FFI_ArgType type, void* arg, DiagnosticEngine& diagnostic) {
    if (ZKX_FFI_PREDICT_FALSE(type != ZKX_FFI_ArgType_BUFFER)) {
      return diagnostic.Emit("Wrong argument type: expected ")
             << ZKX_FFI_ArgType_BUFFER << " but got " << type;
    }

    return internal::DecodeBuffer<dtype, rank>(
        reinterpret_cast<ZKX_FFI_Buffer*>(arg), diagnostic);
  }
};

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing a variable number of arguments.
//===----------------------------------------------------------------------===//

class RemainingArgs : public internal::RemainingArgsBase {
 public:
  using internal::RemainingArgsBase::RemainingArgsBase;

  template <typename T>
  ErrorOr<T> get(size_t index) const {
    size_t idx = offset() + index;
    if (ZKX_FFI_PREDICT_FALSE(idx >= args()->size)) {
      return Unexpected(
          Error(ErrorCode::kInvalidArgument, "Index out of range"));
    }

    DiagnosticEngine diagnostic;
    std::optional<T> value = ArgDecoding<T>::Decode(
        args()->types[idx], args()->args[idx], diagnostic);
    if (ZKX_FFI_PREDICT_FALSE(!value.has_value())) {
      return Unexpected(Error::Internal(diagnostic.Result()));
    }

    return *value;
  }
};

template <>
struct internal::Decode<internal::RemainingArgsTag> {
  static std::optional<RemainingArgs> call(DecodingOffsets& offsets,
                                           DecodingContext& ctx,
                                           DiagnosticEngine& diagnostic) {
    return RemainingArgs(&ctx.call_frame->args, offsets.args);
  }
};

//===----------------------------------------------------------------------===//
// Results decoding
//===----------------------------------------------------------------------===//

inline std::ostream& operator<<(std::ostream& os, const ZKX_FFI_RetType type) {
  switch (type) {
    case ZKX_FFI_RetType_BUFFER:
      return os << "buffer";
  }
}

template <>
struct RetDecoding<AnyBuffer> {
  ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<AnyBuffer>> Decode(ZKX_FFI_RetType type,
                                                 void* ret,
                                                 DiagnosticEngine& diagnostic) {
    if (ZKX_FFI_PREDICT_FALSE(type != ZKX_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << ZKX_FFI_RetType_BUFFER << " but got " << type;
    }
    return AnyBuffer(reinterpret_cast<ZKX_FFI_Buffer*>(ret));
  }
};

template <DataType dtype, size_t rank>
struct RetDecoding<Buffer<dtype, rank>> {
  ZKX_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<Buffer<dtype, rank>>> Decode(
      ZKX_FFI_RetType type, void* ret, DiagnosticEngine& diagnostic) {
    if (ZKX_FFI_PREDICT_FALSE(type != ZKX_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << ZKX_FFI_RetType_BUFFER << " but got " << type;
    }

    return internal::DecodeBuffer<dtype, rank>(
        reinterpret_cast<ZKX_FFI_Buffer*>(ret), diagnostic);
  }
};

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing a variable number of results.
//===----------------------------------------------------------------------===//

class RemainingRets : public internal::RemainingRetsBase {
 public:
  using internal::RemainingRetsBase::RemainingRetsBase;

  template <typename T>
  ErrorOr<Result<T>> get(size_t index) const {
    size_t idx = offset() + index;
    if (ZKX_FFI_PREDICT_FALSE(idx >= rets()->size)) {
      return Unexpected(
          Error(ErrorCode::kInvalidArgument, "Index out of range"));
    }

    DiagnosticEngine diagnostic;
    std::optional<Result<T>> value = RetDecoding<T>::Decode(
        rets()->types[idx], rets()->rets[idx], diagnostic);
    if (ZKX_FFI_PREDICT_FALSE(!value.has_value())) {
      return Unexpected(Error::Internal(diagnostic.Result()));
    }

    return *value;
  }
};

template <>
struct internal::Decode<internal::RemainingRetsTag> {
  static std::optional<RemainingRets> call(DecodingOffsets& offsets,
                                           DecodingContext& ctx,
                                           DiagnosticEngine& diagnostic) {
    return RemainingRets(&ctx.call_frame->rets, offsets.rets);
  }
};

//===----------------------------------------------------------------------===//
// Attributes decoding
//===----------------------------------------------------------------------===//

#define ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(T, TYPE)                       \
  template <>                                                               \
  struct AttrDecoding<Span<const T>> {                                      \
    using Type = Span<const T>;                                             \
    static std::optional<Type> Decode(ZKX_FFI_AttrType type, void* attr,    \
                                      DiagnosticEngine& diagnostic) {       \
      if (ZKX_FFI_PREDICT_FALSE(type != ZKX_FFI_AttrType_ARRAY)) {          \
        return diagnostic.Emit("Wrong attribute type: expected ")           \
               << ZKX_FFI_AttrType_ARRAY << " but got " << type;            \
      }                                                                     \
                                                                            \
      auto* array = reinterpret_cast<ZKX_FFI_Array*>(attr);                 \
      if (ZKX_FFI_PREDICT_FALSE(array->dtype != TYPE)) {                    \
        return diagnostic.Emit("Wrong array data type: expected ")          \
               << TYPE << " but got " << array->dtype;                      \
      }                                                                     \
                                                                            \
      return Span<const T>(reinterpret_cast<T*>(array->data), array->size); \
    }                                                                       \
  }

ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(int8_t, ZKX_FFI_DataType_S8);
ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(int16_t, ZKX_FFI_DataType_S16);
ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(int32_t, ZKX_FFI_DataType_S32);
ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(int64_t, ZKX_FFI_DataType_S64);
ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(uint8_t, ZKX_FFI_DataType_U8);
ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(uint16_t, ZKX_FFI_DataType_U16);
ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(uint32_t, ZKX_FFI_DataType_U32);
ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING(uint64_t, ZKX_FFI_DataType_U64);

#undef ZKX_FFI_REGISTER_ARRAY_ATTR_DECODING

template <>
struct AttrDecoding<std::string_view> {
  using Type = std::string_view;
  static std::optional<std::string_view> Decode(ZKX_FFI_AttrType type,
                                                void* attr,
                                                DiagnosticEngine& diagnostic) {
    if (ZKX_FFI_PREDICT_FALSE(type != ZKX_FFI_AttrType_STRING)) {
      return diagnostic.Emit("Wrong attribute type: expected ")
             << ZKX_FFI_AttrType_STRING << " but got " << type;
    }

    auto* span = reinterpret_cast<ZKX_FFI_ByteSpan*>(attr);
    return std::string_view(span->ptr, span->len);
  }
};

// A type tag to mark i64 attributes as pointers to `T`.
template <typename T>
struct Pointer {};

template <typename T>
struct AttrDecoding<Pointer<T>> {
  using Type = T*;

  static std::optional<Type> Decode(ZKX_FFI_AttrType type, void* attr,
                                    DiagnosticEngine& diagnostic) {
    auto* scalar = reinterpret_cast<ZKX_FFI_Scalar*>(attr);
    if (ZKX_FFI_PREDICT_FALSE(type != ZKX_FFI_AttrType_SCALAR ||
                              scalar->dtype != ZKX_FFI_DataType_S64)) {
      return diagnostic.Emit("Wrong attribute type: ")
             << "expected i64 scalar for passing pointer but got " << type;
    }

    static_assert(sizeof(uintptr_t) == sizeof(int64_t));
    uintptr_t ptr = *reinterpret_cast<uintptr_t*>(scalar->value);
    return reinterpret_cast<Type>(ptr);
  }
};

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing dictionary attributes.
//===----------------------------------------------------------------------===//

class Dictionary : public internal::DictionaryBase {
 public:
  using internal::DictionaryBase::DictionaryBase;

  template <typename T>
  ErrorOr<T> get(std::string_view name) const {
    DiagnosticEngine diagnostic;
    std::optional<T> value = internal::DictionaryBase::get<T>(name, diagnostic);
    if (!value.has_value()) {
      return Unexpected(Error::Internal(diagnostic.Result()));
    }
    return *value;
  }
};

// Decode `AttrsTag` (all attributes) into a `Dictionary`.
template <>
struct internal::Decode<internal::AttrsTag<Dictionary>> {
  static std::optional<Dictionary> call(DecodingOffsets& offsets,
                                        DecodingContext& ctx,
                                        DiagnosticEngine& diagnostic) {
    return Dictionary(&ctx.call_frame->attrs);
  }
};

// Decode individual attribute into `Dictionary` type.
template <>
struct AttrDecoding<Dictionary> {
  using Type = Dictionary;
  static std::optional<Dictionary> Decode(ZKX_FFI_AttrType type, void* attr,
                                          DiagnosticEngine& diagnostic) {
    if (ZKX_FFI_PREDICT_FALSE(type != ZKX_FFI_AttrType_DICTIONARY)) {
      return diagnostic.Emit("Wrong attribute type: expected ")
             << ZKX_FFI_AttrType_DICTIONARY << " but got " << type;
    }
    return Dictionary(reinterpret_cast<ZKX_FFI_Attrs*>(attr));
  }
};

//===----------------------------------------------------------------------===//
// Error helpers
//===----------------------------------------------------------------------===//

namespace internal {

inline ZKX_FFI_Error* CreateError(const ZKX_FFI_Api* api, const Error& error) {
  ZKX_FFI_Error_Create_Args args;
  args.struct_size = ZKX_FFI_Error_Create_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.errc = static_cast<ZKX_FFI_Error_Code>(*error.errc());
  args.message = error.message().c_str();
  return api->ZKX_FFI_Error_Create(&args);
}

inline void DestroyError(const ZKX_FFI_Api* api, ZKX_FFI_Error* error) {
  ZKX_FFI_Error_Destroy_Args args;
  args.struct_size = ZKX_FFI_Error_Destroy_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.error = error;
  api->ZKX_FFI_Error_Destroy(&args);
}

inline const char* GetErrorMessage(const ZKX_FFI_Api* api,
                                   ZKX_FFI_Error* error) {
  ZKX_FFI_Error_GetMessage_Args args;
  args.struct_size = ZKX_FFI_Error_GetMessage_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.error = error;
  api->ZKX_FFI_Error_GetMessage(&args);
  return args.message;
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Result encoding
//===----------------------------------------------------------------------===//

// Encodes `Error` as an FFI error.
template <ExecutionStage stage>
struct ResultEncoding<stage, Error> {
  static ZKX_FFI_Error* Encode(const ZKX_FFI_Api* api,
                               ZKX_FFI_ExecutionContext* ctx, Error error) {
    if (ZKX_FFI_PREDICT_TRUE(error.success())) {
      return nullptr;
    }

    return internal::CreateError(api, error);
  }
};

// Encodes `ErrorOr<std::unique_ptr<T>>` as an FFI state.
template <typename T>
struct ResultEncoding<ExecutionStage::kInstantiate,
                      ErrorOr<std::unique_ptr<T>>> {
  static_assert(std::is_same_v<decltype(T::id), TypeId>,
                "State type must have a static `TypeId id` field");

  static ZKX_FFI_Error* Encode(const ZKX_FFI_Api* api,
                               ZKX_FFI_ExecutionContext* ctx,
                               ErrorOr<std::unique_ptr<T>> state) {
    if (ZKX_FFI_PREDICT_TRUE(state.has_value())) {
      ZKX_FFI_State_Set_Args args;
      args.struct_size = ZKX_FFI_State_Set_Args_STRUCT_SIZE;
      args.extension_start = nullptr;
      args.ctx = ctx;
      args.type_id = &T::id;
      args.state = state.value().release();
      args.deleter = +[](void* state) { delete reinterpret_cast<T*>(state); };
      return api->ZKX_FFI_State_Set(&args);
    }

    return internal::CreateError(api, state.error());
  }
};

// Encodes `Future` as an asynchronous FFI result.
template <ExecutionStage stage>
struct ResultEncoding<stage, Future> {
  static std::variant<ZKX_FFI_Error*, ZKX_FFI_Future*> Encode(
      const ZKX_FFI_Api* api, ZKX_FFI_ExecutionContext* ctx, Future future) {
    // Create ZKX_FFI_Future object that will signal completion to the runtime.
    ZKX_FFI_Future_Create_Args args;
    args.struct_size = ZKX_FFI_Future_Create_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.future = nullptr;

    if (auto* err = api->ZKX_FFI_Future_Create(&args)) {
      return err;
    }

    assert(args.future != nullptr && "ZKX_FFI_Future_Create failed");

    future.OnReady([api, f = args.future](const std::optional<Error>& error) {
      // When the OnReady callback is invoked, we no longer have access to the
      // diagnostics, and can't signal an error to the runtime. We chose to
      // abort execution, because otherwise it will lead to a deadlock. However
      // we should never get to this point, because execution must be aborted on
      // a synchronous path when checking ZKX FFI version compatibility.
      auto abort_on_error = [api](ZKX_FFI_Error* err) {
        if (ZKX_FFI_PREDICT_TRUE(err == nullptr)) {
          return;
        }

        std::cerr << "Failed to signal ZKX_FFI_Future completion: "
                  << internal::GetErrorMessage(api, err) << std::endl;
        internal::DestroyError(api, err);
        std::abort();
      };

      if (ZKX_FFI_PREDICT_FALSE(error.has_value())) {
        ZKX_FFI_Future_SetError_Args args;
        args.struct_size = ZKX_FFI_Future_SetError_Args_STRUCT_SIZE;
        args.extension_start = nullptr;
        args.future = f;
        args.error = internal::CreateError(api, *error);

        abort_on_error(api->ZKX_FFI_Future_SetError(&args));

      } else {
        ZKX_FFI_Future_SetAvailable_Args args;
        args.struct_size = ZKX_FFI_Future_SetAvailable_Args_STRUCT_SIZE;
        args.extension_start = nullptr;
        args.future = f;

        abort_on_error(api->ZKX_FFI_Future_SetAvailable(&args));
      }
    });

    return args.future;
  }
};

//===----------------------------------------------------------------------===//
// PlatformStream
//===----------------------------------------------------------------------===//

template <typename T>
struct PlatformStream {};

// Context decoding for platform stream.
//
// Example: Ffi::Bind().Ctx<PlatformStream<cudaStream_t>()
//                     .To([](cudaStream_t stream) { ... });
template <typename T>
struct CtxDecoding<PlatformStream<T>> {
  using Type = T;

  static_assert(std::is_pointer_v<T>, "stream type must be a pointer");

  static std::optional<Type> Decode(const ZKX_FFI_Api* api,
                                    ZKX_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    ZKX_FFI_Stream_Get_Args args;
    args.struct_size = ZKX_FFI_Stream_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx;
    args.stream = nullptr;

    if (ZKX_FFI_Error* error = api->ZKX_FFI_Stream_Get(&args)) {
      diagnostic.Emit("Failed to get platform stream: ")
          << internal::GetErrorMessage(api, error);
      internal::DestroyError(api, error);
      return std::nullopt;
    }

    return reinterpret_cast<T>(args.stream);
  }
};

//===----------------------------------------------------------------------===//
// ScratchAllocator
//===----------------------------------------------------------------------===//

// Interface for "scratch" allocator for device memory, which deallocates all
// buffers it has allocated at destruction.
//
// WARNING: It is illegal to keep scratch allocator alive after returning from
// the FFI handler as it relies on execution context whose lifetime is bound to
// the particular call to FFI handler.
class ScratchAllocator {
 public:
  ~ScratchAllocator();

  ScratchAllocator(ScratchAllocator&&) = default;
  ScratchAllocator& operator=(ScratchAllocator&&) = delete;

  std::optional<void*> Allocate(size_t size, size_t alignment = 1);

 private:
  friend struct CtxDecoding<ScratchAllocator>;

  ScratchAllocator(const ZKX_FFI_Api* api, ZKX_FFI_ExecutionContext* ctx,
                   DiagnosticEngine& diagnostic);

  struct Allocation {
    size_t size;
    void* data;
  };

  const ZKX_FFI_Api* api_;
  ZKX_FFI_ExecutionContext* ctx_;

  DiagnosticEngine& diagnostic_;
  std::vector<Allocation> allocations_;
};

// Context decoding for scratch allocator.
//
// Example: Ffi::Bind().Ctx<ScratchAllocator>()
//                     .To([](ScratchAllocator scratch) { ... });
template <>
struct CtxDecoding<ScratchAllocator> {
  using Type = ScratchAllocator;

  static std::optional<Type> Decode(const ZKX_FFI_Api* api,
                                    ZKX_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return ScratchAllocator(api, ctx, diagnostic);
  }
};

inline std::optional<void*> ScratchAllocator::Allocate(size_t size,
                                                       size_t alignment) {
  ZKX_FFI_DeviceMemory_Allocate_Args args;
  args.struct_size = ZKX_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.ctx = ctx_;
  args.size = size;
  args.alignment = alignment;
  args.data = nullptr;
  if (ZKX_FFI_Error* error = api_->ZKX_FFI_DeviceMemory_Allocate(&args)) {
    diagnostic_.Emit("Failed to allocate scratch memory: ")
        << internal::GetErrorMessage(api_, error);
    internal::DestroyError(api_, error);
    return std::nullopt;
  }
  allocations_.push_back({size, args.data});
  return args.data;
}

inline ScratchAllocator::ScratchAllocator(const ZKX_FFI_Api* api,
                                          ZKX_FFI_ExecutionContext* ctx,
                                          DiagnosticEngine& diagnostic)
    : api_(api), ctx_(ctx), diagnostic_(diagnostic) {}

inline ScratchAllocator::~ScratchAllocator() {
  for (Allocation& alloc : allocations_) {
    ZKX_FFI_DeviceMemory_Free_Args args;
    args.struct_size = ZKX_FFI_DeviceMemory_Free_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx_;
    args.size = alloc.size;
    args.data = alloc.data;
    if (ZKX_FFI_Error* error = api_->ZKX_FFI_DeviceMemory_Free(&args)) {
      diagnostic_.Emit("Failed to free scratch memory: ")
          << internal::GetErrorMessage(api_, error);
      internal::DestroyError(api_, error);
    }
  }
}

//===----------------------------------------------------------------------===//
// ThreadPool
//===----------------------------------------------------------------------===//

class ThreadPool {
 public:
  template <typename F>
  void Schedule(F&& f) {
    ZKX_FFI_Task* task = +[](void* data) {
      auto* f = reinterpret_cast<F*>(data);
      (*f)();
      delete f;
    };

    F* data = new F(std::forward<F>(f));

    ZKX_FFI_ThreadPool_Schedule_Args args;
    args.struct_size = ZKX_FFI_ThreadPool_Schedule_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx_;
    args.task = task;
    args.data = data;

    if (ZKX_FFI_Error* error = api_->ZKX_FFI_ThreadPool_Schedule(&args)) {
      diagnostic_.Emit("Failed to schedule task on a thread pool: ")
          << internal::GetErrorMessage(api_, error);
      internal::DestroyError(api_, error);

      // If thread pool is not available, we execute the task in the caller
      // thread. We choose not to return error from `Schedule` for consistency
      // with Eigen thread pool implementation, and because it would make
      // recursive work scheduling more difficult.
      task(data);
    }
  }

  int64_t num_threads() const {
    int64_t num_threads = 0;

    ZKX_FFI_ThreadPool_NumThreads_Args args;
    args.struct_size = ZKX_FFI_ThreadPool_NumThreads_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx_;
    args.num_threads = &num_threads;

    // Silently ignore errors if we can't get the number of threads.
    if (ZKX_FFI_Error* error = api_->ZKX_FFI_ThreadPool_NumThreads(&args)) {
      internal::DestroyError(api_, error);
    }

    return num_threads;
  }

 private:
  friend struct CtxDecoding<ThreadPool>;

  ThreadPool(const ZKX_FFI_Api* api, ZKX_FFI_ExecutionContext* ctx,
             DiagnosticEngine& diagnostic);

  const ZKX_FFI_Api* api_;
  ZKX_FFI_ExecutionContext* ctx_;
  DiagnosticEngine& diagnostic_;
};

// Context decoding for thread pool.
//
// Example: Ffi::Bind().Ctx<ThreadPool>()
//                     .To([](ThreadPool thread_pool) { ... });
template <>
struct CtxDecoding<ThreadPool> {
  using Type = ThreadPool;

  static std::optional<Type> Decode(const ZKX_FFI_Api* api,
                                    ZKX_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return ThreadPool(api, ctx, diagnostic);
  }
};

inline ThreadPool::ThreadPool(const ZKX_FFI_Api* api,
                              ZKX_FFI_ExecutionContext* ctx,
                              DiagnosticEngine& diagnostic)
    : api_(api), ctx_(ctx), diagnostic_(diagnostic) {}

//===----------------------------------------------------------------------===//
// Type Registration
//===----------------------------------------------------------------------===//

#define ZKX_FFI_REGISTER_TYPE(API, NAME, TYPE_ID) \
  ZKX_FFI_REGISTER_TYPE_(API, NAME, TYPE_ID, __COUNTER__)
#define ZKX_FFI_REGISTER_TYPE_(API, NAME, TYPE_ID, N) \
  ZKX_FFI_REGISTER_TYPE__(API, NAME, TYPE_ID, N)
#define ZKX_FFI_REGISTER_TYPE__(API, NAME, TYPE_ID, N) \
  ZKX_FFI_ATTRIBUTE_UNUSED static const ZKX_FFI_Error* \
      ZKX_ffi_type_##N##_registered_ =                 \
          [] { return ::zkx::ffi::Ffi::RegisterTypeId(API, NAME, TYPE_ID); }()

//===----------------------------------------------------------------------===//
// UserData
//===----------------------------------------------------------------------===//

// A type tag for automatic user data decoding passed via the execution
// context.
template <typename T>
struct UserData {};

// Context decoding for user data of type `T`.
//
// Example: Ffi::Bind().Ctx<UserData<MyData>>()
//                     .To([](MyData* data) { ... });
template <typename T>
struct CtxDecoding<UserData<T>> {
  using Type = T*;

  static_assert(std::is_same_v<decltype(T::id), TypeId>,
                "UserData type must have a static `TypeId id` field");

  static std::optional<Type> Decode(const ZKX_FFI_Api* api,
                                    ZKX_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    ZKX_FFI_ExecutionContext_Get_Args args;
    args.struct_size = ZKX_FFI_ExecutionContext_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx;
    args.type_id = &T::id;
    args.data = nullptr;

    assert(args.type_id->type_id > 0 && "type must be registered with ZKX FFI");

    if (ZKX_FFI_Error* err = api->ZKX_FFI_ExecutionContext_Get(&args); err) {
      diagnostic.Emit("Failed to get user data from execution context: ")
          << internal::GetErrorMessage(api, err);
      internal::DestroyError(api, err);
      return std::nullopt;
    }

    return static_cast<Type>(args.data);
  }
};

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

// A type tag for automatic state decoding passed via the execution
// context.
template <typename T>
struct State {};

// Context decoding for state of type `T`.
//
// Example: Ffi::Bind().Ctx<State<MyState>>()
//                     .To([](MyState* state) { ... });
template <typename T>
struct CtxDecoding<State<T>> {
  using Type = T*;

  static_assert(std::is_same_v<decltype(T::id), TypeId>,
                "State type must have a static `TypeId id` field");

  static std::optional<Type> Decode(const ZKX_FFI_Api* api,
                                    ZKX_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    ZKX_FFI_State_Get_Args args;
    args.struct_size = ZKX_FFI_State_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx;
    args.type_id = &T::id;
    args.state = nullptr;

    assert(args.type_id->type_id > 0 && "type must be registered with ZKX FFI");

    if (ZKX_FFI_Error* err = api->ZKX_FFI_State_Get(&args); err) {
      diagnostic.Emit("Failed to get state from execution context: ")
          << internal::GetErrorMessage(api, err);
      internal::DestroyError(api, err);
      return std::nullopt;
    }

    return static_cast<Type>(args.state);
  }
};

//===----------------------------------------------------------------------===//
// RunId
//===----------------------------------------------------------------------===//

struct RunId {
  int64_t run_id;
};

// Context decoding for RunId (unique identifier of a logical execution).
//
// Example: Ffi::Bind().Ctx<RunId>()
//                     .To([](RunId run_id) { ... });
template <>
struct CtxDecoding<RunId> {
  using Type = RunId;

  static std::optional<Type> Decode(const ZKX_FFI_Api* api,
                                    ZKX_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    ZKX_FFI_RunId_Get_Args args;
    args.struct_size = ZKX_FFI_ExecutionContext_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx;
    args.run_id = 0;

    if (ZKX_FFI_Error* err = api->ZKX_FFI_RunId_Get(&args); err) {
      diagnostic.Emit("Failed to get run id from execution context: ")
          << internal::GetErrorMessage(api, err);
      internal::DestroyError(api, err);
      return std::nullopt;
    }

    return RunId{args.run_id};
  }
};

}  // namespace zkx::ffi

#endif  // ZKX_FFI_API_FFI_H_
