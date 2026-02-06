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

#include "zkx/ffi/call_frame.h"

#include <functional>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"

#include "xla/tsl/platform/errors.h"
#include "zk_dtypes/include/all_types.h"
#include "zkx/ffi/api/api.h"
#include "zkx/ffi/api/c_api_internal.h"  // IWYU pragma: keep

namespace zkx::ffi {

//===----------------------------------------------------------------------===//
// CallFrameBuilder
//===----------------------------------------------------------------------===//

struct CallFrameBuilder::Buffer {
  se::DeviceMemoryBase memory;
  PrimitiveType type;
  absl::InlinedVector<int64_t, 4> dims;
};

CallFrameBuilder::AttributesMap CallFrameBuilder::AttributesBuilder::Build() {
  return std::move(attrs_);
}

CallFrameBuilder::AttributesBuilder::AttributesBuilder() = default;
CallFrameBuilder::AttributesBuilder::~AttributesBuilder() = default;

void CallFrameBuilder::AttributesBuilder::Insert(std::string name,
                                                 Attribute attr) {
  attrs_.try_emplace(std::move(name), std::move(attr));
}

void CallFrameBuilder::AttributesBuilder::Insert(std::string name,
                                                 AttributesMap attrs) {
  attrs_.try_emplace(std::move(name),
                     Dictionary{std::make_shared<AttributesMap>(attrs)});
}

void CallFrameBuilder::AttributesBuilder::Append(AttributesMap attrs) {
  for (auto& [name, attr] : attrs) Insert(name, std::move(attr));
}

CallFrameBuilder::CallFrameBuilder(size_t num_args, size_t num_rets) {
  args_.reserve(num_args);
  rets_.reserve(num_rets);
}

CallFrameBuilder::~CallFrameBuilder() = default;

void CallFrameBuilder::AddBufferArg(se::DeviceMemoryBase memory,
                                    PrimitiveType type,
                                    absl::Span<const int64_t> dims) {
  DCHECK(args_.capacity() > args_.size())
      << "CallFrame builder `num_args` argument was too small";
  args_.push_back(Buffer{memory, type, {dims.begin(), dims.end()}});
}

void CallFrameBuilder::AddTokenArg() {
  DCHECK(args_.capacity() > args_.size())
      << "CallFrame builder `num_args` argument was too small";
  args_.push_back(Buffer{se::DeviceMemoryBase(), PrimitiveType::TOKEN, {}});
}

void CallFrameBuilder::AddBufferRet(se::DeviceMemoryBase memory,
                                    PrimitiveType type,
                                    absl::Span<const int64_t> dims) {
  DCHECK(rets_.capacity() > rets_.size())
      << "CallFrame builder `num_rets` argument was too small";
  rets_.push_back(Buffer{memory, type, {dims.begin(), dims.end()}});
}

void CallFrameBuilder::AddTokenRet() {
  DCHECK(rets_.capacity() > rets_.size())
      << "CallFrame builder `num_rets` argument was too small";
  rets_.push_back(Buffer{se::DeviceMemoryBase(), PrimitiveType::TOKEN, {}});
}

void CallFrameBuilder::AddAttributes(AttributesMap attrs) {
  if (ABSL_PREDICT_TRUE(attrs_.empty())) {
    attrs_ = std::move(attrs);
    return;
  }

  for (auto& [name, attr] : attrs) {
    attrs_.try_emplace(std::move(name), std::move(attr));
  }
}

CallFrame CallFrameBuilder::Build() {
  return CallFrame(CallFrame::CreateArgs(args_), CallFrame::CreateRets(rets_),
                   CallFrame::CreateAttrs(attrs_));
}

CallFrameBuilder::CallFrameBuilder(CallFrameBuilder&&) = default;
CallFrameBuilder& CallFrameBuilder::operator=(CallFrameBuilder&&) = default;

// ------------------------    !!! !!! !!!     ------------------------------ //

// WARNING: In many structs defined below we use a pattern where we declare
// a storage (e.g. an `std::string` member) and an ZKX FFI reference type
// pointing into that storage in the same struct (ZKX_FFI_ByteSpan). Extra care
// should be taken of keeping reference type up to date, e.g. if a parent
// struct put into an `std::vector` container, every time vector will reallocate
// storage all reference types will become invalid.

// We intentionally do not use smart pointers that would guarantee pointer
// stability for storage, as we are trying to minimize the number of heap
// allocations required for building a call frame.

// This is a low level internal implementation detail that should not leak via
// public header files, and can be changed at any time in the future.

//----------------------------------------------------------------------------//
// Arguments storage + reference types
//----------------------------------------------------------------------------//

struct CallFrame::Buffer {
  absl::InlinedVector<int64_t, 4> dims;  // ZKX_FFI_Buffer::dims

  ZKX_FFI_Buffer buffer = {ZKX_FFI_Buffer_STRUCT_SIZE, nullptr};
};

struct CallFrame::Dictionary {
  std::unique_ptr<Attributes> attrs;
};

struct CallFrame::Array {
  CallFrameBuilder::Array value;  // ZKX_FFI_Array::data

  ZKX_FFI_Array array = {};
};

struct CallFrame::Scalar {
  CallFrameBuilder::Scalar value;  // ZKX_FFI_Scalar::value

  ZKX_FFI_Scalar scalar = {};
};

struct CallFrame::String {
  std::string value;  // ZKX_FFI_ByteSpan::ptr

  ZKX_FFI_ByteSpan span = {};
};

struct CallFrame::NamedAttribute {
  String name;
  Attribute value;
};

struct CallFrame::Arguments {
  std::vector<Buffer> arguments;

  std::vector<ZKX_FFI_ArgType> types;  // ZKX_FFI_Args::types
  std::vector<void*> args;             // ZKX_FFI_Args::args

  ZKX_FFI_Args ffi_args = {ZKX_FFI_Args_STRUCT_SIZE, nullptr};
};

struct CallFrame::Results {
  std::vector<Buffer> results;

  std::vector<ZKX_FFI_RetType> types;  // ZKX_FFI_Rets::types
  std::vector<void*> rets;             // ZKX_FFI_Rets::rets

  ZKX_FFI_Rets ffi_rets = {ZKX_FFI_Rets_STRUCT_SIZE, nullptr};
};

struct CallFrame::Attributes {
  std::vector<NamedAttribute> attributes;

  std::vector<ZKX_FFI_ByteSpan*> names;  // ZKX_FFI_Attributes::names
  std::vector<ZKX_FFI_AttrType> types;   // ZKX_FFI_Attributes::types
  std::vector<void*> attrs;              // ZKX_FFI_Attributes::attrs

  ZKX_FFI_Attrs ffi_attrs = {ZKX_FFI_Attrs_STRUCT_SIZE, nullptr};
};

//===----------------------------------------------------------------------===//
// CallFrame
//===----------------------------------------------------------------------===//

CallFrame::CallFrame(CallFrame&&) = default;
CallFrame& CallFrame::operator=(CallFrame&&) = default;
CallFrame::~CallFrame() = default;

CallFrame::CallFrame(std::unique_ptr<Arguments> arguments,
                     std::unique_ptr<Results> results,
                     std::shared_ptr<Attributes> attributes)
    : arguments_(std::move(arguments)),
      results_(std::move(results)),
      attributes_(std::move(attributes)) {}

ZKX_FFI_CallFrame CallFrame::Build(const ZKX_FFI_Api* api,
                                   ZKX_FFI_ExecutionContext* ctx,
                                   ZKX_FFI_ExecutionStage stage) {
  ZKX_FFI_CallFrame call_frame = {ZKX_FFI_CallFrame_STRUCT_SIZE, nullptr};
  call_frame.api = api;
  call_frame.ctx = ctx;
  call_frame.stage = stage;
  call_frame.args = arguments_->ffi_args;
  call_frame.rets = results_->ffi_rets;
  call_frame.attrs = attributes_->ffi_attrs;
  return call_frame;
}

// We rely on casting to and from underlying integral type to convert from
// PrimitiveType to ZKX FFI DataType, and for safety convert all unknown types
// to invalid type, otherwise we can accidentally cause UB.
static ZKX_FFI_DataType ToDataType(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case PrimitiveType::PRIMITIVE_TYPE_INVALID:
      return ZKX_FFI_DataType_INVALID;
    case PrimitiveType::PRED:
      return ZKX_FFI_DataType_PRED;
    case PrimitiveType::S8:
      return ZKX_FFI_DataType_S8;
    case PrimitiveType::S16:
      return ZKX_FFI_DataType_S16;
    case PrimitiveType::S32:
      return ZKX_FFI_DataType_S32;
    case PrimitiveType::S64:
      return ZKX_FFI_DataType_S64;
    case PrimitiveType::U8:
      return ZKX_FFI_DataType_U8;
    case PrimitiveType::U16:
      return ZKX_FFI_DataType_U16;
    case PrimitiveType::U32:
      return ZKX_FFI_DataType_U32;
    case PrimitiveType::U64:
      return ZKX_FFI_DataType_U64;
    case PrimitiveType::TOKEN:
      return ZKX_FFI_DataType_TOKEN;
#define ZK_DTYPES_CASE(unused, unused2, enum, unused3) \
  case PrimitiveType::enum:                            \
    return ZKX_FFI_DataType_##enum;
      ZK_DTYPES_PUBLIC_TYPE_LIST(ZK_DTYPES_CASE)
#undef ZK_DTYPES_CASE
    default:
      DCHECK(false) << "Unsupported primitive type "
                    << PrimitiveType_Name(primitive_type);
      return ZKX_FFI_DataType_INVALID;
  }
}

CallFrame::Buffer CallFrame::ConvertBuffer(
    const CallFrameBuilder::Buffer& buffer) {
  Buffer result;
  result.dims = buffer.dims;
  result.buffer.data = const_cast<void*>(buffer.memory.opaque());
  result.buffer.dtype = ToDataType(buffer.type);
  result.buffer.rank = result.dims.size();
  return result;
}

//===----------------------------------------------------------------------===//
// Call frame arguments
//===----------------------------------------------------------------------===//

std::unique_ptr<CallFrame::Arguments> CallFrame::CreateArgs(
    absl::Span<const CallFrameBuilder::Buffer> bargs) {
  size_t num_args = bargs.size();

  auto args = std::make_unique<Arguments>();
  args->types.resize(num_args, ZKX_FFI_ArgType_BUFFER);
  args->args.resize(num_args, nullptr);  // fixed up below

  // Convert call frame builder arguments to call frame arguments.
  args->arguments.reserve(num_args);
  for (const CallFrameBuilder::Buffer& barg : bargs) {
    args->arguments.push_back(ConvertBuffer(barg));
  }

  // Fix up ZKX FFI structs with pointers to valid arguments storage.
  return FixUpArgs(std::move(args));
}

std::unique_ptr<CallFrame::Arguments> CallFrame::CopyArgs(
    const Arguments& args) {
  auto upd_args = std::make_unique<Arguments>();

  upd_args->arguments = args.arguments;
  upd_args->types = args.types;
  upd_args->args.resize(args.args.size(), nullptr);  // fixed up below

  // Fix up ZKX FFI structs with pointers to valid arguments storage.
  return FixUpArgs(std::move(upd_args));
}

std::unique_ptr<CallFrame::Arguments> CallFrame::FixUpArgs(
    std::unique_ptr<Arguments> args) {
  size_t num_args = args->arguments.size();
  DCHECK_EQ(num_args, args->types.size());
  DCHECK_EQ(num_args, args->args.size());

  // Fix up pointers in ZKX FFI structs and initialize vectors required for
  // building ZKX_FFI_Args.
  for (size_t i = 0; i < num_args; ++i) {
    args->arguments[i].buffer.dims = args->arguments[i].dims.data();
    args->args[i] = &args->arguments[i].buffer;
  }

  // Finally initialize the ZKX FFI struct. At this point all storage is
  // allocated and it's safe to grab a pointer to it.
  args->ffi_args.size = num_args;
  args->ffi_args.types = args->types.data();
  args->ffi_args.args = args->args.data();

  return args;
}

//===----------------------------------------------------------------------===//
// Call frame results
//===----------------------------------------------------------------------===//

std::unique_ptr<CallFrame::Results> CallFrame::CreateRets(
    absl::Span<const CallFrameBuilder::Buffer> brets) {
  auto rets = std::make_unique<Results>();

  size_t num_rets = brets.size();
  rets->types.resize(num_rets, ZKX_FFI_RetType_BUFFER);
  rets->rets.resize(num_rets, nullptr);  // fixed up below

  // Convert call frame builder result to call frame results.
  rets->results.reserve(num_rets);
  for (const CallFrameBuilder::Buffer& bret : brets) {
    rets->results.push_back(ConvertBuffer(bret));
  }

  // Fix up ZKX FFI structs with pointers to valid results storage.
  return FixUpRets(std::move(rets));
}

std::unique_ptr<CallFrame::Results> CallFrame::CopyRets(const Results& rets) {
  auto upd_rets = std::make_unique<Results>();

  upd_rets->results = rets.results;
  upd_rets->types = rets.types;
  upd_rets->rets.resize(rets.rets.size(), nullptr);  // fixed up below

  // Fix up ZKX FFI structs with pointers to valid results storage.
  return FixUpRets(std::move(upd_rets));
}

std::unique_ptr<CallFrame::Results> CallFrame::FixUpRets(
    std::unique_ptr<Results> rets) {
  size_t num_rets = rets->results.size();
  DCHECK_EQ(num_rets, rets->types.size());
  DCHECK_EQ(num_rets, rets->rets.size());

  // Fix up pointers in ZKX FFI structs and initialize vectors required for
  // building ZKX_FFI_Args.
  for (size_t i = 0; i < num_rets; ++i) {
    rets->results[i].buffer.dims = rets->results[i].dims.data();
    rets->rets[i] = &rets->results[i].buffer;
  }

  // Finally initialize the ZKX FFI struct. At this point all storage is
  // allocated and it's safe to grab a pointer to it.
  rets->ffi_rets.size = num_rets;
  rets->ffi_rets.types = rets->types.data();
  rets->ffi_rets.rets = rets->rets.data();

  return rets;
}

//===----------------------------------------------------------------------===//
// Call frame attributes
//===----------------------------------------------------------------------===//

// An std::visit overload set for converting CallFrameBuilder::Attribute to
// CallFrame::Attribute.
struct CallFrame::ConvertAttribute {
  CallFrame::Attribute operator()(const CallFrameBuilder::Array& array) {
    return CallFrame::Array{array};
  }

  CallFrame::Attribute operator()(const CallFrameBuilder::Scalar& scalar) {
    return CallFrame::Scalar{scalar};
  }

  CallFrame::Attribute operator()(const std::string& str) {
    return CallFrame::String{str};
  }

  CallFrame::Attribute operator()(const CallFrameBuilder::Dictionary& dict) {
    return CallFrame::Dictionary{CreateAttrs(*dict.attrs)};
  }
};

// An std::visit overload set to fix up CallFrame::Attribute storage and
// initialize ZKX FFI structs with valid pointers into storage objects.
struct CallFrame::FixUpAttribute {
  void operator()(CallFrame::Array& array) {
    auto visitor = [&](auto& value) {
      using T = typename std::remove_reference_t<decltype(value)>::value_type;
      array.array.dtype = internal::NativeTypeToCApiDataType<T>();
      array.array.size = value.size();
      array.array.data = value.data();
    };
    std::visit(visitor, array.value);
  }

  void operator()(CallFrame::Scalar& scalar) {
    auto visitor = [&](auto& value) {
      using T = std::remove_reference_t<decltype(value)>;
      scalar.scalar.dtype = internal::NativeTypeToCApiDataType<T>();
      scalar.scalar.value = &value;
    };
    std::visit(visitor, scalar.value);
  }

  void operator()(CallFrame::String& str) {
    str.span.ptr = str.value.data();
    str.span.len = str.value.size();
  }

  void operator()(CallFrame::Dictionary&) {}
};

// An std::visit overload set to get CallFrame::Attribute ZKX FFI type.
struct CallFrame::AttributeType {
  ZKX_FFI_AttrType operator()(CallFrame::Array&) {
    return ZKX_FFI_AttrType_ARRAY;
  }

  ZKX_FFI_AttrType operator()(CallFrame::Scalar&) {
    return ZKX_FFI_AttrType_SCALAR;
  }

  ZKX_FFI_AttrType operator()(CallFrame::String&) {
    return ZKX_FFI_AttrType_STRING;
  }

  ZKX_FFI_AttrType operator()(CallFrame::Dictionary&) {
    return ZKX_FFI_AttrType_DICTIONARY;
  }
};

// An std::visit overload set to get CallFrame::Attribute storage pointer.
struct CallFrame::AttributeStorage {
  template <typename T>
  void* operator()(T& value) {
    return &value;
  }

  void* operator()(CallFrame::Array& array) { return &array.array; }

  void* operator()(CallFrame::Scalar& scalar) { return &scalar.scalar; }

  void* operator()(CallFrame::String& str) { return &str.span; }

  void* operator()(CallFrame::Dictionary& dict) {
    return &dict.attrs->ffi_attrs;
  }
};

std::unique_ptr<CallFrame::Attributes> CallFrame::CreateAttrs(
    const CallFrameBuilder::AttributesMap& battrs) {
  auto attrs = std::make_unique<Attributes>();

  // Convert call frame builder attributes to a collection of named attributes.
  attrs->attributes.reserve(battrs.size());
  for (auto& [name, battr] : battrs) {
    NamedAttribute attr = {String{name}, std::visit(ConvertAttribute(), battr)};
    attrs->attributes.push_back(std::move(attr));
  }

  // Sort attributes by name to enable binary search at run time.
  absl::c_sort(attrs->attributes,
               [](const NamedAttribute& a, const NamedAttribute& b) {
                 return a.name.value < b.name.value;
               });

  return FixUpAttrs(std::move(attrs));
}

std::unique_ptr<CallFrame::Attributes> CallFrame::FixUpAttrs(
    std::unique_ptr<CallFrame::Attributes> attrs) {
  size_t num_attrs = attrs->attributes.size();
  DCHECK(attrs->names.empty() && attrs->types.empty() && attrs->attrs.empty());

  attrs->names.reserve(num_attrs);
  attrs->types.reserve(num_attrs);
  attrs->attrs.reserve(num_attrs);

  // Fix up ZKX FFI structs to point to correct storage.
  for (NamedAttribute& attr : attrs->attributes) {
    std::invoke(FixUpAttribute{}, attr.name);
    std::visit(FixUpAttribute{}, attr.value);
  }

  // Initialize vectors required for building ZKX_FFI_Attributes.
  for (NamedAttribute& attr : attrs->attributes) {
    attrs->names.push_back(&attr.name.span);
    attrs->types.push_back(std::visit(AttributeType(), attr.value));
    attrs->attrs.push_back(std::visit(AttributeStorage(), attr.value));
  }

  // Finally initialize ZKX FFI struct. At this point all storage is allocated
  // and it's safe to grab a pointer to it.
  attrs->ffi_attrs.size = attrs->attributes.size();
  attrs->ffi_attrs.names = attrs->names.data();
  attrs->ffi_attrs.types = attrs->types.data();
  attrs->ffi_attrs.attrs = attrs->attrs.data();

  return attrs;
}

//===----------------------------------------------------------------------===//
// Call frame update
//===----------------------------------------------------------------------===//

absl::Status CallFrame::UpdateWithBuffers(
    absl::Span<const se::DeviceMemoryBase> args,
    absl::Span<const se::DeviceMemoryBase> rets) {
  if (ABSL_PREDICT_FALSE(args.size() != arguments_->args.size())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid number of updated arguments: ", args.size(),
                     " vs ", arguments_->args.size()));
  }

  if (ABSL_PREDICT_FALSE(rets.size() != results_->rets.size())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid number of updated results: ", rets.size(), " vs ",
                     results_->rets.size()));
  }

  size_t num_args = args.size();
  for (size_t i = 0; i < num_args; ++i) {
    arguments_->arguments[i].buffer.data = const_cast<void*>(args[i].opaque());
  }

  size_t num_rets = rets.size();
  for (size_t i = 0; i < num_rets; ++i) {
    results_->results[i].buffer.data = const_cast<void*>(rets[i].opaque());
  }

  return absl::OkStatus();
}

absl::StatusOr<CallFrame> CallFrame::CopyWithBuffers(
    absl::Span<const se::DeviceMemoryBase> args,
    absl::Span<const se::DeviceMemoryBase> rets) {
  CallFrame clone(CopyArgs(*arguments_), CopyRets(*results_), attributes_);
  TF_RETURN_IF_ERROR(clone.UpdateWithBuffers(args, rets));
  return clone;
}

}  // namespace zkx::ffi
