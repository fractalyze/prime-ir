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

#ifndef ZKX_FFI_API_C_API_H_
#define ZKX_FFI_API_C_API_H_

#include <stddef.h>
#include <stdint.h>

// ZKX FFI C API follows PJRT API style for consistency. See `pjrt_c_api.h`.
// More details on versioning strategy and example version checks:
// https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api/C_API_versioning_strategy.md

// Every struct passed across the C API boundary has its size as a member, and
// we use it as a sanity check for API compatibility.
#define ZKX_FFI_STRUCT_SIZE(struct_type, last_field) \
  /* NOLINTNEXTLINE(readability/casting) */          \
  (offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field))

// Must update ZKX_FFI_DEFINE_STRUCT_TRAITS with the new `last_field` after
// adding a new member to a struct.
#define ZKX_FFI_DEFINE_STRUCT_TRAITS(sname, last_field) \
  typedef struct sname sname;                           \
  enum { sname##_STRUCT_SIZE = ZKX_FFI_STRUCT_SIZE(sname, last_field) }

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZKX_FFI_Api ZKX_FFI_Api;                  // Forward declare
typedef struct ZKX_FFI_InternalApi ZKX_FFI_InternalApi;  // Forward declare

//===----------------------------------------------------------------------===//
// Extensions
//===----------------------------------------------------------------------===//

typedef enum {
  ZKX_FFI_Extension_Metadata = 1,
} ZKX_FFI_Extension_Type;

typedef struct ZKX_FFI_Extension_Base {
  size_t struct_size;
  ZKX_FFI_Extension_Type type;
  struct ZKX_FFI_Extension_Base* next;
} ZKX_FFI_Extension_Base;

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Extension_Base, next);

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

// Incremented when an ABI-incompatible change is made to the interface.
//
// Major changes include:
// * Deleting a method or argument
// * Changing the type of an argument
// * Rearranging fields in the ZKX_FFI_Api or argument structs
#define ZKX_FFI_API_MAJOR 0

// Incremented when the interface is updated in a way that is potentially
// ABI-compatible with older versions, if supported by the caller and/or
// implementation.
//
// Callers can implement forwards compatibility by using ZKX_FFI_Api_Version to
// check if the implementation is aware of newer interface additions.
//
// Implementations can implement backwards compatibility by using the
// `struct_size` fields to detect how many struct fields the caller is aware of.
//
// Minor changes include:
// * Adding a new field to the ZKX_FFI_Api or argument structs
// * Renaming a method or argument (doesn't affect ABI)
#define ZKX_FFI_API_MINOR 1

struct ZKX_FFI_Api_Version {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;
  int major_version;  // out
  int minor_version;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Api_Version, minor_version);

//===----------------------------------------------------------------------===//
// Error codes
//===----------------------------------------------------------------------===//

// ZKX FFI error is a mechanism to communicate errors between ZKX and ZKX FFI
// via a set of C APIs. This is somewhat similar to type-erased version of
// absl::Status exposed via API with opaque pointers.
//
// Returning NULL error is equivalent to returning absl::OkStatus().
//
// Ownership of an ZKX_FFI_Error is always transferred to the caller, and the
// caller is responsible for destroying it:
//
// (1) If the error is returned from an ZKX FFI handler, the ZKX runtime will
//     destroy it (ZKX is the caller who calls into the handler implementation).
//
// (2) If the error is returned from an ZKX FFI API call, the caller is
//     responsible for destroying it.
typedef struct ZKX_FFI_Error ZKX_FFI_Error;

// Codes are based on https://abseil.io/docs/cpp/guides/status-codes
typedef enum {
  ZKX_FFI_Error_Code_OK = 0,
  ZKX_FFI_Error_Code_CANCELLED = 1,
  ZKX_FFI_Error_Code_UNKNOWN = 2,
  ZKX_FFI_Error_Code_INVALID_ARGUMENT = 3,
  ZKX_FFI_Error_Code_DEADLINE_EXCEEDED = 4,
  ZKX_FFI_Error_Code_NOT_FOUND = 5,
  ZKX_FFI_Error_Code_ALREADY_EXISTS = 6,
  ZKX_FFI_Error_Code_PERMISSION_DENIED = 7,
  ZKX_FFI_Error_Code_RESOURCE_EXHAUSTED = 8,
  ZKX_FFI_Error_Code_FAILED_PRECONDITION = 9,
  ZKX_FFI_Error_Code_ABORTED = 10,
  ZKX_FFI_Error_Code_OUT_OF_RANGE = 11,
  ZKX_FFI_Error_Code_UNIMPLEMENTED = 12,
  ZKX_FFI_Error_Code_INTERNAL = 13,
  ZKX_FFI_Error_Code_UNAVAILABLE = 14,
  ZKX_FFI_Error_Code_DATA_LOSS = 15,
  ZKX_FFI_Error_Code_UNAUTHENTICATED = 16
} ZKX_FFI_Error_Code;

//===----------------------------------------------------------------------===//
// Error reporting APIs
//===----------------------------------------------------------------------===//

struct ZKX_FFI_Error_Create_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;
  const char* message;
  ZKX_FFI_Error_Code errc;
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Error_Create_Args, errc);

typedef ZKX_FFI_Error* ZKX_FFI_Error_Create(ZKX_FFI_Error_Create_Args* args);

struct ZKX_FFI_Error_GetMessage_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;
  ZKX_FFI_Error* error;
  const char* message;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Error_GetMessage_Args, message);

typedef void ZKX_FFI_Error_GetMessage(ZKX_FFI_Error_GetMessage_Args* args);

struct ZKX_FFI_Error_Destroy_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;
  ZKX_FFI_Error* error;
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Error_Destroy_Args, error);

typedef void ZKX_FFI_Error_Destroy(ZKX_FFI_Error_Destroy_Args* args);

//===----------------------------------------------------------------------===//
// DataType
//===----------------------------------------------------------------------===//

// This enum corresponds to zkx::PrimitiveType enum defined in `zkx_data.proto`.
// Auto-generated from zkx_data.proto via gen_ffi_data_types.py.
#include "zkx/ffi/api/c_api_data_types.h"

//===----------------------------------------------------------------------===//
// Builtin argument types
//===----------------------------------------------------------------------===//

struct ZKX_FFI_Buffer {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_DataType dtype;
  void* data;
  int64_t rank;
  int64_t* dims;  // length == rank
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Buffer, dims);

typedef enum {
  ZKX_FFI_ArgType_BUFFER = 1,
} ZKX_FFI_ArgType;

//===----------------------------------------------------------------------===//
// Builtin result types
//===----------------------------------------------------------------------===//

typedef enum {
  ZKX_FFI_RetType_BUFFER = 1,
} ZKX_FFI_RetType;

//===----------------------------------------------------------------------===//
// Builtin attribute types
//===----------------------------------------------------------------------===//

typedef enum {
  ZKX_FFI_AttrType_ARRAY = 1,
  ZKX_FFI_AttrType_DICTIONARY = 2,
  ZKX_FFI_AttrType_SCALAR = 3,
  ZKX_FFI_AttrType_STRING = 4,
} ZKX_FFI_AttrType;

//===----------------------------------------------------------------------===//
// Execution context
//===----------------------------------------------------------------------===//

// Execution context provides access to per-invocation state.
typedef struct ZKX_FFI_ExecutionContext ZKX_FFI_ExecutionContext;

//===----------------------------------------------------------------------===//
// Primitives
//===----------------------------------------------------------------------===//

// TypeId uniquely identifies a user-defined type in a given ZKX FFI instance.
typedef struct ZKX_FFI_TypeId {
  int64_t type_id;
} ZKX_FFI_TypeId;

// We use byte spans to pass strings to handlers because strings might not be
// null terminated, and even if they are, looking for a null terminator can
// become very expensive in tight loops.
typedef struct ZKX_FFI_ByteSpan {
  const char* ptr;
  size_t len;
} ZKX_FFI_ByteSpan;

// A struct to pass a scalar value to FFI handler.
typedef struct ZKX_FFI_Scalar {
  ZKX_FFI_DataType dtype;
  void* value;
} ZKX_FFI_Scalar;

// A struct to pass a dense array to FFI handler.
typedef struct ZKX_FFI_Array {
  ZKX_FFI_DataType dtype;
  size_t size;
  void* data;
} ZKX_FFI_Array;

//===----------------------------------------------------------------------===//
// Future
//===----------------------------------------------------------------------===//

// ZKX FFI future is a mechanism to signal a result of asynchronous computation
// (FFI handler) to the ZKX runtime. It is similar to `std::future<void>` in C++
// standard library, and implemented on top of `tsl::AsyncValue` in ZKX runtime.
//
// ZKX FFI users should use `Future` and `Promise` types defined in `zkx::ffi`
// namespace (see `ffi/api/ffi.h`), instead of using this API directly.
typedef struct ZKX_FFI_Future ZKX_FFI_Future;

struct ZKX_FFI_Future_Create_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;
  ZKX_FFI_Future* future;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Future_Create_Args, extension_start);

typedef ZKX_FFI_Error* ZKX_FFI_Future_Create(ZKX_FFI_Future_Create_Args* args);

struct ZKX_FFI_Future_SetAvailable_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;
  ZKX_FFI_Future* future;
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Future_SetAvailable_Args, future);

typedef ZKX_FFI_Error* ZKX_FFI_Future_SetAvailable(
    ZKX_FFI_Future_SetAvailable_Args* args);

struct ZKX_FFI_Future_SetError_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;
  ZKX_FFI_Future* future;
  ZKX_FFI_Error* error;  // ownership is transferred to the ZKX runtime
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Future_SetError_Args, error);

typedef ZKX_FFI_Error* ZKX_FFI_Future_SetError(
    ZKX_FFI_Future_SetError_Args* args);

//===----------------------------------------------------------------------===//
// Call frame
//===----------------------------------------------------------------------===//

// ZKX runtime has multiple execution stages and it is possible to run
// different handlers for each stage:
//
// (1) Instantiate - called when FFI handler is instantiated as a part of ZKX
//     executable instantiation. Every call site will have its own "instance" of
//     the FFI handler, and it is possible to attach an arbitrary user-defined
//     state to the FFI handler instance, and get it back in other execution
//     stages. Constructed state owned by the ZKX runtime and destructed
//     together with a parent executable.
//
// (2) Prepare - called before the execution to let FFI handlers to prepare
//     for the execution and request resources from runtime, i.e. in ZKX:GPU
//     we use prepare stage to request collective cliques.
//
// (3) Initialize - called before the execution after acquiring all the
//     resources requested in the prepare stage.
//
// (4) Execute - called when FFI handler is executed. Note that FFI handler
//     can be called as a part of command buffer capture (CUDA graph capture
//     on GPU backend) and argument buffers might contain uninitialized
//     values in this case.
//
// ZKX program (HLO module) compiled to an ZKX executable that can be executed
// on any device accessible to the process, and by extension FFI handlers are
// not instantiated for any particular device, but for a process. FFI handlers
// running at instantiation stage do not have access to the underlying device
// (memory allocation, stream, etc.) and arguments, however they can access
// execution context and attributes.
//
// It is undefined behavior to access argument buffers in prepare and initialize
// stages as they might not be initialized yet. However it is safe to use memory
// address as it is assigned ahead of time by buffer assignment.
typedef enum {
  ZKX_FFI_ExecutionStage_INSTANTIATE = 0,
  ZKX_FFI_ExecutionStage_PREPARE = 1,
  ZKX_FFI_ExecutionStage_INITIALIZE = 2,
  ZKX_FFI_ExecutionStage_EXECUTE = 3,
} ZKX_FFI_ExecutionStage;

struct ZKX_FFI_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  int64_t size;
  ZKX_FFI_ArgType* types;  // length == size
  void** args;             // length == size
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Args, args);

struct ZKX_FFI_Rets {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  int64_t size;
  ZKX_FFI_RetType* types;  // length == size
  void** rets;             // length == size
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Rets, rets);

// FFI handler attributes are always sorted by name, so that the handler can
// rely on binary search to look up attributes by name.
struct ZKX_FFI_Attrs {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  int64_t size;
  ZKX_FFI_AttrType* types;   // length == size
  ZKX_FFI_ByteSpan** names;  // length == size
  void** attrs;              // length == size
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Attrs, attrs);

struct ZKX_FFI_CallFrame {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  const ZKX_FFI_Api* api;
  ZKX_FFI_ExecutionContext* ctx;
  ZKX_FFI_ExecutionStage stage;
  ZKX_FFI_Args args;
  ZKX_FFI_Rets rets;
  ZKX_FFI_Attrs attrs;

  // ZKX FFI handler implementation can use `future` to signal a result of
  // asynchronous computation to the ZKX runtime. ZKX runtime will keep all
  // arguments, results and attributes alive until `future` is completed.
  ZKX_FFI_Future* future;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_CallFrame, attrs);

//===----------------------------------------------------------------------===//
// FFI handler
//===----------------------------------------------------------------------===//

// External functions registered with ZKX as FFI handlers.
typedef ZKX_FFI_Error* ZKX_FFI_Handler(ZKX_FFI_CallFrame* call_frame);

// ZKX FFI handlers for execution stages (see ZKX_FFI_ExecutionStage).
typedef struct ZKX_FFI_Handler_Bundle {
  ZKX_FFI_Handler* instantiate;  // optional
  ZKX_FFI_Handler* prepare;      // optional
  ZKX_FFI_Handler* initialize;   // optional
  ZKX_FFI_Handler* execute;      // required
} ZKX_FFI_Handler_Bundle;

enum ZKX_FFI_Handler_TraitsBits {
  // Calls to FFI handler are safe to trace into the command buffer. It means
  // that calls to FFI handler always launch exactly the same device operations
  // (can depend on attribute values) that can be captured and then replayed.
  ZKX_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE = 1u << 0,
};

typedef uint32_t ZKX_FFI_Handler_Traits;

struct ZKX_FFI_Handler_Register_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ByteSpan name;
  ZKX_FFI_ByteSpan platform;
  ZKX_FFI_Handler_Bundle bundle;
  ZKX_FFI_Handler_Traits traits;
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Handler_Register_Args, traits);

typedef ZKX_FFI_Error* ZKX_FFI_Handler_Register(
    ZKX_FFI_Handler_Register_Args* args);

//===----------------------------------------------------------------------===//
// TypeId
//===----------------------------------------------------------------------===//

struct ZKX_FFI_TypeId_Register_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ByteSpan name;
  ZKX_FFI_TypeId* type_id;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_TypeId_Register_Args, type_id);

// Registers user type `name` and returns a unique `type_id`.
typedef ZKX_FFI_Error* ZKX_FFI_TypeId_Register(
    ZKX_FFI_TypeId_Register_Args* args);

//===----------------------------------------------------------------------===//
// ExecutionContext
//===----------------------------------------------------------------------===//

struct ZKX_FFI_ExecutionContext_Get_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  ZKX_FFI_TypeId* type_id;
  void* data;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_ExecutionContext_Get_Args, data);

// Returns an opaque data from the execution context for a given type id.
typedef ZKX_FFI_Error* ZKX_FFI_ExecutionContext_Get(
    ZKX_FFI_ExecutionContext_Get_Args* args);

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

struct ZKX_FFI_State_Set_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  ZKX_FFI_TypeId* type_id;
  void* state;
  void (*deleter)(void* state);
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_State_Set_Args, deleter);

// Sets execution state to the `state` of type `type_id`. Returns an error if
// state already set.
typedef ZKX_FFI_Error* ZKX_FFI_State_Set(ZKX_FFI_State_Set_Args* args);

struct ZKX_FFI_State_Get_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  ZKX_FFI_TypeId* type_id;
  void* state;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_State_Get_Args, state);

// Gets execution state of type `type_id`. Returns an error if state is not set,
// or set with a state of a different type.
typedef ZKX_FFI_Error* ZKX_FFI_State_Get(ZKX_FFI_State_Get_Args* args);

//===----------------------------------------------------------------------===//
// Stream
//===----------------------------------------------------------------------===//

struct ZKX_FFI_Stream_Get_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  void* stream;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Stream_Get_Args, stream);

// Returns an underling platform-specific stream via out argument, i.e. for CUDA
// platform it returns `CUstream` (same as `cudaStream`).
typedef ZKX_FFI_Error* ZKX_FFI_Stream_Get(ZKX_FFI_Stream_Get_Args* args);

//===----------------------------------------------------------------------===//
// Device memory allocation
//===----------------------------------------------------------------------===//

struct ZKX_FFI_DeviceMemory_Allocate_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  size_t size;
  size_t alignment;
  void* data;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_DeviceMemory_Allocate_Args, data);

// Allocates a block of memory on the device bound to the execution context.
typedef ZKX_FFI_Error* ZKX_FFI_DeviceMemory_Allocate(
    ZKX_FFI_DeviceMemory_Allocate_Args* args);

struct ZKX_FFI_DeviceMemory_Free_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  size_t size;
  void* data;
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_DeviceMemory_Free_Args, data);

// Frees previously allocated device memory.
typedef ZKX_FFI_Error* ZKX_FFI_DeviceMemory_Free(
    ZKX_FFI_DeviceMemory_Free_Args* args);

//===----------------------------------------------------------------------===//
// ThreadPool
//===----------------------------------------------------------------------===//

// A function pointer for a task to be scheduled on a thread pool. ZKX runtime
// will call this function with a user-defined `data` pointer on one of the
// runtime-managed threads. For ZKX:CPU backends the task will be invoked on
// a thread pool that runs all compute tasks (Eigen thread pool).
//
// IMPORTANT: Users must not rely on any particular execution order or the
// number of available threads. Tasks can be executed in the caller thread, or
// in a thread pool with size `1`, and it is unsafe to assume that all scheduled
// tasks can be executed in parallel.
typedef void ZKX_FFI_Task(void* data);

struct ZKX_FFI_ThreadPool_Schedule_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  ZKX_FFI_Task* task;
  void* data;
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_ThreadPool_Schedule_Args, data);

// Schedules a task to be executed on a thread pool managed by ZKX runtime.
// Returns an error if thread pool is not available.
typedef ZKX_FFI_Error* ZKX_FFI_ThreadPool_Schedule(
    ZKX_FFI_ThreadPool_Schedule_Args* args);

struct ZKX_FFI_ThreadPool_NumThreads_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  int64_t* num_threads;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_ThreadPool_NumThreads_Args, num_threads);

// Returns the number of threads in the thread pool managed by ZKX runtime.
typedef ZKX_FFI_Error* ZKX_FFI_ThreadPool_NumThreads(
    ZKX_FFI_ThreadPool_NumThreads_Args* args);

//===----------------------------------------------------------------------===//
// RunId
//===----------------------------------------------------------------------===//

// RunId is a unique identifier for a particular "logical execution" of an ZKX
// model.
//
// A logical execution might encompass multiple executions of one or more
// HloModules. Runs that are part of the same logical execution can communicate
// via collective ops, whereas runs that are part of different logical
// executions are isolated.
//
// Corresponds to `::zkx::RunId` (see `zkx/executable_run_options.h`).

struct ZKX_FFI_RunId_Get_Args {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_ExecutionContext* ctx;
  int64_t run_id;  // out
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_RunId_Get_Args, run_id);

// Returns a unique identifier for the current logical execution.
typedef ZKX_FFI_Error* ZKX_FFI_RunId_Get(ZKX_FFI_RunId_Get_Args* args);

//===----------------------------------------------------------------------===//
// Metadata extension
//===----------------------------------------------------------------------===//

struct ZKX_FFI_Metadata {
  size_t struct_size;
  ZKX_FFI_Api_Version api_version;
  ZKX_FFI_Handler_Traits traits;
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Metadata, traits);

struct ZKX_FFI_Metadata_Extension {
  ZKX_FFI_Extension_Base extension_base;
  ZKX_FFI_Metadata* metadata;
};

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Metadata_Extension, metadata);

//===----------------------------------------------------------------------===//
// API access
//===----------------------------------------------------------------------===//

#define _ZKX_FFI_API_STRUCT_FIELD(fn_type) fn_type* fn_type

struct ZKX_FFI_Api {
  size_t struct_size;
  ZKX_FFI_Extension_Base* extension_start;

  ZKX_FFI_Api_Version api_version;
  ZKX_FFI_InternalApi* internal_api;

  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_Error_Create);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_Error_GetMessage);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_Error_Destroy);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_Handler_Register);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_Stream_Get);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_TypeId_Register);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_ExecutionContext_Get);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_State_Set);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_State_Get);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_DeviceMemory_Allocate);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_DeviceMemory_Free);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_ThreadPool_Schedule);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_ThreadPool_NumThreads);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_Future_Create);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_Future_SetAvailable);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_Future_SetError);
  _ZKX_FFI_API_STRUCT_FIELD(ZKX_FFI_RunId_Get);
};

#undef _ZKX_FFI_API_STRUCT_FIELD

ZKX_FFI_DEFINE_STRUCT_TRAITS(ZKX_FFI_Api, ZKX_FFI_Stream_Get);

const ZKX_FFI_Api* ZKX_FFI_GetApi();

#ifdef __cplusplus
}
#endif

#endif  // ZKX_FFI_API_C_API_H_
