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

#ifndef TESTS_RUNTIMEFUNCTIONS_H_
#define TESTS_RUNTIMEFUNCTIONS_H_

#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "zk_dtypes/include/all_types.h"

extern "C" {
// clang-format off
#define DECLARE_PRINT_MEMREF_FUNCTION(ActualType, UpperCamelCaseName, ...)                 /*NOLINT(whitespace/line_length)*/\
MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_printMemref##UpperCamelCaseName(void *memref);   /*NOLINT(whitespace/line_length)*/\
MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_printMemref1D##UpperCamelCaseName(void *memref); /*NOLINT(whitespace/line_length)*/\
MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_printMemref2D##UpperCamelCaseName(void *memref); /*NOLINT(whitespace/line_length)*/\
MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_printMemref3D##UpperCamelCaseName(void *memref); /*NOLINT(whitespace/line_length)*/\
ZK_DTYPES_ALL_TYPE_LIST(DECLARE_PRINT_MEMREF_FUNCTION)
#undef DECLARE_PRINT_MEMREF_FUNCTION
// clang-format on
MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_printMemrefI256(void *memref);
MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_printMemref1DI256(void *memref);
MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_printMemref2DI256(void *memref);
MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_printMemref3DI256(void *memref);
}

#endif // TESTS_RUNTIMEFUNCTIONS_H_
