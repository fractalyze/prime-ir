// RUN: cat %S/../default_print_utils.mlir %s \
// RUN:   | zkir-opt -field-to-gpu=parallelize-affine \
// RUN:   | mlir-runner -e testSortPairs -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../printI256%shlibext,%S/../../utils/cuda/cudaRuntimeUtils%shlibext,%mlir_lib_dir/libmlir_cuda_runtime%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_SORT_PAIRS < %t

func.func private @sortPairsI64I64(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

func.func private @mgpuStreamCreate() -> !llvm.ptr
func.func private @mgpuStreamDestroy(!llvm.ptr)
func.func private @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

func.func @testSortPairs() {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %c4 = arith.constant 4 : i64
  %c5 = arith.constant 5 : i64
  %c6 = arith.constant 6 : i64
  %c7 = arith.constant 7 : i64

  // [5, 2, 7, 4, 6, 3]
  %keys_in = tensor.from_elements %c5, %c2, %c7, %c4, %c6, %c3 : tensor<6xi64>
  %keys_in_dynamic = tensor.cast %keys_in : tensor<6xi64> to tensor<?xi64>
  %keys_in_mem = bufferization.to_buffer %keys_in_dynamic : tensor<?xi64> to memref<?xi64>
  %keys_in_mem_ptr_index = memref.extract_aligned_pointer_as_index %keys_in_mem : memref<?xi64> -> index
  %keys_in_mem_ptr_i64 = arith.index_cast %keys_in_mem_ptr_index : index to i64
  %keys_in_mem_ptr = llvm.inttoptr %keys_in_mem_ptr_i64 : i64 to !llvm.ptr

  %keys_out = tensor.empty() : tensor<6xi64>
  %keys_out_dynamic = tensor.cast %keys_out : tensor<6xi64> to tensor<?xi64>
  %keys_out_mem = bufferization.to_buffer %keys_out_dynamic : tensor<?xi64> to memref<?xi64>
  %keys_out_mem_ptr_index = memref.extract_aligned_pointer_as_index %keys_out_mem : memref<?xi64> -> index
  %keys_out_mem_ptr_i64 = arith.index_cast %keys_out_mem_ptr_index : index to i64
  %keys_out_mem_ptr = llvm.inttoptr %keys_out_mem_ptr_i64 : i64 to !llvm.ptr

  // [3, 1, 2, 6, 5, 4]
  %values_in = tensor.from_elements %c3, %c1, %c2, %c6, %c5, %c4 : tensor<6xi64>
  %values_in_dynamic = tensor.cast %values_in : tensor<6xi64> to tensor<?xi64>
  %values_in_mem = bufferization.to_buffer %values_in_dynamic : tensor<?xi64> to memref<?xi64>
  %values_in_mem_ptr_index = memref.extract_aligned_pointer_as_index %values_in_mem : memref<?xi64> -> index
  %values_in_mem_ptr_i64 = arith.index_cast %values_in_mem_ptr_index : index to i64
  %values_in_mem_ptr = llvm.inttoptr %values_in_mem_ptr_i64 : i64 to !llvm.ptr

  %values_out = tensor.empty() : tensor<6xi64>
  %values_out_dynamic = tensor.cast %values_out : tensor<6xi64> to tensor<?xi64>
  %values_out_mem = bufferization.to_buffer %values_out_dynamic : tensor<?xi64> to memref<?xi64>
  %values_out_mem_ptr_index = memref.extract_aligned_pointer_as_index %values_out_mem : memref<?xi64> -> index
  %values_out_mem_ptr_i64 = arith.index_cast %values_out_mem_ptr_index : index to i64
  %values_out_mem_ptr = llvm.inttoptr %values_out_mem_ptr_i64 : i64 to !llvm.ptr

  // Create CUDA stream
  %stream = func.call @mgpuStreamCreate() : () -> !llvm.ptr

  // Allocate memory on GPU
  %true = arith.constant 1 : i1
  %alloc_size = arith.constant 48 : i64 // 6 * sizeof(int64_t) = 6 * 8 = 48
  %d_keys_in = func.call @mgpuMemAlloc(%alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_keys_out = func.call @mgpuMemAlloc(%alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_vals_in = func.call @mgpuMemAlloc(%alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_vals_out = func.call @mgpuMemAlloc(%alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr

  // Copy input data from host to GPU
  func.call @mgpuMemcpy(%d_keys_in, %keys_in_mem_ptr, %alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemcpy(%d_vals_in, %values_in_mem_ptr, %alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

  // Sort data on GPU
  // [2, 3, 4, 5, 6, 7] (keys_out)
  // [1, 4, 6, 3, 5, 2] (values_out)
  func.call @sortPairsI64I64(%d_keys_in, %d_keys_out, %d_vals_in, %d_vals_out, %c6, %stream) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

  // Copy sorted outputdata from GPU to host
  func.call @mgpuMemcpy(%keys_out_mem_ptr, %d_keys_out, %alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemcpy(%values_out_mem_ptr, %d_vals_out, %alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

  // Clean up CUDA stream
  func.call @mgpuStreamDestroy(%stream) : (!llvm.ptr) -> ()

  %keys_out_mem_cast = memref.cast %keys_out_mem : memref<?xi64> to memref<*xi64>
  func.call @printMemrefI64(%keys_out_mem_cast) : (memref<*xi64>) -> ()
  %values_out_mem_cast = memref.cast %values_out_mem : memref<?xi64> to memref<*xi64>
  func.call @printMemrefI64(%values_out_mem_cast) : (memref<*xi64>) -> ()
  return
}

// CHECK_TEST_SORT_PAIRS: [2, 3, 4, 5, 6, 7]
// CHECK_TEST_SORT_PAIRS: [1, 4, 6, 3, 5, 2]
