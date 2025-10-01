// RUN: cat %S/../default_print_utils.mlir %s \
// RUN:   | zkir-opt -field-to-gpu=parallelize-affine \
// RUN:   | mlir-runner -e testOffsetCreation -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../printI256%shlibext,%S/../../utils/cuda/cudaRuntimeUtils%shlibext,%mlir_lib_dir/libmlir_cuda_runtime%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_OFFSET_CREATION < %t

func.func private @sortPairsI64I64(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
func.func private @EncodeI64(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
func.func private @ExclusiveSumI64(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

func.func private @mgpuStreamCreate() -> !llvm.ptr
func.func private @mgpuStreamDestroy(!llvm.ptr)
func.func private @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
func.func private @mgpuMemFree(!llvm.ptr, !llvm.ptr)
func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

func.func @testOffsetCreation() {
  %c1 = arith.constant 1 : i64
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %c4 = arith.constant 4 : i64
  %c5 = arith.constant 5 : i64
  %c6 = arith.constant 6 : i64
  %c7 = arith.constant 7 : i64

  // [5, 2, 7, 4, 6, 3, 2]
  %keys_in = tensor.from_elements %c5, %c2, %c7, %c4, %c6, %c3, %c2 : tensor<7xi64>
  %keys_in_dynamic = tensor.cast %keys_in : tensor<7xi64> to tensor<?xi64>
  %keys_in_mem = bufferization.to_buffer %keys_in_dynamic : tensor<?xi64> to memref<?xi64>
  %keys_in_mem_ptr_index = memref.extract_aligned_pointer_as_index %keys_in_mem : memref<?xi64> -> index
  %keys_in_mem_ptr_i64 = arith.index_cast %keys_in_mem_ptr_index : index to i64
  %keys_in_mem_ptr = llvm.inttoptr %keys_in_mem_ptr_i64 : i64 to !llvm.ptr

  %keys_out = tensor.empty() : tensor<7xi64>
  %keys_out_dynamic = tensor.cast %keys_out : tensor<7xi64> to tensor<?xi64>
  %keys_out_mem = bufferization.to_buffer %keys_out_dynamic : tensor<?xi64> to memref<?xi64>
  %keys_out_mem_ptr_index = memref.extract_aligned_pointer_as_index %keys_out_mem : memref<?xi64> -> index
  %keys_out_mem_ptr_i64 = arith.index_cast %keys_out_mem_ptr_index : index to i64
  %keys_out_mem_ptr = llvm.inttoptr %keys_out_mem_ptr_i64 : i64 to !llvm.ptr

  // [3, 1, 2, 6, 5, 4, 4]
  %vals_in = tensor.from_elements %c3, %c1, %c2, %c6, %c5, %c4, %c4 : tensor<7xi64>
  %vals_in_dynamic = tensor.cast %vals_in : tensor<7xi64> to tensor<?xi64>
  %vals_in_mem = bufferization.to_buffer %vals_in_dynamic : tensor<?xi64> to memref<?xi64>
  %vals_in_mem_ptr_index = memref.extract_aligned_pointer_as_index %vals_in_mem : memref<?xi64> -> index
  %vals_in_mem_ptr_i64 = arith.index_cast %vals_in_mem_ptr_index : index to i64
  %vals_in_mem_ptr = llvm.inttoptr %vals_in_mem_ptr_i64 : i64 to !llvm.ptr

  %vals_out = tensor.empty() : tensor<7xi64>
  %vals_out_dynamic = tensor.cast %vals_out : tensor<7xi64> to tensor<?xi64>
  %vals_out_mem = bufferization.to_buffer %vals_out_dynamic : tensor<?xi64> to memref<?xi64>
  %vals_out_mem_ptr_index = memref.extract_aligned_pointer_as_index %vals_out_mem : memref<?xi64> -> index
  %vals_out_mem_ptr_i64 = arith.index_cast %vals_out_mem_ptr_index : index to i64
  %vals_out_mem_ptr = llvm.inttoptr %vals_out_mem_ptr_i64 : i64 to !llvm.ptr

  // Create CUDA stream
  %stream = func.call @mgpuStreamCreate() : () -> !llvm.ptr

  %true = arith.constant 1 : i1
  %keys_size = arith.constant 56 : i64 // 7 * sizeof(int64_t) = 7 * 8 = 56
  %d_keys_in = func.call @mgpuMemAlloc(%keys_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_keys_out = func.call @mgpuMemAlloc(%keys_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_vals_in = func.call @mgpuMemAlloc(%keys_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_vals_out = func.call @mgpuMemAlloc(%keys_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr

  // Copy input data from host to device
  func.call @mgpuMemcpy(%d_keys_in, %keys_in_mem_ptr, %keys_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemcpy(%d_vals_in, %vals_in_mem_ptr, %keys_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

  // Sort data on device
  // [2, 2, 3, 4, 5, 6, 7] (keys_out)
  // [1, 4, 4, 6, 3, 5, 2] (vals_out)
  func.call @sortPairsI64I64(%d_keys_in, %d_keys_out, %d_vals_in, %d_vals_out, %c7, %stream) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

  func.call @mgpuMemFree(%d_keys_in, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_vals_in, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  func.call @mgpuMemcpy(%vals_out_mem_ptr, %d_vals_out, %keys_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_vals_out, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  %unique_out_size = arith.constant 48 : i64 // 6 * sizeof(int64_t) = 6 * 8 = 48
  %d_unique_out = func.call @mgpuMemAlloc(%unique_out_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_counts_out = func.call @mgpuMemAlloc(%keys_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %int_size = arith.constant 8 : i64 // sizeof(int64_t) = 8
  %d_num_runs_out = func.call @mgpuMemAlloc(%int_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr

  // Generate unique_out, counts_out, and num_runs_out
  func.call @EncodeI64(%d_keys_out, %d_unique_out, %d_counts_out, %d_num_runs_out, %c7, %stream) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_num_runs_out, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_keys_out, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Generate offsets
  %d_offsets = func.call @mgpuMemAlloc(%keys_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  func.call @ExclusiveSumI64(%d_counts_out, %d_offsets, %c7, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_counts_out, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  %num_unique_out = arith.constant 6 : index
  %unique_out = tensor.empty(%num_unique_out) : tensor<?xi64>
  %unique_out_mem = bufferization.to_buffer %unique_out : tensor<?xi64> to memref<?xi64>
  %unique_out_ptr_index = memref.extract_aligned_pointer_as_index %unique_out_mem : memref<?xi64> -> index
  %unique_out_ptr_i64 = arith.index_cast %unique_out_ptr_index : index to i64
  %unique_out_ptr = llvm.inttoptr %unique_out_ptr_i64 : i64 to !llvm.ptr

  %num_offsets = arith.constant 7 : index
  %offsets = tensor.empty(%num_offsets) : tensor<?xi64>
  %offsets_mem = bufferization.to_buffer %offsets : tensor<?xi64> to memref<?xi64>
  %offsets_ptr_index = memref.extract_aligned_pointer_as_index %offsets_mem : memref<?xi64> -> index
  %offsets_ptr_i64 = arith.index_cast %offsets_ptr_index : index to i64
  %offsets_ptr = llvm.inttoptr %offsets_ptr_i64 : i64 to !llvm.ptr

  // Move values from device to host
  func.call @mgpuMemcpy(%unique_out_ptr, %d_unique_out, %unique_out_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemcpy(%offsets_ptr, %d_offsets, %keys_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

  func.call @mgpuMemFree(%d_unique_out, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_offsets, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Clean up CUDA stream
  func.call @mgpuStreamDestroy(%stream) : (!llvm.ptr) -> ()

  %vals_out_mem_cast = memref.cast %vals_out_mem : memref<?xi64> to memref<*xi64>
  func.call @printMemrefI64(%vals_out_mem_cast) : (memref<*xi64>) -> ()

  %unique_out_mem_cast = memref.cast %unique_out_mem : memref<?xi64> to memref<*xi64>
  func.call @printMemrefI64(%unique_out_mem_cast) : (memref<*xi64>) -> ()

  %offsets_mem_cast = memref.cast %offsets_mem : memref<?xi64> to memref<*xi64>
  func.call @printMemrefI64(%offsets_mem_cast) : (memref<*xi64>) -> ()
  return
}

// keys_in
//   [5, 2, 7, 4, 6, 3, 2]
// vals_in
//   [3, 1, 2, 6, 5, 4, 4]

// vals_out
//   CHECK_TEST_OFFSET_CREATION: [1, 4, 4, 6, 3, 5, 2]
// unique_out
//   CHECK_TEST_OFFSET_CREATION: [2, 3, 4, 5, 6, 7]
// offsets
//   CHECK_TEST_OFFSET_CREATION: [0, 2, 3, 4, 5, 6, 7]
