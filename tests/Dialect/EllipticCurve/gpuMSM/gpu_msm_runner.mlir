// RUN: cat %S/../../../default_print_utils.mlir %S/../../../bn254_field_defs.mlir %S/../../../bn254_ec_mont_defs.mlir %S/../../../bn254_ec_mont_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-gpu=parallelize-affine \
// RUN:   | mlir-runner -e test_gpu_msm -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../../printI256%shlibext,%S/../../../../utils/cuda/cudaRuntimeUtils%shlibext,%mlir_lib_dir/libmlir_cuda_runtime%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_GPU_MSM < %t

func.func private @mgpuStreamCreate() -> !llvm.ptr
func.func private @mgpuStreamDestroy(!llvm.ptr)
func.func private @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
func.func private @mgpuMemFree(!llvm.ptr, !llvm.ptr)
func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

func.func private @sortPairsI64I64(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
func.func private @EncodeI64(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
func.func private @ExclusiveSumI64(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

!SF_T = tensor<16x!SFm>
!SF_M = memref<16x!SFm>
!points_T = tensor<16x!affine>
!points_M = memref<16x!affine>
!split_scalars_T = tensor<32xindex>
!split_scalars_M = memref<32xindex>
!split_scalars_M_64 = memref<32xi64>
!buckets = tensor<8x!jacobian>
!buckets_2d = tensor<2x4x!jacobian>
!windows = tensor<2x!jacobian>

#bits_per_window = 2 : i16
#scalar_max_bits = 4 : i16

func.func @test_gpu_msm() {
  %scalar_max_bits = arith.constant 4 : i64
  %bits_per_window = arith.constant 2 : i64
  // 2^4 = 16
  %num_scalar_muls = arith.constant 16 : i64
  // scalar_max_bits / bits_per_window = 4 / 2 = 2
  %num_windows = arith.divsi %scalar_max_bits, %bits_per_window : i64 // 2
  // 2^(bits_per_window) * num_windows = 2^2 * 2 = 8
  %num_buckets = arith.constant 8 : i64
  // num_windows * num_scalar_muls = 2 * 16 = 32
  %num_split_scalars = arith.constant 32 : i64

  // Create points and scalars
  %p1 = field.constant 1 : !PF
  %p2 = field.constant 2 : !PF

  %p1_mont = field.to_mont %p1 : !PFm
  %p2_mont = field.to_mont %p2 : !PFm

  %g = elliptic_curve.point %p1_mont, %p2_mont : (!PFm, !PFm) -> !affine

  %points = tensor.splat %g : !points_T

  %scalars = tensor.generate{
  ^bb0(%i : index):
    %i256 = arith.index_cast %i : index to i256
    %elem = field.bitcast %i256 : i256 -> !SF
    %elem_mont = field.to_mont %elem : !SFm
    tensor.yield %elem_mont : !SFm
  } : !SF_T

  /////////////////////////////////
  ///////// scalar decomp /////////
  /////////////////////////////////

  // Takes in scalars tensor of size num_scalar_muls and outputs bucket_indices and point_indices tensors of size num_windows * num_scalar_muls
  // bits_per_window is chosen as desired.
  %bucket_indices, %point_indices = elliptic_curve.scalar_decomp %scalars {bitsPerWindow = #bits_per_window, scalarMaxBits = #scalar_max_bits} : (!SF_T) -> (!split_scalars_T, !split_scalars_T)

  /////////////////////////////////
  ///////////// sort //////////////
  /////////////////////////////////

  // Create bucket indices llvm ptr
  %bucket_indices_memref = bufferization.to_buffer %bucket_indices : !split_scalars_T to !split_scalars_M
  %bucket_indices_ptr_index = memref.extract_aligned_pointer_as_index %bucket_indices_memref : !split_scalars_M -> index
  %bucket_indices_ptr_i64 = arith.index_cast %bucket_indices_ptr_index : index to i64
  %bucket_indices_ptr = llvm.inttoptr %bucket_indices_ptr_i64 : i64 to !llvm.ptr
  // Create point indices llvm ptr
  %point_indices_memref = bufferization.to_buffer %point_indices : !split_scalars_T to !split_scalars_M
  %point_indices_ptr_index = memref.extract_aligned_pointer_as_index %point_indices_memref : !split_scalars_M -> index
  %point_indices_ptr_i64 = arith.index_cast %point_indices_ptr_index : index to i64
  %point_indices_ptr = llvm.inttoptr %point_indices_ptr_i64 : i64 to !llvm.ptr

  // Create CUDA stream
  %stream = func.call @mgpuStreamCreate() : () -> !llvm.ptr

  // Allocate memory on GPU for sortPairsI64I64()
  //
  // The "isHostShared" argument in mgpuMemAlloc()
  %true = arith.constant 1 : i1
  // sizeof(int64_t)
  %int_size = arith.constant 8 : i64
  // num_windows * num_scalar_muls * sizeof(int64_t)
  %indices_alloc_size = arith.muli %num_split_scalars, %int_size : i64
  %d_bucket_indices = func.call @mgpuMemAlloc(%indices_alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_sorted_bucket_indices = func.call @mgpuMemAlloc(%indices_alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_point_indices = func.call @mgpuMemAlloc(%indices_alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_sorted_point_indices = func.call @mgpuMemAlloc(%indices_alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr

  // Copy input data from host to GPU
  func.call @mgpuMemcpy(%d_bucket_indices, %bucket_indices_ptr, %indices_alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemcpy(%d_point_indices, %point_indices_ptr, %indices_alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

  // Sort d_bucket_indices and d_point_indices pairs to d_sorted_bucket_indices and d_sorted_point_indices
  func.call @sortPairsI64I64(%d_bucket_indices, %d_sorted_bucket_indices, %d_point_indices, %d_sorted_point_indices, %num_split_scalars, %stream) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_bucket_indices, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_point_indices, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Move sorted_point_indices from gpu to host
  %sorted_point_indices = tensor.empty() : !split_scalars_T
  %sorted_point_indices_mem = bufferization.to_buffer %sorted_point_indices : !split_scalars_T to !split_scalars_M
  %sorted_point_indices_ptr_index = memref.extract_aligned_pointer_as_index %sorted_point_indices_mem : !split_scalars_M -> index
  %sorted_point_indices_ptr_i64 = arith.index_cast %sorted_point_indices_ptr_index : index to i64
  %sorted_point_indices_ptr = llvm.inttoptr %sorted_point_indices_ptr_i64 : i64 to !llvm.ptr
  func.call @mgpuMemcpy(%sorted_point_indices_ptr, %d_sorted_point_indices, %indices_alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_sorted_point_indices, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  %sorted_point_indices_mem_cast = arith.index_cast %sorted_point_indices_mem : !split_scalars_M to !split_scalars_M_64
  %sorted_point_indices_mem_cast_cast = memref.cast %sorted_point_indices_mem_cast : !split_scalars_M_64 to memref<*xi64>
  func.call @printMemrefI64(%sorted_point_indices_mem_cast_cast) : (memref<*xi64>) -> ()

  /////////////////////////////////////
  ///////// calculate offsets /////////
  /////////////////////////////////////

  // Allocate memory on GPU for EncodeI64()
  //
  // num_windows - 1 buckets become the zero bucket anyway, so we can remove this from the allocation computation.
  // unique_bucket_indices_alloc_size = (num_buckets - num_windows + 1) * sizeof(int64_t)
  %c1 = arith.constant 1 : i64
  %temp1 = arith.subi %num_buckets, %num_windows : i64
  %temp2 = arith.addi %temp1, %c1 : i64
  %unique_bucket_indices_alloc_size = arith.muli %temp2, %int_size : i64
  %d_sorted_unique_bucket_indices = func.call @mgpuMemAlloc(%unique_bucket_indices_alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  // Note that for the ExclusiveSumI64() later on, the size of bucket_sizes must be greater than or equal to the size of bucket_offsets.
  // As bucket_offsets should have one more value than bucket_sizes, we allocate max expected size of bucket_sizes + 1 to bucket_sizes
  // bucket_offsets_alloc_size = (num_buckets - num_windows + 1 + 1) * sizeof(int64_t)
  %temp3 = arith.addi %temp2, %c1 : i64
  %bucket_offsets_alloc_size = arith.muli %temp3, %int_size : i64
  %d_bucket_sizes = func.call @mgpuMemAlloc(%bucket_offsets_alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr
  %d_nof_buckets_to_compute = func.call @mgpuMemAlloc(%int_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr

  // Takes in d_sorted_bucket_indices and num_split_scalars and outputs d_sorted_unique_bucket_indices, d_bucket_sizes, and d_nof_buckets_to_compute
  func.call @EncodeI64(%d_sorted_bucket_indices, %d_sorted_unique_bucket_indices, %d_bucket_sizes, %d_nof_buckets_to_compute, %num_split_scalars, %stream) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_sorted_bucket_indices, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Move nof_buckets_to_compute from gpu to host
  %nof_buckets_to_compute_mem = memref.alloc() : memref<i64>
  %nof_buckets_to_compute_ptr_index = memref.extract_aligned_pointer_as_index %nof_buckets_to_compute_mem : memref<i64> -> index
  %nof_buckets_to_compute_ptr_i64 = arith.index_cast %nof_buckets_to_compute_ptr_index : index to i64
  %nof_buckets_to_compute_ptr = llvm.inttoptr %nof_buckets_to_compute_ptr_i64 : i64 to !llvm.ptr
  func.call @mgpuMemcpy(%nof_buckets_to_compute_ptr, %d_nof_buckets_to_compute, %int_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_nof_buckets_to_compute, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Move sorted_unique_bucket_indices from gpu to host
  %nof_buckets_to_compute = memref.load %nof_buckets_to_compute_mem[] : memref<i64>
  %nof_buckets_to_compute_index = arith.index_cast %nof_buckets_to_compute : i64 to index
  %sorted_unique_bucket_indices = tensor.empty(%nof_buckets_to_compute_index) : tensor<?xindex>
  %sorted_unique_bucket_indices_mem = bufferization.to_buffer %sorted_unique_bucket_indices : tensor<?xindex> to memref<?xindex>
  %sorted_unique_bucket_indices_ptr_index = memref.extract_aligned_pointer_as_index %sorted_unique_bucket_indices_mem : memref<?xindex> -> index
  %sorted_unique_bucket_indices_ptr_i64 = arith.index_cast %sorted_unique_bucket_indices_ptr_index : index to i64
  %sorted_unique_bucket_indices_ptr = llvm.inttoptr %sorted_unique_bucket_indices_ptr_i64 : i64 to !llvm.ptr
  %sorted_unique_bucket_indices_size = arith.muli %nof_buckets_to_compute, %int_size : i64
  func.call @mgpuMemcpy(%sorted_unique_bucket_indices_ptr, %d_sorted_unique_bucket_indices, %sorted_unique_bucket_indices_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_sorted_unique_bucket_indices, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  %sorted_unique_bucket_indices_mem_cast = arith.index_cast %sorted_unique_bucket_indices_mem : memref<?xindex> to memref<?xi64>
  %sorted_unique_bucket_indices_mem_cast_cast = memref.cast %sorted_unique_bucket_indices_mem_cast : memref<?xi64> to memref<*xi64>
  func.call @printMemrefI64(%sorted_unique_bucket_indices_mem_cast_cast) : (memref<*xi64>) -> ()

  // Create bucket_offsets llvm ptr
  %nof_buckets_to_compute_and_1 = arith.addi %nof_buckets_to_compute, %c1 : i64
  %nof_buckets_to_compute_and_1_index = arith.index_cast %nof_buckets_to_compute_and_1 : i64 to index
  %bucket_offsets = tensor.empty(%nof_buckets_to_compute_and_1_index) : tensor<?xindex>
  %bucket_offsets_mem = bufferization.to_buffer %bucket_offsets : tensor<?xindex> to memref<?xindex>
  %bucket_offsets_ptr_index = memref.extract_aligned_pointer_as_index %bucket_offsets_mem : memref<?xindex> -> index
  %bucket_offsets_ptr_i64 = arith.index_cast %bucket_offsets_ptr_index : index to i64
  %bucket_offsets_ptr = llvm.inttoptr %bucket_offsets_ptr_i64 : i64 to !llvm.ptr

  // Allocate memory on GPU for ExclusiveSumI64()
  %d_bucket_offsets = func.call @mgpuMemAlloc(%bucket_offsets_alloc_size, %stream, %true) : (i64, !llvm.ptr, i1) -> !llvm.ptr

  // Takes in d_bucket_sizes and nof_buckets_to_compute_and_1 and generates d_bucket_offsets
  // The third argument of ExclusiveSumI64() MUST be the size of bucket_offsets to get the intended number of bucket offsets.
  func.call @ExclusiveSumI64(%d_bucket_sizes, %d_bucket_offsets, %nof_buckets_to_compute_and_1, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_bucket_sizes, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Move bucket_offsets from gpu to host
  %bucket_offsets_size = arith.muli %nof_buckets_to_compute_and_1, %int_size : i64
  func.call @mgpuMemcpy(%bucket_offsets_ptr, %d_bucket_offsets, %bucket_offsets_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_bucket_offsets, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  %bucket_offsets_mem_cast = arith.index_cast %bucket_offsets_mem : memref<?xindex> to memref<?xi64>
  %bucket_offsets_mem_cast_cast = memref.cast %bucket_offsets_mem_cast : memref<?xi64> to memref<*xi64>
  func.call @printMemrefI64(%bucket_offsets_mem_cast_cast) : (memref<*xi64>) -> ()

  // Clean up CUDA stream
  func.call @mgpuStreamDestroy(%stream) : (!llvm.ptr) -> ()

  ///////////////////////////////////////
  ///////// bucket accumulation /////////
  ///////////////////////////////////////

  %buckets = elliptic_curve.bucket_acc %points, %sorted_point_indices, %sorted_unique_bucket_indices, %bucket_offsets: (!points_T, !split_scalars_T, tensor<?xindex>, tensor<?xindex>) -> !buckets

  ////////////////////////////////////
  ///////// bucket reduction /////////
  ////////////////////////////////////

  // Reshape buckets to 2D tensor
  %num_buckets_index = arith.index_cast %num_buckets : i64 to index
  %num_windows_index = arith.index_cast %num_windows : i64 to index
  %buckets_per_window = arith.divsi %num_buckets_index, %num_windows_index : index
  %shape = tensor.from_elements %num_windows_index, %buckets_per_window : tensor<2xindex>
  %buckets_2d = tensor.reshape %buckets(%shape) : (!buckets, tensor<2xindex>) -> !buckets_2d

  %windows = elliptic_curve.bucket_reduce %buckets_2d {scalarType = !SFm}: (!buckets_2d) -> !windows

  ////////////////////////////////////
  ///////// window reduction /////////
  ////////////////////////////////////

  %res = elliptic_curve.window_reduce %windows {bitsPerWindow = #bits_per_window, scalarMaxBits = #scalar_max_bits, scalarType = !SFm}: (!windows) -> !jacobian
  func.call @printAffineFromJacobian(%res) : (!jacobian) -> ()

  %c120 = field.constant 120 : !SF
  %true_result = func.call @getGeneratorMultiple(%c120) : (!SF) -> (!affine)
  func.call @printAffine(%true_result) : (!affine) -> ()
  return
}

// points
//   [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
// scalars
//   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

// bucket_indices
//   [0, 0, 1, 0, 2, 0, 3, 0, 0, 5, 1, 5, 2, 5, 3, 5, 0, 6, 1, 6, 2, 6, 3, 6, 0, 7, 1, 7, 2, 7, 3, 7]
// point_indices
//   [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]

// sorted_point_indices
//   CHECK_TEST_GPU_MSM: [0, 0, 1, 2, 3, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// sorted_unique_bucket_indices
//   CHECK_TEST_GPU_MSM: [0, 1, 2, 3, 5, 6, 7]
// offsets
//   CHECK_TEST_GPU_MSM: [0, 8, 12, 16, 20, 24, 28, 32]

// msm result
//   CHECK_TEST_GPU_MSM: [2747517507890653313006032249699734168352039494722462666318484735518429114319, 17769594319884551394326400904703145588834032543352917093414832572556669509807]
// expected msm result
//   CHECK_TEST_GPU_MSM: [2747517507890653313006032249699734168352039494722462666318484735518429114319, 17769594319884551394326400904703145588834032543352917093414832572556669509807]
