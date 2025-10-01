!PF = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!SF = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
!PFm = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256, true>
!SFm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256, true>

#a = #field.pf.elem<0:i256> : !PFm
#b = #field.pf.elem<3:i256> : !PFm
#1 = #field.pf.elem<1:i256> : !PFm
#2 = #field.pf.elem<2:i256> : !PFm

#curve = #elliptic_curve.sw<#a, #b, (#1, #2)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

// Degree = 20
!SF_T = tensor<1048576x!SFm>
!SF_M = memref<1048576x!SFm>
!points_T = tensor<1048576x!affine>
!points_M = memref<1048576x!affine>
!split_scalars_T = tensor<16777216xindex>
!split_scalars_M = memref<16777216xindex>
!buckets = tensor<1048576x!jacobian>
!buckets_2d = tensor<16x65536x!jacobian>
!windows = tensor<16x!jacobian>

func.func private @getGeneratorMultiple(%k: !SF) -> !affine {
  %onePF = field.constant 1 : !PF
  %twoPF = field.constant 2 : !PF
  %onePFm = field.to_mont %onePF : !PFm
  %twoPFm = field.to_mont %twoPF : !PFm
  %g = elliptic_curve.point %onePFm, %twoPFm : (!PFm, !PFm) -> !affine
  %g_multiple = elliptic_curve.scalar_mul %k, %g : !SF, !affine -> !jacobian
  %g_multiple_affine = elliptic_curve.convert_point_type %g_multiple : !jacobian -> !affine
  return %g_multiple_affine : !affine
}

func.func private @generate_points(%rand_scalar : memref<!SF>, %points : !points_M) attributes { llvm.emit_c_interface } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rand_scalar_extracted = memref.load %rand_scalar[] : memref<!SF>
  %first_point = func.call @getGeneratorMultiple(%rand_scalar_extracted) : (!SF) -> !affine

  %num_scalar_muls = memref.dim %points, %c0 : !points_M
  scf.for %i = %c0 to %num_scalar_muls step %c1 iter_args(%sum_iter = %first_point) -> (!affine){
    %point = elliptic_curve.double %sum_iter : !affine -> !jacobian
    %point_affine = elliptic_curve.convert_point_type %point : !jacobian -> !affine
    memref.store %point_affine, %points[%i] : !points_M
    scf.yield %point_affine : !affine
  }
  return
}

func.func private @mgpuStreamCreate() -> !llvm.ptr
func.func private @mgpuStreamDestroy(!llvm.ptr)
func.func private @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
func.func private @mgpuMemFree(!llvm.ptr, !llvm.ptr)
func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

func.func private @sortPairsI64I64(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
func.func private @EncodeI64(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
func.func private @ExclusiveSumI64(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

#bits_per_window = 16 : i16

func.func @gpu_msm(%scalars : !SF_M, %points : !points_M) attributes { llvm.emit_c_interface } {
  %scalar_bit_width = arith.constant 256 : i64
  %bits_per_window = arith.constant 16 : i64
  // 2^20 = 1048576
  %num_scalar_muls = arith.constant 1048576 : i64
  // scalar_bit_width / bits_per_window = 256 / 16 = 16
  %num_windows = arith.divsi %scalar_bit_width, %bits_per_window : i64 // 16
  // 2^(bits_per_window) * num_windows = 2^16 * 16 = 1048576
  %num_buckets = arith.constant 1048576 : i64
  // num_windows * num_scalar_muls = 16 * 1048576 = 16777216
  %num_split_scalars = arith.constant 16777216 : i64

  %s = bufferization.to_tensor %scalars restrict writable : !SF_M to !SF_T
  %p = bufferization.to_tensor %points restrict writable : !points_M to !points_T

  /////////////////////////////////
  ///////// scalar decomp /////////
  /////////////////////////////////

  // Takes in scalars tensor of size num_scalar_muls and outputs bucket_indices and point_indices tensors of size num_windows * num_scalar_muls
  // bits_per_window is chosen as desired.
  %bucket_indices, %point_indices = elliptic_curve.scalar_decomp %s {bitsPerWindow = #bits_per_window} : (!SF_T) -> (!split_scalars_T, !split_scalars_T)

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

  // Copy input data from host to device
  func.call @mgpuMemcpy(%d_bucket_indices, %bucket_indices_ptr, %indices_alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemcpy(%d_point_indices, %point_indices_ptr, %indices_alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

  // Sort d_bucket_indices and d_point_indices pairs to d_sorted_bucket_indices and d_sorted_point_indices
  func.call @sortPairsI64I64(%d_bucket_indices, %d_sorted_bucket_indices, %d_point_indices, %d_sorted_point_indices, %num_split_scalars, %stream) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_bucket_indices, %stream) : (!llvm.ptr, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_point_indices, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Move sorted_point_indices from device to host
  %sorted_point_indices = tensor.empty() : !split_scalars_T
  %sorted_point_indices_mem = bufferization.to_buffer %sorted_point_indices : !split_scalars_T to !split_scalars_M
  %sorted_point_indices_ptr_index = memref.extract_aligned_pointer_as_index %sorted_point_indices_mem : !split_scalars_M -> index
  %sorted_point_indices_ptr_i64 = arith.index_cast %sorted_point_indices_ptr_index : index to i64
  %sorted_point_indices_ptr = llvm.inttoptr %sorted_point_indices_ptr_i64 : i64 to !llvm.ptr
  func.call @mgpuMemcpy(%sorted_point_indices_ptr, %d_sorted_point_indices, %indices_alloc_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_sorted_point_indices, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

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

  // Move nof_buckets_to_compute from device to host
  %nof_buckets_to_compute_mem = memref.alloc() : memref<i64>
  %nof_buckets_to_compute_ptr_index = memref.extract_aligned_pointer_as_index %nof_buckets_to_compute_mem : memref<i64> -> index
  %nof_buckets_to_compute_ptr_i64 = arith.index_cast %nof_buckets_to_compute_ptr_index : index to i64
  %nof_buckets_to_compute_ptr = llvm.inttoptr %nof_buckets_to_compute_ptr_i64 : i64 to !llvm.ptr
  func.call @mgpuMemcpy(%nof_buckets_to_compute_ptr, %d_nof_buckets_to_compute, %int_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_nof_buckets_to_compute, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Move sorted_unique_bucket_indices from device to host
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
  func.call @ExclusiveSumI64(%d_bucket_sizes, %d_bucket_offsets, %nof_buckets_to_compute_and_1, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_bucket_sizes, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Move bucket_offsets from device to host
  %bucket_offsets_size = arith.muli %nof_buckets_to_compute_and_1, %int_size : i64
  func.call @mgpuMemcpy(%bucket_offsets_ptr, %d_bucket_offsets, %bucket_offsets_size, %stream) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
  func.call @mgpuMemFree(%d_bucket_offsets, %stream) : (!llvm.ptr, !llvm.ptr) -> ()

  // Clean up CUDA stream
  func.call @mgpuStreamDestroy(%stream) : (!llvm.ptr) -> ()

  ///////////////////////////////////////
  ///////// bucket accumulation /////////
  ///////////////////////////////////////

  %buckets = elliptic_curve.bucket_acc %p, %sorted_point_indices, %sorted_unique_bucket_indices, %bucket_offsets: (!points_T, !split_scalars_T, tensor<?xindex>, tensor<?xindex>) -> !buckets

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

  %res = elliptic_curve.window_reduce %windows {bitsPerWindow = #bits_per_window, scalarType = !SFm}: (!windows) -> !jacobian
  return
}
