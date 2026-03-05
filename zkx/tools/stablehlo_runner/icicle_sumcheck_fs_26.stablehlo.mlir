!pf_babybear_mont = !field.pf<2013265921 : i32, true>
module @jit_combine_sumcheck_prove_fs attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x67108864x!pf_babybear_mont>, %arg1: tensor<!pf_babybear_mont>) -> (tensor<26x4x!pf_babybear_mont> {jax.result_info = "result[0]"}, tensor<4x!pf_babybear_mont> {jax.result_info = "result[1]"}) {
    %cst = stablehlo.constant() <{value = dense<[1073741816, 0]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_0 = stablehlo.constant() <{value = dense<[939524041, 805306362]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_1 = stablehlo.constant() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1x!pf_babybear_mont>
    %cst_2 = stablehlo.constant() <{value = dense<[1073741816, 268435454]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_3 = stablehlo.constant() <{value = dense<[1073741816, 536870908]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_4 = stablehlo.constant() <{value = dense<[1073741816, 805306362]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_5 = stablehlo.constant() <{value = dense<1073741816> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_6 = stablehlo.constant() <{value = dense<[1073741816, 1342177270]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_7 = stablehlo.constant() <{value = dense<[1073741816, 1610612724]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_8 = stablehlo.constant() <{value = dense<[1073741816, 1879048178]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_9 = stablehlo.constant() <{value = dense<[1073741816, 134217711]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_10 = stablehlo.constant() <{value = dense<[1073741816, 402653165]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_11 = stablehlo.constant() <{value = dense<[1073741816, 671088619]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_12 = stablehlo.constant() <{value = dense<[1073741816, 939524073]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_13 = stablehlo.constant() <{value = dense<[1073741816, 1207959527]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_14 = stablehlo.constant() <{value = dense<[1073741816, 1476394981]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_15 = stablehlo.constant() <{value = dense<[1073741816, 1744830435]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_16 = stablehlo.constant() <{value = dense<[1073741816, 2013265889]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_17 = stablehlo.constant() <{value = dense<[1073741816, 268435422]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_18 = stablehlo.constant() <{value = dense<[1073741816, 536870876]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_19 = stablehlo.constant() <{value = dense<[1073741816, 805306330]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_20 = stablehlo.constant() <{value = dense<[1073741816, 1073741784]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_21 = stablehlo.constant() <{value = dense<[1073741816, 1342177238]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_22 = stablehlo.constant() <{value = dense<[1073741816, 1610612692]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_23 = stablehlo.constant() <{value = dense<[1073741816, 1879048146]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_24 = stablehlo.constant() <{value = dense<[1073741816, 134217679]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_25 = stablehlo.constant() <{value = dense<[1073741816, 402653133]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %cst_26 = stablehlo.constant() <{value = dense<[1073741816, 671088587]> : tensor<2xi32>}> : () -> tensor<2x!pf_babybear_mont>
    %0 = stablehlo.iota dim = 0 : tensor<33554432xi32>
    %c = stablehlo.constant dense<2> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<33554432xi32>
    %2 = stablehlo.multiply %1, %0 : tensor<33554432xi32>
    %c_27 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %c_27, dims = [] : (tensor<i32>) -> tensor<33554432xi32>
    %4 = stablehlo.add %3, %2 : tensor<33554432xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<33554432xi32>) -> tensor<33554432x1xi32>
    %6 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x67108864x!pf_babybear_mont>, tensor<33554432x1xi32>) -> tensor<4x33554432x!pf_babybear_mont>
    %7 = stablehlo.iota dim = 0 : tensor<33554432xi32>
    %c_28 = stablehlo.constant dense<2> : tensor<i32>
    %8 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i32>) -> tensor<33554432xi32>
    %9 = stablehlo.multiply %8, %7 : tensor<33554432xi32>
    %c_29 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.broadcast_in_dim %c_29, dims = [] : (tensor<i32>) -> tensor<33554432xi32>
    %11 = stablehlo.add %10, %9 : tensor<33554432xi32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<33554432xi32>) -> tensor<33554432x1xi32>
    %13 = "stablehlo.gather"(%arg0, %12) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x67108864x!pf_babybear_mont>, tensor<33554432x1xi32>) -> tensor<4x33554432x!pf_babybear_mont>
    %14 = stablehlo.subtract %13, %6 : tensor<4x33554432x!pf_babybear_mont>
    %cst_30 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %15 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x33554432x!pf_babybear_mont>
    %16 = stablehlo.multiply %14, %15 : tensor<4x33554432x!pf_babybear_mont>
    %17 = stablehlo.add %16, %6 : tensor<4x33554432x!pf_babybear_mont>
    %18 = stablehlo.slice %17 [3:4, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %19 = stablehlo.reshape %18 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %20 = stablehlo.slice %17 [0:1, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %21 = stablehlo.reshape %20 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %22 = stablehlo.slice %17 [1:2, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %23 = stablehlo.reshape %22 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %24 = stablehlo.multiply %21, %23 : tensor<33554432x!pf_babybear_mont>
    %25 = stablehlo.slice %17 [2:3, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %26 = stablehlo.reshape %25 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %27 = stablehlo.subtract %24, %26 : tensor<33554432x!pf_babybear_mont>
    %28 = stablehlo.multiply %19, %27 : tensor<33554432x!pf_babybear_mont>
    %cst_31 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %29 = stablehlo.reduce(%28 init: %cst_31) applies stablehlo.add across dimensions = [0] : (tensor<33554432x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_32 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %30 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x33554432x!pf_babybear_mont>
    %31 = stablehlo.multiply %14, %30 : tensor<4x33554432x!pf_babybear_mont>
    %32 = stablehlo.add %31, %6 : tensor<4x33554432x!pf_babybear_mont>
    %33 = stablehlo.slice %32 [3:4, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %34 = stablehlo.reshape %33 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %35 = stablehlo.slice %32 [0:1, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %36 = stablehlo.reshape %35 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %37 = stablehlo.slice %32 [1:2, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %38 = stablehlo.reshape %37 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %39 = stablehlo.multiply %36, %38 : tensor<33554432x!pf_babybear_mont>
    %40 = stablehlo.slice %32 [2:3, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %41 = stablehlo.reshape %40 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %42 = stablehlo.subtract %39, %41 : tensor<33554432x!pf_babybear_mont>
    %43 = stablehlo.multiply %34, %42 : tensor<33554432x!pf_babybear_mont>
    %cst_33 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %44 = stablehlo.reduce(%43 init: %cst_33) applies stablehlo.add across dimensions = [0] : (tensor<33554432x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_34 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %45 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x33554432x!pf_babybear_mont>
    %46 = stablehlo.multiply %14, %45 : tensor<4x33554432x!pf_babybear_mont>
    %47 = stablehlo.add %46, %6 : tensor<4x33554432x!pf_babybear_mont>
    %48 = stablehlo.slice %47 [3:4, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %49 = stablehlo.reshape %48 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %50 = stablehlo.slice %47 [0:1, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %51 = stablehlo.reshape %50 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %52 = stablehlo.slice %47 [1:2, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %53 = stablehlo.reshape %52 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %54 = stablehlo.multiply %51, %53 : tensor<33554432x!pf_babybear_mont>
    %55 = stablehlo.slice %47 [2:3, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %56 = stablehlo.reshape %55 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %57 = stablehlo.subtract %54, %56 : tensor<33554432x!pf_babybear_mont>
    %58 = stablehlo.multiply %49, %57 : tensor<33554432x!pf_babybear_mont>
    %cst_35 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %59 = stablehlo.reduce(%58 init: %cst_35) applies stablehlo.add across dimensions = [0] : (tensor<33554432x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_36 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %60 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x33554432x!pf_babybear_mont>
    %61 = stablehlo.multiply %14, %60 : tensor<4x33554432x!pf_babybear_mont>
    %62 = stablehlo.add %61, %6 : tensor<4x33554432x!pf_babybear_mont>
    %63 = stablehlo.slice %62 [3:4, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %64 = stablehlo.reshape %63 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %65 = stablehlo.slice %62 [0:1, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %66 = stablehlo.reshape %65 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %67 = stablehlo.slice %62 [1:2, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %68 = stablehlo.reshape %67 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %69 = stablehlo.multiply %66, %68 : tensor<33554432x!pf_babybear_mont>
    %70 = stablehlo.slice %62 [2:3, 0:33554432] : (tensor<4x33554432x!pf_babybear_mont>) -> tensor<1x33554432x!pf_babybear_mont>
    %71 = stablehlo.reshape %70 : (tensor<1x33554432x!pf_babybear_mont>) -> tensor<33554432x!pf_babybear_mont>
    %72 = stablehlo.subtract %69, %71 : tensor<33554432x!pf_babybear_mont>
    %73 = stablehlo.multiply %64, %72 : tensor<33554432x!pf_babybear_mont>
    %cst_37 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %74 = stablehlo.reduce(%73 init: %cst_37) applies stablehlo.add across dimensions = [0] : (tensor<33554432x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %75 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %76 = stablehlo.broadcast_in_dim %44, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %77 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %78 = stablehlo.broadcast_in_dim %74, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %79 = stablehlo.concatenate %75, %76, %77, %78, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %80 = stablehlo.reshape %arg1 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %81 = stablehlo.concatenate %cst_0, %80, %cst_1, %79, %cst, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>) -> tensor<10x!pf_babybear_mont>
    %cst_38 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %82 = stablehlo.broadcast_in_dim %cst_38, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %83 = stablehlo.slice %81 [0:1] : (tensor<10x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %84 = stablehlo.reshape %83 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_39 = stablehlo.constant dense<0> : tensor<i32>
    %85 = stablehlo.broadcast_in_dim %c_39, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %86 = "stablehlo.scatter"(%82, %85, %84) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_40 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %87 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %88 = stablehlo.slice %81 [1:10] : (tensor<10x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %c_41 = stablehlo.constant dense<0> : tensor<i32>
    %89 = stablehlo.broadcast_in_dim %c_41, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %90 = "stablehlo.scatter"(%87, %89, %88) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<9x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_42 = stablehlo.constant dense<9> : tensor<i32>
    %91 = stablehlo.broadcast_in_dim %c_42, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_43 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %92 = "stablehlo.scatter"(%90, %91, %cst_43) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_44 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %93 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %94 = stablehlo.concatenate %93, %92, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %95 = stablehlo.add %86, %94 : tensor<16x!pf_babybear_mont>
    %96 = call @permute(%95) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %97 = stablehlo.slice %96 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %98 = stablehlo.reshape %97 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %99 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x33554432x!pf_babybear_mont>
    %100 = stablehlo.multiply %14, %99 : tensor<4x33554432x!pf_babybear_mont>
    %101 = stablehlo.add %100, %6 : tensor<4x33554432x!pf_babybear_mont>
    %102 = stablehlo.iota dim = 0 : tensor<16777216xi32>
    %c_45 = stablehlo.constant dense<2> : tensor<i32>
    %103 = stablehlo.broadcast_in_dim %c_45, dims = [] : (tensor<i32>) -> tensor<16777216xi32>
    %104 = stablehlo.multiply %103, %102 : tensor<16777216xi32>
    %c_46 = stablehlo.constant dense<0> : tensor<i32>
    %105 = stablehlo.broadcast_in_dim %c_46, dims = [] : (tensor<i32>) -> tensor<16777216xi32>
    %106 = stablehlo.add %105, %104 : tensor<16777216xi32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0] : (tensor<16777216xi32>) -> tensor<16777216x1xi32>
    %108 = "stablehlo.gather"(%101, %107) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x33554432x!pf_babybear_mont>, tensor<16777216x1xi32>) -> tensor<4x16777216x!pf_babybear_mont>
    %109 = stablehlo.iota dim = 0 : tensor<16777216xi32>
    %c_47 = stablehlo.constant dense<2> : tensor<i32>
    %110 = stablehlo.broadcast_in_dim %c_47, dims = [] : (tensor<i32>) -> tensor<16777216xi32>
    %111 = stablehlo.multiply %110, %109 : tensor<16777216xi32>
    %c_48 = stablehlo.constant dense<1> : tensor<i32>
    %112 = stablehlo.broadcast_in_dim %c_48, dims = [] : (tensor<i32>) -> tensor<16777216xi32>
    %113 = stablehlo.add %112, %111 : tensor<16777216xi32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0] : (tensor<16777216xi32>) -> tensor<16777216x1xi32>
    %115 = "stablehlo.gather"(%101, %114) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x33554432x!pf_babybear_mont>, tensor<16777216x1xi32>) -> tensor<4x16777216x!pf_babybear_mont>
    %116 = stablehlo.subtract %115, %108 : tensor<4x16777216x!pf_babybear_mont>
    %cst_49 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %117 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16777216x!pf_babybear_mont>
    %118 = stablehlo.multiply %116, %117 : tensor<4x16777216x!pf_babybear_mont>
    %119 = stablehlo.add %118, %108 : tensor<4x16777216x!pf_babybear_mont>
    %120 = stablehlo.slice %119 [3:4, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %121 = stablehlo.reshape %120 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %122 = stablehlo.slice %119 [0:1, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %123 = stablehlo.reshape %122 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %124 = stablehlo.slice %119 [1:2, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %125 = stablehlo.reshape %124 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %126 = stablehlo.multiply %123, %125 : tensor<16777216x!pf_babybear_mont>
    %127 = stablehlo.slice %119 [2:3, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %128 = stablehlo.reshape %127 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %129 = stablehlo.subtract %126, %128 : tensor<16777216x!pf_babybear_mont>
    %130 = stablehlo.multiply %121, %129 : tensor<16777216x!pf_babybear_mont>
    %cst_50 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %131 = stablehlo.reduce(%130 init: %cst_50) applies stablehlo.add across dimensions = [0] : (tensor<16777216x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_51 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %132 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16777216x!pf_babybear_mont>
    %133 = stablehlo.multiply %116, %132 : tensor<4x16777216x!pf_babybear_mont>
    %134 = stablehlo.add %133, %108 : tensor<4x16777216x!pf_babybear_mont>
    %135 = stablehlo.slice %134 [3:4, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %136 = stablehlo.reshape %135 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %137 = stablehlo.slice %134 [0:1, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %138 = stablehlo.reshape %137 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %139 = stablehlo.slice %134 [1:2, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %140 = stablehlo.reshape %139 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %141 = stablehlo.multiply %138, %140 : tensor<16777216x!pf_babybear_mont>
    %142 = stablehlo.slice %134 [2:3, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %143 = stablehlo.reshape %142 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %144 = stablehlo.subtract %141, %143 : tensor<16777216x!pf_babybear_mont>
    %145 = stablehlo.multiply %136, %144 : tensor<16777216x!pf_babybear_mont>
    %cst_52 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %146 = stablehlo.reduce(%145 init: %cst_52) applies stablehlo.add across dimensions = [0] : (tensor<16777216x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_53 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %147 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16777216x!pf_babybear_mont>
    %148 = stablehlo.multiply %116, %147 : tensor<4x16777216x!pf_babybear_mont>
    %149 = stablehlo.add %148, %108 : tensor<4x16777216x!pf_babybear_mont>
    %150 = stablehlo.slice %149 [3:4, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %151 = stablehlo.reshape %150 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %152 = stablehlo.slice %149 [0:1, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %153 = stablehlo.reshape %152 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %154 = stablehlo.slice %149 [1:2, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %155 = stablehlo.reshape %154 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %156 = stablehlo.multiply %153, %155 : tensor<16777216x!pf_babybear_mont>
    %157 = stablehlo.slice %149 [2:3, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %158 = stablehlo.reshape %157 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %159 = stablehlo.subtract %156, %158 : tensor<16777216x!pf_babybear_mont>
    %160 = stablehlo.multiply %151, %159 : tensor<16777216x!pf_babybear_mont>
    %cst_54 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %161 = stablehlo.reduce(%160 init: %cst_54) applies stablehlo.add across dimensions = [0] : (tensor<16777216x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_55 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %162 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16777216x!pf_babybear_mont>
    %163 = stablehlo.multiply %116, %162 : tensor<4x16777216x!pf_babybear_mont>
    %164 = stablehlo.add %163, %108 : tensor<4x16777216x!pf_babybear_mont>
    %165 = stablehlo.slice %164 [3:4, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %166 = stablehlo.reshape %165 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %167 = stablehlo.slice %164 [0:1, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %168 = stablehlo.reshape %167 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %169 = stablehlo.slice %164 [1:2, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %170 = stablehlo.reshape %169 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %171 = stablehlo.multiply %168, %170 : tensor<16777216x!pf_babybear_mont>
    %172 = stablehlo.slice %164 [2:3, 0:16777216] : (tensor<4x16777216x!pf_babybear_mont>) -> tensor<1x16777216x!pf_babybear_mont>
    %173 = stablehlo.reshape %172 : (tensor<1x16777216x!pf_babybear_mont>) -> tensor<16777216x!pf_babybear_mont>
    %174 = stablehlo.subtract %171, %173 : tensor<16777216x!pf_babybear_mont>
    %175 = stablehlo.multiply %166, %174 : tensor<16777216x!pf_babybear_mont>
    %cst_56 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %176 = stablehlo.reduce(%175 init: %cst_56) applies stablehlo.add across dimensions = [0] : (tensor<16777216x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %177 = stablehlo.broadcast_in_dim %131, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %178 = stablehlo.broadcast_in_dim %146, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %179 = stablehlo.broadcast_in_dim %161, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %180 = stablehlo.broadcast_in_dim %176, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %181 = stablehlo.concatenate %177, %178, %179, %180, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %182 = stablehlo.reshape %98 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %183 = stablehlo.concatenate %cst, %182, %cst_2, %181, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_57 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %184 = stablehlo.broadcast_in_dim %cst_57, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %185 = stablehlo.slice %183 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %186 = stablehlo.reshape %185 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_58 = stablehlo.constant dense<0> : tensor<i32>
    %187 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %188 = "stablehlo.scatter"(%184, %187, %186) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_59 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %189 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %190 = stablehlo.slice %183 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_60 = stablehlo.constant dense<0> : tensor<i32>
    %191 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %192 = "stablehlo.scatter"(%189, %191, %190) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_61 = stablehlo.constant dense<8> : tensor<i32>
    %193 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_62 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %194 = "stablehlo.scatter"(%192, %193, %cst_62) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_63 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %195 = stablehlo.broadcast_in_dim %cst_63, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %196 = stablehlo.concatenate %195, %194, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %197 = stablehlo.add %188, %196 : tensor<16x!pf_babybear_mont>
    %198 = call @permute(%197) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %199 = stablehlo.slice %198 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %200 = stablehlo.reshape %199 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %201 = stablehlo.broadcast_in_dim %200, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16777216x!pf_babybear_mont>
    %202 = stablehlo.multiply %116, %201 : tensor<4x16777216x!pf_babybear_mont>
    %203 = stablehlo.add %202, %108 : tensor<4x16777216x!pf_babybear_mont>
    %204 = stablehlo.iota dim = 0 : tensor<8388608xi32>
    %c_64 = stablehlo.constant dense<2> : tensor<i32>
    %205 = stablehlo.broadcast_in_dim %c_64, dims = [] : (tensor<i32>) -> tensor<8388608xi32>
    %206 = stablehlo.multiply %205, %204 : tensor<8388608xi32>
    %c_65 = stablehlo.constant dense<0> : tensor<i32>
    %207 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<8388608xi32>
    %208 = stablehlo.add %207, %206 : tensor<8388608xi32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0] : (tensor<8388608xi32>) -> tensor<8388608x1xi32>
    %210 = "stablehlo.gather"(%203, %209) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x16777216x!pf_babybear_mont>, tensor<8388608x1xi32>) -> tensor<4x8388608x!pf_babybear_mont>
    %211 = stablehlo.iota dim = 0 : tensor<8388608xi32>
    %c_66 = stablehlo.constant dense<2> : tensor<i32>
    %212 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<8388608xi32>
    %213 = stablehlo.multiply %212, %211 : tensor<8388608xi32>
    %c_67 = stablehlo.constant dense<1> : tensor<i32>
    %214 = stablehlo.broadcast_in_dim %c_67, dims = [] : (tensor<i32>) -> tensor<8388608xi32>
    %215 = stablehlo.add %214, %213 : tensor<8388608xi32>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0] : (tensor<8388608xi32>) -> tensor<8388608x1xi32>
    %217 = "stablehlo.gather"(%203, %216) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x16777216x!pf_babybear_mont>, tensor<8388608x1xi32>) -> tensor<4x8388608x!pf_babybear_mont>
    %218 = stablehlo.subtract %217, %210 : tensor<4x8388608x!pf_babybear_mont>
    %cst_68 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %219 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8388608x!pf_babybear_mont>
    %220 = stablehlo.multiply %218, %219 : tensor<4x8388608x!pf_babybear_mont>
    %221 = stablehlo.add %220, %210 : tensor<4x8388608x!pf_babybear_mont>
    %222 = stablehlo.slice %221 [3:4, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %223 = stablehlo.reshape %222 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %224 = stablehlo.slice %221 [0:1, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %225 = stablehlo.reshape %224 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %226 = stablehlo.slice %221 [1:2, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %227 = stablehlo.reshape %226 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %228 = stablehlo.multiply %225, %227 : tensor<8388608x!pf_babybear_mont>
    %229 = stablehlo.slice %221 [2:3, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %230 = stablehlo.reshape %229 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %231 = stablehlo.subtract %228, %230 : tensor<8388608x!pf_babybear_mont>
    %232 = stablehlo.multiply %223, %231 : tensor<8388608x!pf_babybear_mont>
    %cst_69 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %233 = stablehlo.reduce(%232 init: %cst_69) applies stablehlo.add across dimensions = [0] : (tensor<8388608x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_70 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %234 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8388608x!pf_babybear_mont>
    %235 = stablehlo.multiply %218, %234 : tensor<4x8388608x!pf_babybear_mont>
    %236 = stablehlo.add %235, %210 : tensor<4x8388608x!pf_babybear_mont>
    %237 = stablehlo.slice %236 [3:4, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %238 = stablehlo.reshape %237 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %239 = stablehlo.slice %236 [0:1, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %240 = stablehlo.reshape %239 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %241 = stablehlo.slice %236 [1:2, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %242 = stablehlo.reshape %241 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %243 = stablehlo.multiply %240, %242 : tensor<8388608x!pf_babybear_mont>
    %244 = stablehlo.slice %236 [2:3, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %245 = stablehlo.reshape %244 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %246 = stablehlo.subtract %243, %245 : tensor<8388608x!pf_babybear_mont>
    %247 = stablehlo.multiply %238, %246 : tensor<8388608x!pf_babybear_mont>
    %cst_71 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %248 = stablehlo.reduce(%247 init: %cst_71) applies stablehlo.add across dimensions = [0] : (tensor<8388608x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_72 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %249 = stablehlo.broadcast_in_dim %cst_72, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8388608x!pf_babybear_mont>
    %250 = stablehlo.multiply %218, %249 : tensor<4x8388608x!pf_babybear_mont>
    %251 = stablehlo.add %250, %210 : tensor<4x8388608x!pf_babybear_mont>
    %252 = stablehlo.slice %251 [3:4, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %253 = stablehlo.reshape %252 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %254 = stablehlo.slice %251 [0:1, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %255 = stablehlo.reshape %254 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %256 = stablehlo.slice %251 [1:2, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %257 = stablehlo.reshape %256 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %258 = stablehlo.multiply %255, %257 : tensor<8388608x!pf_babybear_mont>
    %259 = stablehlo.slice %251 [2:3, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %260 = stablehlo.reshape %259 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %261 = stablehlo.subtract %258, %260 : tensor<8388608x!pf_babybear_mont>
    %262 = stablehlo.multiply %253, %261 : tensor<8388608x!pf_babybear_mont>
    %cst_73 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %263 = stablehlo.reduce(%262 init: %cst_73) applies stablehlo.add across dimensions = [0] : (tensor<8388608x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_74 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %264 = stablehlo.broadcast_in_dim %cst_74, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8388608x!pf_babybear_mont>
    %265 = stablehlo.multiply %218, %264 : tensor<4x8388608x!pf_babybear_mont>
    %266 = stablehlo.add %265, %210 : tensor<4x8388608x!pf_babybear_mont>
    %267 = stablehlo.slice %266 [3:4, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %268 = stablehlo.reshape %267 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %269 = stablehlo.slice %266 [0:1, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %270 = stablehlo.reshape %269 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %271 = stablehlo.slice %266 [1:2, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %272 = stablehlo.reshape %271 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %273 = stablehlo.multiply %270, %272 : tensor<8388608x!pf_babybear_mont>
    %274 = stablehlo.slice %266 [2:3, 0:8388608] : (tensor<4x8388608x!pf_babybear_mont>) -> tensor<1x8388608x!pf_babybear_mont>
    %275 = stablehlo.reshape %274 : (tensor<1x8388608x!pf_babybear_mont>) -> tensor<8388608x!pf_babybear_mont>
    %276 = stablehlo.subtract %273, %275 : tensor<8388608x!pf_babybear_mont>
    %277 = stablehlo.multiply %268, %276 : tensor<8388608x!pf_babybear_mont>
    %cst_75 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %278 = stablehlo.reduce(%277 init: %cst_75) applies stablehlo.add across dimensions = [0] : (tensor<8388608x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %279 = stablehlo.broadcast_in_dim %233, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %280 = stablehlo.broadcast_in_dim %248, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %281 = stablehlo.broadcast_in_dim %263, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %282 = stablehlo.broadcast_in_dim %278, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %283 = stablehlo.concatenate %279, %280, %281, %282, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %284 = stablehlo.reshape %200 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %285 = stablehlo.concatenate %cst, %284, %cst_3, %283, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_76 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %286 = stablehlo.broadcast_in_dim %cst_76, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %287 = stablehlo.slice %285 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %288 = stablehlo.reshape %287 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_77 = stablehlo.constant dense<0> : tensor<i32>
    %289 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %290 = "stablehlo.scatter"(%286, %289, %288) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_78 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %291 = stablehlo.broadcast_in_dim %cst_78, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %292 = stablehlo.slice %285 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_79 = stablehlo.constant dense<0> : tensor<i32>
    %293 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %294 = "stablehlo.scatter"(%291, %293, %292) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_80 = stablehlo.constant dense<8> : tensor<i32>
    %295 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_81 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %296 = "stablehlo.scatter"(%294, %295, %cst_81) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_82 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %297 = stablehlo.broadcast_in_dim %cst_82, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %298 = stablehlo.concatenate %297, %296, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %299 = stablehlo.add %290, %298 : tensor<16x!pf_babybear_mont>
    %300 = call @permute(%299) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %301 = stablehlo.slice %300 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %302 = stablehlo.reshape %301 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %303 = stablehlo.broadcast_in_dim %302, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8388608x!pf_babybear_mont>
    %304 = stablehlo.multiply %218, %303 : tensor<4x8388608x!pf_babybear_mont>
    %305 = stablehlo.add %304, %210 : tensor<4x8388608x!pf_babybear_mont>
    %306 = stablehlo.iota dim = 0 : tensor<4194304xi32>
    %c_83 = stablehlo.constant dense<2> : tensor<i32>
    %307 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<4194304xi32>
    %308 = stablehlo.multiply %307, %306 : tensor<4194304xi32>
    %c_84 = stablehlo.constant dense<0> : tensor<i32>
    %309 = stablehlo.broadcast_in_dim %c_84, dims = [] : (tensor<i32>) -> tensor<4194304xi32>
    %310 = stablehlo.add %309, %308 : tensor<4194304xi32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [0] : (tensor<4194304xi32>) -> tensor<4194304x1xi32>
    %312 = "stablehlo.gather"(%305, %311) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x8388608x!pf_babybear_mont>, tensor<4194304x1xi32>) -> tensor<4x4194304x!pf_babybear_mont>
    %313 = stablehlo.iota dim = 0 : tensor<4194304xi32>
    %c_85 = stablehlo.constant dense<2> : tensor<i32>
    %314 = stablehlo.broadcast_in_dim %c_85, dims = [] : (tensor<i32>) -> tensor<4194304xi32>
    %315 = stablehlo.multiply %314, %313 : tensor<4194304xi32>
    %c_86 = stablehlo.constant dense<1> : tensor<i32>
    %316 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<4194304xi32>
    %317 = stablehlo.add %316, %315 : tensor<4194304xi32>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0] : (tensor<4194304xi32>) -> tensor<4194304x1xi32>
    %319 = "stablehlo.gather"(%305, %318) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x8388608x!pf_babybear_mont>, tensor<4194304x1xi32>) -> tensor<4x4194304x!pf_babybear_mont>
    %320 = stablehlo.subtract %319, %312 : tensor<4x4194304x!pf_babybear_mont>
    %cst_87 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %321 = stablehlo.broadcast_in_dim %cst_87, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4194304x!pf_babybear_mont>
    %322 = stablehlo.multiply %320, %321 : tensor<4x4194304x!pf_babybear_mont>
    %323 = stablehlo.add %322, %312 : tensor<4x4194304x!pf_babybear_mont>
    %324 = stablehlo.slice %323 [3:4, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %325 = stablehlo.reshape %324 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %326 = stablehlo.slice %323 [0:1, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %327 = stablehlo.reshape %326 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %328 = stablehlo.slice %323 [1:2, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %329 = stablehlo.reshape %328 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %330 = stablehlo.multiply %327, %329 : tensor<4194304x!pf_babybear_mont>
    %331 = stablehlo.slice %323 [2:3, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %332 = stablehlo.reshape %331 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %333 = stablehlo.subtract %330, %332 : tensor<4194304x!pf_babybear_mont>
    %334 = stablehlo.multiply %325, %333 : tensor<4194304x!pf_babybear_mont>
    %cst_88 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %335 = stablehlo.reduce(%334 init: %cst_88) applies stablehlo.add across dimensions = [0] : (tensor<4194304x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_89 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %336 = stablehlo.broadcast_in_dim %cst_89, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4194304x!pf_babybear_mont>
    %337 = stablehlo.multiply %320, %336 : tensor<4x4194304x!pf_babybear_mont>
    %338 = stablehlo.add %337, %312 : tensor<4x4194304x!pf_babybear_mont>
    %339 = stablehlo.slice %338 [3:4, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %340 = stablehlo.reshape %339 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %341 = stablehlo.slice %338 [0:1, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %342 = stablehlo.reshape %341 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %343 = stablehlo.slice %338 [1:2, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %344 = stablehlo.reshape %343 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %345 = stablehlo.multiply %342, %344 : tensor<4194304x!pf_babybear_mont>
    %346 = stablehlo.slice %338 [2:3, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %347 = stablehlo.reshape %346 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %348 = stablehlo.subtract %345, %347 : tensor<4194304x!pf_babybear_mont>
    %349 = stablehlo.multiply %340, %348 : tensor<4194304x!pf_babybear_mont>
    %cst_90 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %350 = stablehlo.reduce(%349 init: %cst_90) applies stablehlo.add across dimensions = [0] : (tensor<4194304x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_91 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %351 = stablehlo.broadcast_in_dim %cst_91, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4194304x!pf_babybear_mont>
    %352 = stablehlo.multiply %320, %351 : tensor<4x4194304x!pf_babybear_mont>
    %353 = stablehlo.add %352, %312 : tensor<4x4194304x!pf_babybear_mont>
    %354 = stablehlo.slice %353 [3:4, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %355 = stablehlo.reshape %354 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %356 = stablehlo.slice %353 [0:1, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %357 = stablehlo.reshape %356 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %358 = stablehlo.slice %353 [1:2, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %359 = stablehlo.reshape %358 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %360 = stablehlo.multiply %357, %359 : tensor<4194304x!pf_babybear_mont>
    %361 = stablehlo.slice %353 [2:3, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %362 = stablehlo.reshape %361 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %363 = stablehlo.subtract %360, %362 : tensor<4194304x!pf_babybear_mont>
    %364 = stablehlo.multiply %355, %363 : tensor<4194304x!pf_babybear_mont>
    %cst_92 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %365 = stablehlo.reduce(%364 init: %cst_92) applies stablehlo.add across dimensions = [0] : (tensor<4194304x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_93 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %366 = stablehlo.broadcast_in_dim %cst_93, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4194304x!pf_babybear_mont>
    %367 = stablehlo.multiply %320, %366 : tensor<4x4194304x!pf_babybear_mont>
    %368 = stablehlo.add %367, %312 : tensor<4x4194304x!pf_babybear_mont>
    %369 = stablehlo.slice %368 [3:4, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %370 = stablehlo.reshape %369 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %371 = stablehlo.slice %368 [0:1, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %372 = stablehlo.reshape %371 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %373 = stablehlo.slice %368 [1:2, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %374 = stablehlo.reshape %373 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %375 = stablehlo.multiply %372, %374 : tensor<4194304x!pf_babybear_mont>
    %376 = stablehlo.slice %368 [2:3, 0:4194304] : (tensor<4x4194304x!pf_babybear_mont>) -> tensor<1x4194304x!pf_babybear_mont>
    %377 = stablehlo.reshape %376 : (tensor<1x4194304x!pf_babybear_mont>) -> tensor<4194304x!pf_babybear_mont>
    %378 = stablehlo.subtract %375, %377 : tensor<4194304x!pf_babybear_mont>
    %379 = stablehlo.multiply %370, %378 : tensor<4194304x!pf_babybear_mont>
    %cst_94 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %380 = stablehlo.reduce(%379 init: %cst_94) applies stablehlo.add across dimensions = [0] : (tensor<4194304x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %381 = stablehlo.broadcast_in_dim %335, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %382 = stablehlo.broadcast_in_dim %350, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %383 = stablehlo.broadcast_in_dim %365, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %384 = stablehlo.broadcast_in_dim %380, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %385 = stablehlo.concatenate %381, %382, %383, %384, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %386 = stablehlo.reshape %302 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %387 = stablehlo.concatenate %cst, %386, %cst_4, %385, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_95 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %388 = stablehlo.broadcast_in_dim %cst_95, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %389 = stablehlo.slice %387 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %390 = stablehlo.reshape %389 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_96 = stablehlo.constant dense<0> : tensor<i32>
    %391 = stablehlo.broadcast_in_dim %c_96, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %392 = "stablehlo.scatter"(%388, %391, %390) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_97 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %393 = stablehlo.broadcast_in_dim %cst_97, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %394 = stablehlo.slice %387 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_98 = stablehlo.constant dense<0> : tensor<i32>
    %395 = stablehlo.broadcast_in_dim %c_98, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %396 = "stablehlo.scatter"(%393, %395, %394) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_99 = stablehlo.constant dense<8> : tensor<i32>
    %397 = stablehlo.broadcast_in_dim %c_99, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_100 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %398 = "stablehlo.scatter"(%396, %397, %cst_100) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_101 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %399 = stablehlo.broadcast_in_dim %cst_101, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %400 = stablehlo.concatenate %399, %398, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %401 = stablehlo.add %392, %400 : tensor<16x!pf_babybear_mont>
    %402 = call @permute(%401) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %403 = stablehlo.slice %402 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %404 = stablehlo.reshape %403 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %405 = stablehlo.broadcast_in_dim %404, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4194304x!pf_babybear_mont>
    %406 = stablehlo.multiply %320, %405 : tensor<4x4194304x!pf_babybear_mont>
    %407 = stablehlo.add %406, %312 : tensor<4x4194304x!pf_babybear_mont>
    %408 = stablehlo.iota dim = 0 : tensor<2097152xi32>
    %c_102 = stablehlo.constant dense<2> : tensor<i32>
    %409 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i32>) -> tensor<2097152xi32>
    %410 = stablehlo.multiply %409, %408 : tensor<2097152xi32>
    %c_103 = stablehlo.constant dense<0> : tensor<i32>
    %411 = stablehlo.broadcast_in_dim %c_103, dims = [] : (tensor<i32>) -> tensor<2097152xi32>
    %412 = stablehlo.add %411, %410 : tensor<2097152xi32>
    %413 = stablehlo.broadcast_in_dim %412, dims = [0] : (tensor<2097152xi32>) -> tensor<2097152x1xi32>
    %414 = "stablehlo.gather"(%407, %413) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x4194304x!pf_babybear_mont>, tensor<2097152x1xi32>) -> tensor<4x2097152x!pf_babybear_mont>
    %415 = stablehlo.iota dim = 0 : tensor<2097152xi32>
    %c_104 = stablehlo.constant dense<2> : tensor<i32>
    %416 = stablehlo.broadcast_in_dim %c_104, dims = [] : (tensor<i32>) -> tensor<2097152xi32>
    %417 = stablehlo.multiply %416, %415 : tensor<2097152xi32>
    %c_105 = stablehlo.constant dense<1> : tensor<i32>
    %418 = stablehlo.broadcast_in_dim %c_105, dims = [] : (tensor<i32>) -> tensor<2097152xi32>
    %419 = stablehlo.add %418, %417 : tensor<2097152xi32>
    %420 = stablehlo.broadcast_in_dim %419, dims = [0] : (tensor<2097152xi32>) -> tensor<2097152x1xi32>
    %421 = "stablehlo.gather"(%407, %420) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x4194304x!pf_babybear_mont>, tensor<2097152x1xi32>) -> tensor<4x2097152x!pf_babybear_mont>
    %422 = stablehlo.subtract %421, %414 : tensor<4x2097152x!pf_babybear_mont>
    %cst_106 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %423 = stablehlo.broadcast_in_dim %cst_106, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2097152x!pf_babybear_mont>
    %424 = stablehlo.multiply %422, %423 : tensor<4x2097152x!pf_babybear_mont>
    %425 = stablehlo.add %424, %414 : tensor<4x2097152x!pf_babybear_mont>
    %426 = stablehlo.slice %425 [3:4, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %427 = stablehlo.reshape %426 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %428 = stablehlo.slice %425 [0:1, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %429 = stablehlo.reshape %428 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %430 = stablehlo.slice %425 [1:2, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %431 = stablehlo.reshape %430 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %432 = stablehlo.multiply %429, %431 : tensor<2097152x!pf_babybear_mont>
    %433 = stablehlo.slice %425 [2:3, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %434 = stablehlo.reshape %433 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %435 = stablehlo.subtract %432, %434 : tensor<2097152x!pf_babybear_mont>
    %436 = stablehlo.multiply %427, %435 : tensor<2097152x!pf_babybear_mont>
    %cst_107 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %437 = stablehlo.reduce(%436 init: %cst_107) applies stablehlo.add across dimensions = [0] : (tensor<2097152x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_108 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %438 = stablehlo.broadcast_in_dim %cst_108, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2097152x!pf_babybear_mont>
    %439 = stablehlo.multiply %422, %438 : tensor<4x2097152x!pf_babybear_mont>
    %440 = stablehlo.add %439, %414 : tensor<4x2097152x!pf_babybear_mont>
    %441 = stablehlo.slice %440 [3:4, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %442 = stablehlo.reshape %441 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %443 = stablehlo.slice %440 [0:1, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %444 = stablehlo.reshape %443 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %445 = stablehlo.slice %440 [1:2, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %446 = stablehlo.reshape %445 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %447 = stablehlo.multiply %444, %446 : tensor<2097152x!pf_babybear_mont>
    %448 = stablehlo.slice %440 [2:3, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %449 = stablehlo.reshape %448 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %450 = stablehlo.subtract %447, %449 : tensor<2097152x!pf_babybear_mont>
    %451 = stablehlo.multiply %442, %450 : tensor<2097152x!pf_babybear_mont>
    %cst_109 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %452 = stablehlo.reduce(%451 init: %cst_109) applies stablehlo.add across dimensions = [0] : (tensor<2097152x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_110 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %453 = stablehlo.broadcast_in_dim %cst_110, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2097152x!pf_babybear_mont>
    %454 = stablehlo.multiply %422, %453 : tensor<4x2097152x!pf_babybear_mont>
    %455 = stablehlo.add %454, %414 : tensor<4x2097152x!pf_babybear_mont>
    %456 = stablehlo.slice %455 [3:4, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %457 = stablehlo.reshape %456 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %458 = stablehlo.slice %455 [0:1, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %459 = stablehlo.reshape %458 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %460 = stablehlo.slice %455 [1:2, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %461 = stablehlo.reshape %460 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %462 = stablehlo.multiply %459, %461 : tensor<2097152x!pf_babybear_mont>
    %463 = stablehlo.slice %455 [2:3, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %464 = stablehlo.reshape %463 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %465 = stablehlo.subtract %462, %464 : tensor<2097152x!pf_babybear_mont>
    %466 = stablehlo.multiply %457, %465 : tensor<2097152x!pf_babybear_mont>
    %cst_111 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %467 = stablehlo.reduce(%466 init: %cst_111) applies stablehlo.add across dimensions = [0] : (tensor<2097152x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_112 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %468 = stablehlo.broadcast_in_dim %cst_112, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2097152x!pf_babybear_mont>
    %469 = stablehlo.multiply %422, %468 : tensor<4x2097152x!pf_babybear_mont>
    %470 = stablehlo.add %469, %414 : tensor<4x2097152x!pf_babybear_mont>
    %471 = stablehlo.slice %470 [3:4, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %472 = stablehlo.reshape %471 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %473 = stablehlo.slice %470 [0:1, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %474 = stablehlo.reshape %473 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %475 = stablehlo.slice %470 [1:2, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %476 = stablehlo.reshape %475 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %477 = stablehlo.multiply %474, %476 : tensor<2097152x!pf_babybear_mont>
    %478 = stablehlo.slice %470 [2:3, 0:2097152] : (tensor<4x2097152x!pf_babybear_mont>) -> tensor<1x2097152x!pf_babybear_mont>
    %479 = stablehlo.reshape %478 : (tensor<1x2097152x!pf_babybear_mont>) -> tensor<2097152x!pf_babybear_mont>
    %480 = stablehlo.subtract %477, %479 : tensor<2097152x!pf_babybear_mont>
    %481 = stablehlo.multiply %472, %480 : tensor<2097152x!pf_babybear_mont>
    %cst_113 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %482 = stablehlo.reduce(%481 init: %cst_113) applies stablehlo.add across dimensions = [0] : (tensor<2097152x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %483 = stablehlo.broadcast_in_dim %437, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %484 = stablehlo.broadcast_in_dim %452, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %485 = stablehlo.broadcast_in_dim %467, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %486 = stablehlo.broadcast_in_dim %482, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %487 = stablehlo.concatenate %483, %484, %485, %486, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %488 = stablehlo.reshape %404 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %489 = stablehlo.concatenate %cst, %488, %cst_5, %487, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_114 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %490 = stablehlo.broadcast_in_dim %cst_114, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %491 = stablehlo.slice %489 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %492 = stablehlo.reshape %491 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_115 = stablehlo.constant dense<0> : tensor<i32>
    %493 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %494 = "stablehlo.scatter"(%490, %493, %492) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_116 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %495 = stablehlo.broadcast_in_dim %cst_116, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %496 = stablehlo.slice %489 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_117 = stablehlo.constant dense<0> : tensor<i32>
    %497 = stablehlo.broadcast_in_dim %c_117, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %498 = "stablehlo.scatter"(%495, %497, %496) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_118 = stablehlo.constant dense<8> : tensor<i32>
    %499 = stablehlo.broadcast_in_dim %c_118, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_119 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %500 = "stablehlo.scatter"(%498, %499, %cst_119) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_120 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %501 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %502 = stablehlo.concatenate %501, %500, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %503 = stablehlo.add %494, %502 : tensor<16x!pf_babybear_mont>
    %504 = call @permute(%503) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %505 = stablehlo.slice %504 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %506 = stablehlo.reshape %505 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %507 = stablehlo.broadcast_in_dim %506, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2097152x!pf_babybear_mont>
    %508 = stablehlo.multiply %422, %507 : tensor<4x2097152x!pf_babybear_mont>
    %509 = stablehlo.add %508, %414 : tensor<4x2097152x!pf_babybear_mont>
    %510 = stablehlo.iota dim = 0 : tensor<1048576xi32>
    %c_121 = stablehlo.constant dense<2> : tensor<i32>
    %511 = stablehlo.broadcast_in_dim %c_121, dims = [] : (tensor<i32>) -> tensor<1048576xi32>
    %512 = stablehlo.multiply %511, %510 : tensor<1048576xi32>
    %c_122 = stablehlo.constant dense<0> : tensor<i32>
    %513 = stablehlo.broadcast_in_dim %c_122, dims = [] : (tensor<i32>) -> tensor<1048576xi32>
    %514 = stablehlo.add %513, %512 : tensor<1048576xi32>
    %515 = stablehlo.broadcast_in_dim %514, dims = [0] : (tensor<1048576xi32>) -> tensor<1048576x1xi32>
    %516 = "stablehlo.gather"(%509, %515) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x2097152x!pf_babybear_mont>, tensor<1048576x1xi32>) -> tensor<4x1048576x!pf_babybear_mont>
    %517 = stablehlo.iota dim = 0 : tensor<1048576xi32>
    %c_123 = stablehlo.constant dense<2> : tensor<i32>
    %518 = stablehlo.broadcast_in_dim %c_123, dims = [] : (tensor<i32>) -> tensor<1048576xi32>
    %519 = stablehlo.multiply %518, %517 : tensor<1048576xi32>
    %c_124 = stablehlo.constant dense<1> : tensor<i32>
    %520 = stablehlo.broadcast_in_dim %c_124, dims = [] : (tensor<i32>) -> tensor<1048576xi32>
    %521 = stablehlo.add %520, %519 : tensor<1048576xi32>
    %522 = stablehlo.broadcast_in_dim %521, dims = [0] : (tensor<1048576xi32>) -> tensor<1048576x1xi32>
    %523 = "stablehlo.gather"(%509, %522) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x2097152x!pf_babybear_mont>, tensor<1048576x1xi32>) -> tensor<4x1048576x!pf_babybear_mont>
    %524 = stablehlo.subtract %523, %516 : tensor<4x1048576x!pf_babybear_mont>
    %cst_125 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %525 = stablehlo.broadcast_in_dim %cst_125, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1048576x!pf_babybear_mont>
    %526 = stablehlo.multiply %524, %525 : tensor<4x1048576x!pf_babybear_mont>
    %527 = stablehlo.add %526, %516 : tensor<4x1048576x!pf_babybear_mont>
    %528 = stablehlo.slice %527 [3:4, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %529 = stablehlo.reshape %528 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %530 = stablehlo.slice %527 [0:1, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %531 = stablehlo.reshape %530 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %532 = stablehlo.slice %527 [1:2, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %533 = stablehlo.reshape %532 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %534 = stablehlo.multiply %531, %533 : tensor<1048576x!pf_babybear_mont>
    %535 = stablehlo.slice %527 [2:3, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %536 = stablehlo.reshape %535 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %537 = stablehlo.subtract %534, %536 : tensor<1048576x!pf_babybear_mont>
    %538 = stablehlo.multiply %529, %537 : tensor<1048576x!pf_babybear_mont>
    %cst_126 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %539 = stablehlo.reduce(%538 init: %cst_126) applies stablehlo.add across dimensions = [0] : (tensor<1048576x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_127 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %540 = stablehlo.broadcast_in_dim %cst_127, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1048576x!pf_babybear_mont>
    %541 = stablehlo.multiply %524, %540 : tensor<4x1048576x!pf_babybear_mont>
    %542 = stablehlo.add %541, %516 : tensor<4x1048576x!pf_babybear_mont>
    %543 = stablehlo.slice %542 [3:4, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %544 = stablehlo.reshape %543 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %545 = stablehlo.slice %542 [0:1, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %546 = stablehlo.reshape %545 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %547 = stablehlo.slice %542 [1:2, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %548 = stablehlo.reshape %547 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %549 = stablehlo.multiply %546, %548 : tensor<1048576x!pf_babybear_mont>
    %550 = stablehlo.slice %542 [2:3, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %551 = stablehlo.reshape %550 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %552 = stablehlo.subtract %549, %551 : tensor<1048576x!pf_babybear_mont>
    %553 = stablehlo.multiply %544, %552 : tensor<1048576x!pf_babybear_mont>
    %cst_128 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %554 = stablehlo.reduce(%553 init: %cst_128) applies stablehlo.add across dimensions = [0] : (tensor<1048576x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_129 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %555 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1048576x!pf_babybear_mont>
    %556 = stablehlo.multiply %524, %555 : tensor<4x1048576x!pf_babybear_mont>
    %557 = stablehlo.add %556, %516 : tensor<4x1048576x!pf_babybear_mont>
    %558 = stablehlo.slice %557 [3:4, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %559 = stablehlo.reshape %558 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %560 = stablehlo.slice %557 [0:1, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %561 = stablehlo.reshape %560 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %562 = stablehlo.slice %557 [1:2, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %563 = stablehlo.reshape %562 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %564 = stablehlo.multiply %561, %563 : tensor<1048576x!pf_babybear_mont>
    %565 = stablehlo.slice %557 [2:3, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %566 = stablehlo.reshape %565 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %567 = stablehlo.subtract %564, %566 : tensor<1048576x!pf_babybear_mont>
    %568 = stablehlo.multiply %559, %567 : tensor<1048576x!pf_babybear_mont>
    %cst_130 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %569 = stablehlo.reduce(%568 init: %cst_130) applies stablehlo.add across dimensions = [0] : (tensor<1048576x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_131 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %570 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1048576x!pf_babybear_mont>
    %571 = stablehlo.multiply %524, %570 : tensor<4x1048576x!pf_babybear_mont>
    %572 = stablehlo.add %571, %516 : tensor<4x1048576x!pf_babybear_mont>
    %573 = stablehlo.slice %572 [3:4, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %574 = stablehlo.reshape %573 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %575 = stablehlo.slice %572 [0:1, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %576 = stablehlo.reshape %575 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %577 = stablehlo.slice %572 [1:2, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %578 = stablehlo.reshape %577 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %579 = stablehlo.multiply %576, %578 : tensor<1048576x!pf_babybear_mont>
    %580 = stablehlo.slice %572 [2:3, 0:1048576] : (tensor<4x1048576x!pf_babybear_mont>) -> tensor<1x1048576x!pf_babybear_mont>
    %581 = stablehlo.reshape %580 : (tensor<1x1048576x!pf_babybear_mont>) -> tensor<1048576x!pf_babybear_mont>
    %582 = stablehlo.subtract %579, %581 : tensor<1048576x!pf_babybear_mont>
    %583 = stablehlo.multiply %574, %582 : tensor<1048576x!pf_babybear_mont>
    %cst_132 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %584 = stablehlo.reduce(%583 init: %cst_132) applies stablehlo.add across dimensions = [0] : (tensor<1048576x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %585 = stablehlo.broadcast_in_dim %539, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %586 = stablehlo.broadcast_in_dim %554, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %587 = stablehlo.broadcast_in_dim %569, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %588 = stablehlo.broadcast_in_dim %584, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %589 = stablehlo.concatenate %585, %586, %587, %588, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %590 = stablehlo.reshape %506 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %591 = stablehlo.concatenate %cst, %590, %cst_6, %589, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_133 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %592 = stablehlo.broadcast_in_dim %cst_133, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %593 = stablehlo.slice %591 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %594 = stablehlo.reshape %593 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_134 = stablehlo.constant dense<0> : tensor<i32>
    %595 = stablehlo.broadcast_in_dim %c_134, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %596 = "stablehlo.scatter"(%592, %595, %594) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_135 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %597 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %598 = stablehlo.slice %591 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_136 = stablehlo.constant dense<0> : tensor<i32>
    %599 = stablehlo.broadcast_in_dim %c_136, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %600 = "stablehlo.scatter"(%597, %599, %598) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_137 = stablehlo.constant dense<8> : tensor<i32>
    %601 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_138 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %602 = "stablehlo.scatter"(%600, %601, %cst_138) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_139 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %603 = stablehlo.broadcast_in_dim %cst_139, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %604 = stablehlo.concatenate %603, %602, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %605 = stablehlo.add %596, %604 : tensor<16x!pf_babybear_mont>
    %606 = call @permute(%605) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %607 = stablehlo.slice %606 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %608 = stablehlo.reshape %607 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %609 = stablehlo.broadcast_in_dim %608, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1048576x!pf_babybear_mont>
    %610 = stablehlo.multiply %524, %609 : tensor<4x1048576x!pf_babybear_mont>
    %611 = stablehlo.add %610, %516 : tensor<4x1048576x!pf_babybear_mont>
    %612 = stablehlo.iota dim = 0 : tensor<524288xi32>
    %c_140 = stablehlo.constant dense<2> : tensor<i32>
    %613 = stablehlo.broadcast_in_dim %c_140, dims = [] : (tensor<i32>) -> tensor<524288xi32>
    %614 = stablehlo.multiply %613, %612 : tensor<524288xi32>
    %c_141 = stablehlo.constant dense<0> : tensor<i32>
    %615 = stablehlo.broadcast_in_dim %c_141, dims = [] : (tensor<i32>) -> tensor<524288xi32>
    %616 = stablehlo.add %615, %614 : tensor<524288xi32>
    %617 = stablehlo.broadcast_in_dim %616, dims = [0] : (tensor<524288xi32>) -> tensor<524288x1xi32>
    %618 = "stablehlo.gather"(%611, %617) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x1048576x!pf_babybear_mont>, tensor<524288x1xi32>) -> tensor<4x524288x!pf_babybear_mont>
    %619 = stablehlo.iota dim = 0 : tensor<524288xi32>
    %c_142 = stablehlo.constant dense<2> : tensor<i32>
    %620 = stablehlo.broadcast_in_dim %c_142, dims = [] : (tensor<i32>) -> tensor<524288xi32>
    %621 = stablehlo.multiply %620, %619 : tensor<524288xi32>
    %c_143 = stablehlo.constant dense<1> : tensor<i32>
    %622 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<524288xi32>
    %623 = stablehlo.add %622, %621 : tensor<524288xi32>
    %624 = stablehlo.broadcast_in_dim %623, dims = [0] : (tensor<524288xi32>) -> tensor<524288x1xi32>
    %625 = "stablehlo.gather"(%611, %624) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x1048576x!pf_babybear_mont>, tensor<524288x1xi32>) -> tensor<4x524288x!pf_babybear_mont>
    %626 = stablehlo.subtract %625, %618 : tensor<4x524288x!pf_babybear_mont>
    %cst_144 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %627 = stablehlo.broadcast_in_dim %cst_144, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x524288x!pf_babybear_mont>
    %628 = stablehlo.multiply %626, %627 : tensor<4x524288x!pf_babybear_mont>
    %629 = stablehlo.add %628, %618 : tensor<4x524288x!pf_babybear_mont>
    %630 = stablehlo.slice %629 [3:4, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %631 = stablehlo.reshape %630 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %632 = stablehlo.slice %629 [0:1, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %633 = stablehlo.reshape %632 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %634 = stablehlo.slice %629 [1:2, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %635 = stablehlo.reshape %634 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %636 = stablehlo.multiply %633, %635 : tensor<524288x!pf_babybear_mont>
    %637 = stablehlo.slice %629 [2:3, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %638 = stablehlo.reshape %637 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %639 = stablehlo.subtract %636, %638 : tensor<524288x!pf_babybear_mont>
    %640 = stablehlo.multiply %631, %639 : tensor<524288x!pf_babybear_mont>
    %cst_145 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %641 = stablehlo.reduce(%640 init: %cst_145) applies stablehlo.add across dimensions = [0] : (tensor<524288x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_146 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %642 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x524288x!pf_babybear_mont>
    %643 = stablehlo.multiply %626, %642 : tensor<4x524288x!pf_babybear_mont>
    %644 = stablehlo.add %643, %618 : tensor<4x524288x!pf_babybear_mont>
    %645 = stablehlo.slice %644 [3:4, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %646 = stablehlo.reshape %645 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %647 = stablehlo.slice %644 [0:1, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %648 = stablehlo.reshape %647 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %649 = stablehlo.slice %644 [1:2, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %650 = stablehlo.reshape %649 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %651 = stablehlo.multiply %648, %650 : tensor<524288x!pf_babybear_mont>
    %652 = stablehlo.slice %644 [2:3, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %653 = stablehlo.reshape %652 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %654 = stablehlo.subtract %651, %653 : tensor<524288x!pf_babybear_mont>
    %655 = stablehlo.multiply %646, %654 : tensor<524288x!pf_babybear_mont>
    %cst_147 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %656 = stablehlo.reduce(%655 init: %cst_147) applies stablehlo.add across dimensions = [0] : (tensor<524288x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_148 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %657 = stablehlo.broadcast_in_dim %cst_148, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x524288x!pf_babybear_mont>
    %658 = stablehlo.multiply %626, %657 : tensor<4x524288x!pf_babybear_mont>
    %659 = stablehlo.add %658, %618 : tensor<4x524288x!pf_babybear_mont>
    %660 = stablehlo.slice %659 [3:4, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %661 = stablehlo.reshape %660 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %662 = stablehlo.slice %659 [0:1, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %663 = stablehlo.reshape %662 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %664 = stablehlo.slice %659 [1:2, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %665 = stablehlo.reshape %664 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %666 = stablehlo.multiply %663, %665 : tensor<524288x!pf_babybear_mont>
    %667 = stablehlo.slice %659 [2:3, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %668 = stablehlo.reshape %667 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %669 = stablehlo.subtract %666, %668 : tensor<524288x!pf_babybear_mont>
    %670 = stablehlo.multiply %661, %669 : tensor<524288x!pf_babybear_mont>
    %cst_149 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %671 = stablehlo.reduce(%670 init: %cst_149) applies stablehlo.add across dimensions = [0] : (tensor<524288x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_150 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %672 = stablehlo.broadcast_in_dim %cst_150, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x524288x!pf_babybear_mont>
    %673 = stablehlo.multiply %626, %672 : tensor<4x524288x!pf_babybear_mont>
    %674 = stablehlo.add %673, %618 : tensor<4x524288x!pf_babybear_mont>
    %675 = stablehlo.slice %674 [3:4, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %676 = stablehlo.reshape %675 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %677 = stablehlo.slice %674 [0:1, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %678 = stablehlo.reshape %677 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %679 = stablehlo.slice %674 [1:2, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %680 = stablehlo.reshape %679 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %681 = stablehlo.multiply %678, %680 : tensor<524288x!pf_babybear_mont>
    %682 = stablehlo.slice %674 [2:3, 0:524288] : (tensor<4x524288x!pf_babybear_mont>) -> tensor<1x524288x!pf_babybear_mont>
    %683 = stablehlo.reshape %682 : (tensor<1x524288x!pf_babybear_mont>) -> tensor<524288x!pf_babybear_mont>
    %684 = stablehlo.subtract %681, %683 : tensor<524288x!pf_babybear_mont>
    %685 = stablehlo.multiply %676, %684 : tensor<524288x!pf_babybear_mont>
    %cst_151 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %686 = stablehlo.reduce(%685 init: %cst_151) applies stablehlo.add across dimensions = [0] : (tensor<524288x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %687 = stablehlo.broadcast_in_dim %641, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %688 = stablehlo.broadcast_in_dim %656, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %689 = stablehlo.broadcast_in_dim %671, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %690 = stablehlo.broadcast_in_dim %686, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %691 = stablehlo.concatenate %687, %688, %689, %690, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %692 = stablehlo.reshape %608 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %693 = stablehlo.concatenate %cst, %692, %cst_7, %691, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_152 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %694 = stablehlo.broadcast_in_dim %cst_152, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %695 = stablehlo.slice %693 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %696 = stablehlo.reshape %695 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_153 = stablehlo.constant dense<0> : tensor<i32>
    %697 = stablehlo.broadcast_in_dim %c_153, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %698 = "stablehlo.scatter"(%694, %697, %696) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_154 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %699 = stablehlo.broadcast_in_dim %cst_154, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %700 = stablehlo.slice %693 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_155 = stablehlo.constant dense<0> : tensor<i32>
    %701 = stablehlo.broadcast_in_dim %c_155, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %702 = "stablehlo.scatter"(%699, %701, %700) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_156 = stablehlo.constant dense<8> : tensor<i32>
    %703 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_157 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %704 = "stablehlo.scatter"(%702, %703, %cst_157) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_158 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %705 = stablehlo.broadcast_in_dim %cst_158, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %706 = stablehlo.concatenate %705, %704, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %707 = stablehlo.add %698, %706 : tensor<16x!pf_babybear_mont>
    %708 = call @permute(%707) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %709 = stablehlo.slice %708 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %710 = stablehlo.reshape %709 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %711 = stablehlo.broadcast_in_dim %710, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x524288x!pf_babybear_mont>
    %712 = stablehlo.multiply %626, %711 : tensor<4x524288x!pf_babybear_mont>
    %713 = stablehlo.add %712, %618 : tensor<4x524288x!pf_babybear_mont>
    %714 = stablehlo.iota dim = 0 : tensor<262144xi32>
    %c_159 = stablehlo.constant dense<2> : tensor<i32>
    %715 = stablehlo.broadcast_in_dim %c_159, dims = [] : (tensor<i32>) -> tensor<262144xi32>
    %716 = stablehlo.multiply %715, %714 : tensor<262144xi32>
    %c_160 = stablehlo.constant dense<0> : tensor<i32>
    %717 = stablehlo.broadcast_in_dim %c_160, dims = [] : (tensor<i32>) -> tensor<262144xi32>
    %718 = stablehlo.add %717, %716 : tensor<262144xi32>
    %719 = stablehlo.broadcast_in_dim %718, dims = [0] : (tensor<262144xi32>) -> tensor<262144x1xi32>
    %720 = "stablehlo.gather"(%713, %719) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x524288x!pf_babybear_mont>, tensor<262144x1xi32>) -> tensor<4x262144x!pf_babybear_mont>
    %721 = stablehlo.iota dim = 0 : tensor<262144xi32>
    %c_161 = stablehlo.constant dense<2> : tensor<i32>
    %722 = stablehlo.broadcast_in_dim %c_161, dims = [] : (tensor<i32>) -> tensor<262144xi32>
    %723 = stablehlo.multiply %722, %721 : tensor<262144xi32>
    %c_162 = stablehlo.constant dense<1> : tensor<i32>
    %724 = stablehlo.broadcast_in_dim %c_162, dims = [] : (tensor<i32>) -> tensor<262144xi32>
    %725 = stablehlo.add %724, %723 : tensor<262144xi32>
    %726 = stablehlo.broadcast_in_dim %725, dims = [0] : (tensor<262144xi32>) -> tensor<262144x1xi32>
    %727 = "stablehlo.gather"(%713, %726) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x524288x!pf_babybear_mont>, tensor<262144x1xi32>) -> tensor<4x262144x!pf_babybear_mont>
    %728 = stablehlo.subtract %727, %720 : tensor<4x262144x!pf_babybear_mont>
    %cst_163 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %729 = stablehlo.broadcast_in_dim %cst_163, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x262144x!pf_babybear_mont>
    %730 = stablehlo.multiply %728, %729 : tensor<4x262144x!pf_babybear_mont>
    %731 = stablehlo.add %730, %720 : tensor<4x262144x!pf_babybear_mont>
    %732 = stablehlo.slice %731 [3:4, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %733 = stablehlo.reshape %732 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %734 = stablehlo.slice %731 [0:1, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %735 = stablehlo.reshape %734 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %736 = stablehlo.slice %731 [1:2, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %737 = stablehlo.reshape %736 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %738 = stablehlo.multiply %735, %737 : tensor<262144x!pf_babybear_mont>
    %739 = stablehlo.slice %731 [2:3, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %740 = stablehlo.reshape %739 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %741 = stablehlo.subtract %738, %740 : tensor<262144x!pf_babybear_mont>
    %742 = stablehlo.multiply %733, %741 : tensor<262144x!pf_babybear_mont>
    %cst_164 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %743 = stablehlo.reduce(%742 init: %cst_164) applies stablehlo.add across dimensions = [0] : (tensor<262144x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_165 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %744 = stablehlo.broadcast_in_dim %cst_165, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x262144x!pf_babybear_mont>
    %745 = stablehlo.multiply %728, %744 : tensor<4x262144x!pf_babybear_mont>
    %746 = stablehlo.add %745, %720 : tensor<4x262144x!pf_babybear_mont>
    %747 = stablehlo.slice %746 [3:4, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %748 = stablehlo.reshape %747 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %749 = stablehlo.slice %746 [0:1, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %750 = stablehlo.reshape %749 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %751 = stablehlo.slice %746 [1:2, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %752 = stablehlo.reshape %751 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %753 = stablehlo.multiply %750, %752 : tensor<262144x!pf_babybear_mont>
    %754 = stablehlo.slice %746 [2:3, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %755 = stablehlo.reshape %754 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %756 = stablehlo.subtract %753, %755 : tensor<262144x!pf_babybear_mont>
    %757 = stablehlo.multiply %748, %756 : tensor<262144x!pf_babybear_mont>
    %cst_166 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %758 = stablehlo.reduce(%757 init: %cst_166) applies stablehlo.add across dimensions = [0] : (tensor<262144x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_167 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %759 = stablehlo.broadcast_in_dim %cst_167, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x262144x!pf_babybear_mont>
    %760 = stablehlo.multiply %728, %759 : tensor<4x262144x!pf_babybear_mont>
    %761 = stablehlo.add %760, %720 : tensor<4x262144x!pf_babybear_mont>
    %762 = stablehlo.slice %761 [3:4, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %763 = stablehlo.reshape %762 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %764 = stablehlo.slice %761 [0:1, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %765 = stablehlo.reshape %764 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %766 = stablehlo.slice %761 [1:2, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %767 = stablehlo.reshape %766 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %768 = stablehlo.multiply %765, %767 : tensor<262144x!pf_babybear_mont>
    %769 = stablehlo.slice %761 [2:3, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %770 = stablehlo.reshape %769 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %771 = stablehlo.subtract %768, %770 : tensor<262144x!pf_babybear_mont>
    %772 = stablehlo.multiply %763, %771 : tensor<262144x!pf_babybear_mont>
    %cst_168 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %773 = stablehlo.reduce(%772 init: %cst_168) applies stablehlo.add across dimensions = [0] : (tensor<262144x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_169 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %774 = stablehlo.broadcast_in_dim %cst_169, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x262144x!pf_babybear_mont>
    %775 = stablehlo.multiply %728, %774 : tensor<4x262144x!pf_babybear_mont>
    %776 = stablehlo.add %775, %720 : tensor<4x262144x!pf_babybear_mont>
    %777 = stablehlo.slice %776 [3:4, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %778 = stablehlo.reshape %777 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %779 = stablehlo.slice %776 [0:1, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %780 = stablehlo.reshape %779 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %781 = stablehlo.slice %776 [1:2, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %782 = stablehlo.reshape %781 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %783 = stablehlo.multiply %780, %782 : tensor<262144x!pf_babybear_mont>
    %784 = stablehlo.slice %776 [2:3, 0:262144] : (tensor<4x262144x!pf_babybear_mont>) -> tensor<1x262144x!pf_babybear_mont>
    %785 = stablehlo.reshape %784 : (tensor<1x262144x!pf_babybear_mont>) -> tensor<262144x!pf_babybear_mont>
    %786 = stablehlo.subtract %783, %785 : tensor<262144x!pf_babybear_mont>
    %787 = stablehlo.multiply %778, %786 : tensor<262144x!pf_babybear_mont>
    %cst_170 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %788 = stablehlo.reduce(%787 init: %cst_170) applies stablehlo.add across dimensions = [0] : (tensor<262144x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %789 = stablehlo.broadcast_in_dim %743, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %790 = stablehlo.broadcast_in_dim %758, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %791 = stablehlo.broadcast_in_dim %773, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %792 = stablehlo.broadcast_in_dim %788, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %793 = stablehlo.concatenate %789, %790, %791, %792, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %794 = stablehlo.reshape %710 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %795 = stablehlo.concatenate %cst, %794, %cst_8, %793, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_171 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %796 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %797 = stablehlo.slice %795 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %798 = stablehlo.reshape %797 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_172 = stablehlo.constant dense<0> : tensor<i32>
    %799 = stablehlo.broadcast_in_dim %c_172, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %800 = "stablehlo.scatter"(%796, %799, %798) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_173 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %801 = stablehlo.broadcast_in_dim %cst_173, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %802 = stablehlo.slice %795 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_174 = stablehlo.constant dense<0> : tensor<i32>
    %803 = stablehlo.broadcast_in_dim %c_174, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %804 = "stablehlo.scatter"(%801, %803, %802) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_175 = stablehlo.constant dense<8> : tensor<i32>
    %805 = stablehlo.broadcast_in_dim %c_175, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_176 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %806 = "stablehlo.scatter"(%804, %805, %cst_176) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_177 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %807 = stablehlo.broadcast_in_dim %cst_177, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %808 = stablehlo.concatenate %807, %806, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %809 = stablehlo.add %800, %808 : tensor<16x!pf_babybear_mont>
    %810 = call @permute(%809) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %811 = stablehlo.slice %810 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %812 = stablehlo.reshape %811 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %813 = stablehlo.broadcast_in_dim %812, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x262144x!pf_babybear_mont>
    %814 = stablehlo.multiply %728, %813 : tensor<4x262144x!pf_babybear_mont>
    %815 = stablehlo.add %814, %720 : tensor<4x262144x!pf_babybear_mont>
    %816 = stablehlo.iota dim = 0 : tensor<131072xi32>
    %c_178 = stablehlo.constant dense<2> : tensor<i32>
    %817 = stablehlo.broadcast_in_dim %c_178, dims = [] : (tensor<i32>) -> tensor<131072xi32>
    %818 = stablehlo.multiply %817, %816 : tensor<131072xi32>
    %c_179 = stablehlo.constant dense<0> : tensor<i32>
    %819 = stablehlo.broadcast_in_dim %c_179, dims = [] : (tensor<i32>) -> tensor<131072xi32>
    %820 = stablehlo.add %819, %818 : tensor<131072xi32>
    %821 = stablehlo.broadcast_in_dim %820, dims = [0] : (tensor<131072xi32>) -> tensor<131072x1xi32>
    %822 = "stablehlo.gather"(%815, %821) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x262144x!pf_babybear_mont>, tensor<131072x1xi32>) -> tensor<4x131072x!pf_babybear_mont>
    %823 = stablehlo.iota dim = 0 : tensor<131072xi32>
    %c_180 = stablehlo.constant dense<2> : tensor<i32>
    %824 = stablehlo.broadcast_in_dim %c_180, dims = [] : (tensor<i32>) -> tensor<131072xi32>
    %825 = stablehlo.multiply %824, %823 : tensor<131072xi32>
    %c_181 = stablehlo.constant dense<1> : tensor<i32>
    %826 = stablehlo.broadcast_in_dim %c_181, dims = [] : (tensor<i32>) -> tensor<131072xi32>
    %827 = stablehlo.add %826, %825 : tensor<131072xi32>
    %828 = stablehlo.broadcast_in_dim %827, dims = [0] : (tensor<131072xi32>) -> tensor<131072x1xi32>
    %829 = "stablehlo.gather"(%815, %828) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x262144x!pf_babybear_mont>, tensor<131072x1xi32>) -> tensor<4x131072x!pf_babybear_mont>
    %830 = stablehlo.subtract %829, %822 : tensor<4x131072x!pf_babybear_mont>
    %cst_182 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %831 = stablehlo.broadcast_in_dim %cst_182, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x131072x!pf_babybear_mont>
    %832 = stablehlo.multiply %830, %831 : tensor<4x131072x!pf_babybear_mont>
    %833 = stablehlo.add %832, %822 : tensor<4x131072x!pf_babybear_mont>
    %834 = stablehlo.slice %833 [3:4, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %835 = stablehlo.reshape %834 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %836 = stablehlo.slice %833 [0:1, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %837 = stablehlo.reshape %836 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %838 = stablehlo.slice %833 [1:2, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %839 = stablehlo.reshape %838 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %840 = stablehlo.multiply %837, %839 : tensor<131072x!pf_babybear_mont>
    %841 = stablehlo.slice %833 [2:3, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %842 = stablehlo.reshape %841 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %843 = stablehlo.subtract %840, %842 : tensor<131072x!pf_babybear_mont>
    %844 = stablehlo.multiply %835, %843 : tensor<131072x!pf_babybear_mont>
    %cst_183 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %845 = stablehlo.reduce(%844 init: %cst_183) applies stablehlo.add across dimensions = [0] : (tensor<131072x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_184 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %846 = stablehlo.broadcast_in_dim %cst_184, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x131072x!pf_babybear_mont>
    %847 = stablehlo.multiply %830, %846 : tensor<4x131072x!pf_babybear_mont>
    %848 = stablehlo.add %847, %822 : tensor<4x131072x!pf_babybear_mont>
    %849 = stablehlo.slice %848 [3:4, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %850 = stablehlo.reshape %849 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %851 = stablehlo.slice %848 [0:1, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %852 = stablehlo.reshape %851 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %853 = stablehlo.slice %848 [1:2, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %854 = stablehlo.reshape %853 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %855 = stablehlo.multiply %852, %854 : tensor<131072x!pf_babybear_mont>
    %856 = stablehlo.slice %848 [2:3, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %857 = stablehlo.reshape %856 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %858 = stablehlo.subtract %855, %857 : tensor<131072x!pf_babybear_mont>
    %859 = stablehlo.multiply %850, %858 : tensor<131072x!pf_babybear_mont>
    %cst_185 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %860 = stablehlo.reduce(%859 init: %cst_185) applies stablehlo.add across dimensions = [0] : (tensor<131072x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_186 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %861 = stablehlo.broadcast_in_dim %cst_186, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x131072x!pf_babybear_mont>
    %862 = stablehlo.multiply %830, %861 : tensor<4x131072x!pf_babybear_mont>
    %863 = stablehlo.add %862, %822 : tensor<4x131072x!pf_babybear_mont>
    %864 = stablehlo.slice %863 [3:4, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %865 = stablehlo.reshape %864 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %866 = stablehlo.slice %863 [0:1, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %867 = stablehlo.reshape %866 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %868 = stablehlo.slice %863 [1:2, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %869 = stablehlo.reshape %868 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %870 = stablehlo.multiply %867, %869 : tensor<131072x!pf_babybear_mont>
    %871 = stablehlo.slice %863 [2:3, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %872 = stablehlo.reshape %871 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %873 = stablehlo.subtract %870, %872 : tensor<131072x!pf_babybear_mont>
    %874 = stablehlo.multiply %865, %873 : tensor<131072x!pf_babybear_mont>
    %cst_187 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %875 = stablehlo.reduce(%874 init: %cst_187) applies stablehlo.add across dimensions = [0] : (tensor<131072x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_188 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %876 = stablehlo.broadcast_in_dim %cst_188, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x131072x!pf_babybear_mont>
    %877 = stablehlo.multiply %830, %876 : tensor<4x131072x!pf_babybear_mont>
    %878 = stablehlo.add %877, %822 : tensor<4x131072x!pf_babybear_mont>
    %879 = stablehlo.slice %878 [3:4, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %880 = stablehlo.reshape %879 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %881 = stablehlo.slice %878 [0:1, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %882 = stablehlo.reshape %881 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %883 = stablehlo.slice %878 [1:2, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %884 = stablehlo.reshape %883 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %885 = stablehlo.multiply %882, %884 : tensor<131072x!pf_babybear_mont>
    %886 = stablehlo.slice %878 [2:3, 0:131072] : (tensor<4x131072x!pf_babybear_mont>) -> tensor<1x131072x!pf_babybear_mont>
    %887 = stablehlo.reshape %886 : (tensor<1x131072x!pf_babybear_mont>) -> tensor<131072x!pf_babybear_mont>
    %888 = stablehlo.subtract %885, %887 : tensor<131072x!pf_babybear_mont>
    %889 = stablehlo.multiply %880, %888 : tensor<131072x!pf_babybear_mont>
    %cst_189 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %890 = stablehlo.reduce(%889 init: %cst_189) applies stablehlo.add across dimensions = [0] : (tensor<131072x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %891 = stablehlo.broadcast_in_dim %845, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %892 = stablehlo.broadcast_in_dim %860, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %893 = stablehlo.broadcast_in_dim %875, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %894 = stablehlo.broadcast_in_dim %890, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %895 = stablehlo.concatenate %891, %892, %893, %894, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %896 = stablehlo.reshape %812 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %897 = stablehlo.concatenate %cst, %896, %cst_9, %895, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_190 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %898 = stablehlo.broadcast_in_dim %cst_190, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %899 = stablehlo.slice %897 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %900 = stablehlo.reshape %899 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_191 = stablehlo.constant dense<0> : tensor<i32>
    %901 = stablehlo.broadcast_in_dim %c_191, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %902 = "stablehlo.scatter"(%898, %901, %900) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_192 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %903 = stablehlo.broadcast_in_dim %cst_192, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %904 = stablehlo.slice %897 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_193 = stablehlo.constant dense<0> : tensor<i32>
    %905 = stablehlo.broadcast_in_dim %c_193, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %906 = "stablehlo.scatter"(%903, %905, %904) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_194 = stablehlo.constant dense<8> : tensor<i32>
    %907 = stablehlo.broadcast_in_dim %c_194, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_195 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %908 = "stablehlo.scatter"(%906, %907, %cst_195) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_196 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %909 = stablehlo.broadcast_in_dim %cst_196, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %910 = stablehlo.concatenate %909, %908, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %911 = stablehlo.add %902, %910 : tensor<16x!pf_babybear_mont>
    %912 = call @permute(%911) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %913 = stablehlo.slice %912 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %914 = stablehlo.reshape %913 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %915 = stablehlo.broadcast_in_dim %914, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x131072x!pf_babybear_mont>
    %916 = stablehlo.multiply %830, %915 : tensor<4x131072x!pf_babybear_mont>
    %917 = stablehlo.add %916, %822 : tensor<4x131072x!pf_babybear_mont>
    %918 = stablehlo.iota dim = 0 : tensor<65536xi32>
    %c_197 = stablehlo.constant dense<2> : tensor<i32>
    %919 = stablehlo.broadcast_in_dim %c_197, dims = [] : (tensor<i32>) -> tensor<65536xi32>
    %920 = stablehlo.multiply %919, %918 : tensor<65536xi32>
    %c_198 = stablehlo.constant dense<0> : tensor<i32>
    %921 = stablehlo.broadcast_in_dim %c_198, dims = [] : (tensor<i32>) -> tensor<65536xi32>
    %922 = stablehlo.add %921, %920 : tensor<65536xi32>
    %923 = stablehlo.broadcast_in_dim %922, dims = [0] : (tensor<65536xi32>) -> tensor<65536x1xi32>
    %924 = "stablehlo.gather"(%917, %923) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x131072x!pf_babybear_mont>, tensor<65536x1xi32>) -> tensor<4x65536x!pf_babybear_mont>
    %925 = stablehlo.iota dim = 0 : tensor<65536xi32>
    %c_199 = stablehlo.constant dense<2> : tensor<i32>
    %926 = stablehlo.broadcast_in_dim %c_199, dims = [] : (tensor<i32>) -> tensor<65536xi32>
    %927 = stablehlo.multiply %926, %925 : tensor<65536xi32>
    %c_200 = stablehlo.constant dense<1> : tensor<i32>
    %928 = stablehlo.broadcast_in_dim %c_200, dims = [] : (tensor<i32>) -> tensor<65536xi32>
    %929 = stablehlo.add %928, %927 : tensor<65536xi32>
    %930 = stablehlo.broadcast_in_dim %929, dims = [0] : (tensor<65536xi32>) -> tensor<65536x1xi32>
    %931 = "stablehlo.gather"(%917, %930) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x131072x!pf_babybear_mont>, tensor<65536x1xi32>) -> tensor<4x65536x!pf_babybear_mont>
    %932 = stablehlo.subtract %931, %924 : tensor<4x65536x!pf_babybear_mont>
    %cst_201 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %933 = stablehlo.broadcast_in_dim %cst_201, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x65536x!pf_babybear_mont>
    %934 = stablehlo.multiply %932, %933 : tensor<4x65536x!pf_babybear_mont>
    %935 = stablehlo.add %934, %924 : tensor<4x65536x!pf_babybear_mont>
    %936 = stablehlo.slice %935 [3:4, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %937 = stablehlo.reshape %936 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %938 = stablehlo.slice %935 [0:1, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %939 = stablehlo.reshape %938 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %940 = stablehlo.slice %935 [1:2, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %941 = stablehlo.reshape %940 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %942 = stablehlo.multiply %939, %941 : tensor<65536x!pf_babybear_mont>
    %943 = stablehlo.slice %935 [2:3, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %944 = stablehlo.reshape %943 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %945 = stablehlo.subtract %942, %944 : tensor<65536x!pf_babybear_mont>
    %946 = stablehlo.multiply %937, %945 : tensor<65536x!pf_babybear_mont>
    %cst_202 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %947 = stablehlo.reduce(%946 init: %cst_202) applies stablehlo.add across dimensions = [0] : (tensor<65536x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_203 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %948 = stablehlo.broadcast_in_dim %cst_203, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x65536x!pf_babybear_mont>
    %949 = stablehlo.multiply %932, %948 : tensor<4x65536x!pf_babybear_mont>
    %950 = stablehlo.add %949, %924 : tensor<4x65536x!pf_babybear_mont>
    %951 = stablehlo.slice %950 [3:4, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %952 = stablehlo.reshape %951 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %953 = stablehlo.slice %950 [0:1, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %954 = stablehlo.reshape %953 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %955 = stablehlo.slice %950 [1:2, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %956 = stablehlo.reshape %955 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %957 = stablehlo.multiply %954, %956 : tensor<65536x!pf_babybear_mont>
    %958 = stablehlo.slice %950 [2:3, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %959 = stablehlo.reshape %958 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %960 = stablehlo.subtract %957, %959 : tensor<65536x!pf_babybear_mont>
    %961 = stablehlo.multiply %952, %960 : tensor<65536x!pf_babybear_mont>
    %cst_204 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %962 = stablehlo.reduce(%961 init: %cst_204) applies stablehlo.add across dimensions = [0] : (tensor<65536x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_205 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %963 = stablehlo.broadcast_in_dim %cst_205, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x65536x!pf_babybear_mont>
    %964 = stablehlo.multiply %932, %963 : tensor<4x65536x!pf_babybear_mont>
    %965 = stablehlo.add %964, %924 : tensor<4x65536x!pf_babybear_mont>
    %966 = stablehlo.slice %965 [3:4, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %967 = stablehlo.reshape %966 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %968 = stablehlo.slice %965 [0:1, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %969 = stablehlo.reshape %968 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %970 = stablehlo.slice %965 [1:2, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %971 = stablehlo.reshape %970 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %972 = stablehlo.multiply %969, %971 : tensor<65536x!pf_babybear_mont>
    %973 = stablehlo.slice %965 [2:3, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %974 = stablehlo.reshape %973 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %975 = stablehlo.subtract %972, %974 : tensor<65536x!pf_babybear_mont>
    %976 = stablehlo.multiply %967, %975 : tensor<65536x!pf_babybear_mont>
    %cst_206 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %977 = stablehlo.reduce(%976 init: %cst_206) applies stablehlo.add across dimensions = [0] : (tensor<65536x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_207 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %978 = stablehlo.broadcast_in_dim %cst_207, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x65536x!pf_babybear_mont>
    %979 = stablehlo.multiply %932, %978 : tensor<4x65536x!pf_babybear_mont>
    %980 = stablehlo.add %979, %924 : tensor<4x65536x!pf_babybear_mont>
    %981 = stablehlo.slice %980 [3:4, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %982 = stablehlo.reshape %981 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %983 = stablehlo.slice %980 [0:1, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %984 = stablehlo.reshape %983 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %985 = stablehlo.slice %980 [1:2, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %986 = stablehlo.reshape %985 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %987 = stablehlo.multiply %984, %986 : tensor<65536x!pf_babybear_mont>
    %988 = stablehlo.slice %980 [2:3, 0:65536] : (tensor<4x65536x!pf_babybear_mont>) -> tensor<1x65536x!pf_babybear_mont>
    %989 = stablehlo.reshape %988 : (tensor<1x65536x!pf_babybear_mont>) -> tensor<65536x!pf_babybear_mont>
    %990 = stablehlo.subtract %987, %989 : tensor<65536x!pf_babybear_mont>
    %991 = stablehlo.multiply %982, %990 : tensor<65536x!pf_babybear_mont>
    %cst_208 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %992 = stablehlo.reduce(%991 init: %cst_208) applies stablehlo.add across dimensions = [0] : (tensor<65536x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %993 = stablehlo.broadcast_in_dim %947, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %994 = stablehlo.broadcast_in_dim %962, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %995 = stablehlo.broadcast_in_dim %977, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %996 = stablehlo.broadcast_in_dim %992, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %997 = stablehlo.concatenate %993, %994, %995, %996, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %998 = stablehlo.reshape %914 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %999 = stablehlo.concatenate %cst, %998, %cst_10, %997, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_209 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1000 = stablehlo.broadcast_in_dim %cst_209, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1001 = stablehlo.slice %999 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1002 = stablehlo.reshape %1001 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_210 = stablehlo.constant dense<0> : tensor<i32>
    %1003 = stablehlo.broadcast_in_dim %c_210, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1004 = "stablehlo.scatter"(%1000, %1003, %1002) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_211 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1005 = stablehlo.broadcast_in_dim %cst_211, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1006 = stablehlo.slice %999 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_212 = stablehlo.constant dense<0> : tensor<i32>
    %1007 = stablehlo.broadcast_in_dim %c_212, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1008 = "stablehlo.scatter"(%1005, %1007, %1006) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_213 = stablehlo.constant dense<8> : tensor<i32>
    %1009 = stablehlo.broadcast_in_dim %c_213, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_214 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1010 = "stablehlo.scatter"(%1008, %1009, %cst_214) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_215 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1011 = stablehlo.broadcast_in_dim %cst_215, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1012 = stablehlo.concatenate %1011, %1010, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1013 = stablehlo.add %1004, %1012 : tensor<16x!pf_babybear_mont>
    %1014 = call @permute(%1013) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1015 = stablehlo.slice %1014 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1016 = stablehlo.reshape %1015 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1017 = stablehlo.broadcast_in_dim %1016, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x65536x!pf_babybear_mont>
    %1018 = stablehlo.multiply %932, %1017 : tensor<4x65536x!pf_babybear_mont>
    %1019 = stablehlo.add %1018, %924 : tensor<4x65536x!pf_babybear_mont>
    %1020 = stablehlo.iota dim = 0 : tensor<32768xi32>
    %c_216 = stablehlo.constant dense<2> : tensor<i32>
    %1021 = stablehlo.broadcast_in_dim %c_216, dims = [] : (tensor<i32>) -> tensor<32768xi32>
    %1022 = stablehlo.multiply %1021, %1020 : tensor<32768xi32>
    %c_217 = stablehlo.constant dense<0> : tensor<i32>
    %1023 = stablehlo.broadcast_in_dim %c_217, dims = [] : (tensor<i32>) -> tensor<32768xi32>
    %1024 = stablehlo.add %1023, %1022 : tensor<32768xi32>
    %1025 = stablehlo.broadcast_in_dim %1024, dims = [0] : (tensor<32768xi32>) -> tensor<32768x1xi32>
    %1026 = "stablehlo.gather"(%1019, %1025) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x65536x!pf_babybear_mont>, tensor<32768x1xi32>) -> tensor<4x32768x!pf_babybear_mont>
    %1027 = stablehlo.iota dim = 0 : tensor<32768xi32>
    %c_218 = stablehlo.constant dense<2> : tensor<i32>
    %1028 = stablehlo.broadcast_in_dim %c_218, dims = [] : (tensor<i32>) -> tensor<32768xi32>
    %1029 = stablehlo.multiply %1028, %1027 : tensor<32768xi32>
    %c_219 = stablehlo.constant dense<1> : tensor<i32>
    %1030 = stablehlo.broadcast_in_dim %c_219, dims = [] : (tensor<i32>) -> tensor<32768xi32>
    %1031 = stablehlo.add %1030, %1029 : tensor<32768xi32>
    %1032 = stablehlo.broadcast_in_dim %1031, dims = [0] : (tensor<32768xi32>) -> tensor<32768x1xi32>
    %1033 = "stablehlo.gather"(%1019, %1032) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x65536x!pf_babybear_mont>, tensor<32768x1xi32>) -> tensor<4x32768x!pf_babybear_mont>
    %1034 = stablehlo.subtract %1033, %1026 : tensor<4x32768x!pf_babybear_mont>
    %cst_220 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1035 = stablehlo.broadcast_in_dim %cst_220, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32768x!pf_babybear_mont>
    %1036 = stablehlo.multiply %1034, %1035 : tensor<4x32768x!pf_babybear_mont>
    %1037 = stablehlo.add %1036, %1026 : tensor<4x32768x!pf_babybear_mont>
    %1038 = stablehlo.slice %1037 [3:4, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1039 = stablehlo.reshape %1038 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1040 = stablehlo.slice %1037 [0:1, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1041 = stablehlo.reshape %1040 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1042 = stablehlo.slice %1037 [1:2, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1043 = stablehlo.reshape %1042 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1044 = stablehlo.multiply %1041, %1043 : tensor<32768x!pf_babybear_mont>
    %1045 = stablehlo.slice %1037 [2:3, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1046 = stablehlo.reshape %1045 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1047 = stablehlo.subtract %1044, %1046 : tensor<32768x!pf_babybear_mont>
    %1048 = stablehlo.multiply %1039, %1047 : tensor<32768x!pf_babybear_mont>
    %cst_221 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1049 = stablehlo.reduce(%1048 init: %cst_221) applies stablehlo.add across dimensions = [0] : (tensor<32768x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_222 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1050 = stablehlo.broadcast_in_dim %cst_222, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32768x!pf_babybear_mont>
    %1051 = stablehlo.multiply %1034, %1050 : tensor<4x32768x!pf_babybear_mont>
    %1052 = stablehlo.add %1051, %1026 : tensor<4x32768x!pf_babybear_mont>
    %1053 = stablehlo.slice %1052 [3:4, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1054 = stablehlo.reshape %1053 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1055 = stablehlo.slice %1052 [0:1, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1056 = stablehlo.reshape %1055 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1057 = stablehlo.slice %1052 [1:2, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1058 = stablehlo.reshape %1057 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1059 = stablehlo.multiply %1056, %1058 : tensor<32768x!pf_babybear_mont>
    %1060 = stablehlo.slice %1052 [2:3, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1061 = stablehlo.reshape %1060 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1062 = stablehlo.subtract %1059, %1061 : tensor<32768x!pf_babybear_mont>
    %1063 = stablehlo.multiply %1054, %1062 : tensor<32768x!pf_babybear_mont>
    %cst_223 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1064 = stablehlo.reduce(%1063 init: %cst_223) applies stablehlo.add across dimensions = [0] : (tensor<32768x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_224 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1065 = stablehlo.broadcast_in_dim %cst_224, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32768x!pf_babybear_mont>
    %1066 = stablehlo.multiply %1034, %1065 : tensor<4x32768x!pf_babybear_mont>
    %1067 = stablehlo.add %1066, %1026 : tensor<4x32768x!pf_babybear_mont>
    %1068 = stablehlo.slice %1067 [3:4, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1069 = stablehlo.reshape %1068 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1070 = stablehlo.slice %1067 [0:1, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1071 = stablehlo.reshape %1070 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1072 = stablehlo.slice %1067 [1:2, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1073 = stablehlo.reshape %1072 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1074 = stablehlo.multiply %1071, %1073 : tensor<32768x!pf_babybear_mont>
    %1075 = stablehlo.slice %1067 [2:3, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1076 = stablehlo.reshape %1075 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1077 = stablehlo.subtract %1074, %1076 : tensor<32768x!pf_babybear_mont>
    %1078 = stablehlo.multiply %1069, %1077 : tensor<32768x!pf_babybear_mont>
    %cst_225 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1079 = stablehlo.reduce(%1078 init: %cst_225) applies stablehlo.add across dimensions = [0] : (tensor<32768x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_226 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1080 = stablehlo.broadcast_in_dim %cst_226, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32768x!pf_babybear_mont>
    %1081 = stablehlo.multiply %1034, %1080 : tensor<4x32768x!pf_babybear_mont>
    %1082 = stablehlo.add %1081, %1026 : tensor<4x32768x!pf_babybear_mont>
    %1083 = stablehlo.slice %1082 [3:4, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1084 = stablehlo.reshape %1083 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1085 = stablehlo.slice %1082 [0:1, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1086 = stablehlo.reshape %1085 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1087 = stablehlo.slice %1082 [1:2, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1088 = stablehlo.reshape %1087 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1089 = stablehlo.multiply %1086, %1088 : tensor<32768x!pf_babybear_mont>
    %1090 = stablehlo.slice %1082 [2:3, 0:32768] : (tensor<4x32768x!pf_babybear_mont>) -> tensor<1x32768x!pf_babybear_mont>
    %1091 = stablehlo.reshape %1090 : (tensor<1x32768x!pf_babybear_mont>) -> tensor<32768x!pf_babybear_mont>
    %1092 = stablehlo.subtract %1089, %1091 : tensor<32768x!pf_babybear_mont>
    %1093 = stablehlo.multiply %1084, %1092 : tensor<32768x!pf_babybear_mont>
    %cst_227 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1094 = stablehlo.reduce(%1093 init: %cst_227) applies stablehlo.add across dimensions = [0] : (tensor<32768x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1095 = stablehlo.broadcast_in_dim %1049, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1096 = stablehlo.broadcast_in_dim %1064, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1097 = stablehlo.broadcast_in_dim %1079, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1098 = stablehlo.broadcast_in_dim %1094, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1099 = stablehlo.concatenate %1095, %1096, %1097, %1098, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1100 = stablehlo.reshape %1016 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1101 = stablehlo.concatenate %cst, %1100, %cst_11, %1099, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_228 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1102 = stablehlo.broadcast_in_dim %cst_228, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1103 = stablehlo.slice %1101 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1104 = stablehlo.reshape %1103 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_229 = stablehlo.constant dense<0> : tensor<i32>
    %1105 = stablehlo.broadcast_in_dim %c_229, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1106 = "stablehlo.scatter"(%1102, %1105, %1104) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_230 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1107 = stablehlo.broadcast_in_dim %cst_230, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1108 = stablehlo.slice %1101 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_231 = stablehlo.constant dense<0> : tensor<i32>
    %1109 = stablehlo.broadcast_in_dim %c_231, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1110 = "stablehlo.scatter"(%1107, %1109, %1108) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_232 = stablehlo.constant dense<8> : tensor<i32>
    %1111 = stablehlo.broadcast_in_dim %c_232, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_233 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1112 = "stablehlo.scatter"(%1110, %1111, %cst_233) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_234 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1113 = stablehlo.broadcast_in_dim %cst_234, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1114 = stablehlo.concatenate %1113, %1112, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1115 = stablehlo.add %1106, %1114 : tensor<16x!pf_babybear_mont>
    %1116 = call @permute(%1115) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1117 = stablehlo.slice %1116 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1118 = stablehlo.reshape %1117 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1119 = stablehlo.broadcast_in_dim %1118, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32768x!pf_babybear_mont>
    %1120 = stablehlo.multiply %1034, %1119 : tensor<4x32768x!pf_babybear_mont>
    %1121 = stablehlo.add %1120, %1026 : tensor<4x32768x!pf_babybear_mont>
    %1122 = stablehlo.iota dim = 0 : tensor<16384xi32>
    %c_235 = stablehlo.constant dense<2> : tensor<i32>
    %1123 = stablehlo.broadcast_in_dim %c_235, dims = [] : (tensor<i32>) -> tensor<16384xi32>
    %1124 = stablehlo.multiply %1123, %1122 : tensor<16384xi32>
    %c_236 = stablehlo.constant dense<0> : tensor<i32>
    %1125 = stablehlo.broadcast_in_dim %c_236, dims = [] : (tensor<i32>) -> tensor<16384xi32>
    %1126 = stablehlo.add %1125, %1124 : tensor<16384xi32>
    %1127 = stablehlo.broadcast_in_dim %1126, dims = [0] : (tensor<16384xi32>) -> tensor<16384x1xi32>
    %1128 = "stablehlo.gather"(%1121, %1127) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x32768x!pf_babybear_mont>, tensor<16384x1xi32>) -> tensor<4x16384x!pf_babybear_mont>
    %1129 = stablehlo.iota dim = 0 : tensor<16384xi32>
    %c_237 = stablehlo.constant dense<2> : tensor<i32>
    %1130 = stablehlo.broadcast_in_dim %c_237, dims = [] : (tensor<i32>) -> tensor<16384xi32>
    %1131 = stablehlo.multiply %1130, %1129 : tensor<16384xi32>
    %c_238 = stablehlo.constant dense<1> : tensor<i32>
    %1132 = stablehlo.broadcast_in_dim %c_238, dims = [] : (tensor<i32>) -> tensor<16384xi32>
    %1133 = stablehlo.add %1132, %1131 : tensor<16384xi32>
    %1134 = stablehlo.broadcast_in_dim %1133, dims = [0] : (tensor<16384xi32>) -> tensor<16384x1xi32>
    %1135 = "stablehlo.gather"(%1121, %1134) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x32768x!pf_babybear_mont>, tensor<16384x1xi32>) -> tensor<4x16384x!pf_babybear_mont>
    %1136 = stablehlo.subtract %1135, %1128 : tensor<4x16384x!pf_babybear_mont>
    %cst_239 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1137 = stablehlo.broadcast_in_dim %cst_239, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16384x!pf_babybear_mont>
    %1138 = stablehlo.multiply %1136, %1137 : tensor<4x16384x!pf_babybear_mont>
    %1139 = stablehlo.add %1138, %1128 : tensor<4x16384x!pf_babybear_mont>
    %1140 = stablehlo.slice %1139 [3:4, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1141 = stablehlo.reshape %1140 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1142 = stablehlo.slice %1139 [0:1, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1143 = stablehlo.reshape %1142 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1144 = stablehlo.slice %1139 [1:2, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1145 = stablehlo.reshape %1144 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1146 = stablehlo.multiply %1143, %1145 : tensor<16384x!pf_babybear_mont>
    %1147 = stablehlo.slice %1139 [2:3, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1148 = stablehlo.reshape %1147 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1149 = stablehlo.subtract %1146, %1148 : tensor<16384x!pf_babybear_mont>
    %1150 = stablehlo.multiply %1141, %1149 : tensor<16384x!pf_babybear_mont>
    %cst_240 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1151 = stablehlo.reduce(%1150 init: %cst_240) applies stablehlo.add across dimensions = [0] : (tensor<16384x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_241 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1152 = stablehlo.broadcast_in_dim %cst_241, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16384x!pf_babybear_mont>
    %1153 = stablehlo.multiply %1136, %1152 : tensor<4x16384x!pf_babybear_mont>
    %1154 = stablehlo.add %1153, %1128 : tensor<4x16384x!pf_babybear_mont>
    %1155 = stablehlo.slice %1154 [3:4, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1156 = stablehlo.reshape %1155 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1157 = stablehlo.slice %1154 [0:1, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1158 = stablehlo.reshape %1157 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1159 = stablehlo.slice %1154 [1:2, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1160 = stablehlo.reshape %1159 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1161 = stablehlo.multiply %1158, %1160 : tensor<16384x!pf_babybear_mont>
    %1162 = stablehlo.slice %1154 [2:3, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1163 = stablehlo.reshape %1162 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1164 = stablehlo.subtract %1161, %1163 : tensor<16384x!pf_babybear_mont>
    %1165 = stablehlo.multiply %1156, %1164 : tensor<16384x!pf_babybear_mont>
    %cst_242 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1166 = stablehlo.reduce(%1165 init: %cst_242) applies stablehlo.add across dimensions = [0] : (tensor<16384x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_243 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1167 = stablehlo.broadcast_in_dim %cst_243, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16384x!pf_babybear_mont>
    %1168 = stablehlo.multiply %1136, %1167 : tensor<4x16384x!pf_babybear_mont>
    %1169 = stablehlo.add %1168, %1128 : tensor<4x16384x!pf_babybear_mont>
    %1170 = stablehlo.slice %1169 [3:4, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1171 = stablehlo.reshape %1170 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1172 = stablehlo.slice %1169 [0:1, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1173 = stablehlo.reshape %1172 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1174 = stablehlo.slice %1169 [1:2, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1175 = stablehlo.reshape %1174 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1176 = stablehlo.multiply %1173, %1175 : tensor<16384x!pf_babybear_mont>
    %1177 = stablehlo.slice %1169 [2:3, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1178 = stablehlo.reshape %1177 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1179 = stablehlo.subtract %1176, %1178 : tensor<16384x!pf_babybear_mont>
    %1180 = stablehlo.multiply %1171, %1179 : tensor<16384x!pf_babybear_mont>
    %cst_244 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1181 = stablehlo.reduce(%1180 init: %cst_244) applies stablehlo.add across dimensions = [0] : (tensor<16384x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_245 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1182 = stablehlo.broadcast_in_dim %cst_245, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16384x!pf_babybear_mont>
    %1183 = stablehlo.multiply %1136, %1182 : tensor<4x16384x!pf_babybear_mont>
    %1184 = stablehlo.add %1183, %1128 : tensor<4x16384x!pf_babybear_mont>
    %1185 = stablehlo.slice %1184 [3:4, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1186 = stablehlo.reshape %1185 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1187 = stablehlo.slice %1184 [0:1, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1188 = stablehlo.reshape %1187 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1189 = stablehlo.slice %1184 [1:2, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1190 = stablehlo.reshape %1189 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1191 = stablehlo.multiply %1188, %1190 : tensor<16384x!pf_babybear_mont>
    %1192 = stablehlo.slice %1184 [2:3, 0:16384] : (tensor<4x16384x!pf_babybear_mont>) -> tensor<1x16384x!pf_babybear_mont>
    %1193 = stablehlo.reshape %1192 : (tensor<1x16384x!pf_babybear_mont>) -> tensor<16384x!pf_babybear_mont>
    %1194 = stablehlo.subtract %1191, %1193 : tensor<16384x!pf_babybear_mont>
    %1195 = stablehlo.multiply %1186, %1194 : tensor<16384x!pf_babybear_mont>
    %cst_246 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1196 = stablehlo.reduce(%1195 init: %cst_246) applies stablehlo.add across dimensions = [0] : (tensor<16384x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1197 = stablehlo.broadcast_in_dim %1151, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1198 = stablehlo.broadcast_in_dim %1166, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1199 = stablehlo.broadcast_in_dim %1181, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1200 = stablehlo.broadcast_in_dim %1196, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1201 = stablehlo.concatenate %1197, %1198, %1199, %1200, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1202 = stablehlo.reshape %1118 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1203 = stablehlo.concatenate %cst, %1202, %cst_12, %1201, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_247 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1204 = stablehlo.broadcast_in_dim %cst_247, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1205 = stablehlo.slice %1203 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1206 = stablehlo.reshape %1205 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_248 = stablehlo.constant dense<0> : tensor<i32>
    %1207 = stablehlo.broadcast_in_dim %c_248, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1208 = "stablehlo.scatter"(%1204, %1207, %1206) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_249 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1209 = stablehlo.broadcast_in_dim %cst_249, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1210 = stablehlo.slice %1203 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_250 = stablehlo.constant dense<0> : tensor<i32>
    %1211 = stablehlo.broadcast_in_dim %c_250, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1212 = "stablehlo.scatter"(%1209, %1211, %1210) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_251 = stablehlo.constant dense<8> : tensor<i32>
    %1213 = stablehlo.broadcast_in_dim %c_251, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_252 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1214 = "stablehlo.scatter"(%1212, %1213, %cst_252) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_253 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1215 = stablehlo.broadcast_in_dim %cst_253, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1216 = stablehlo.concatenate %1215, %1214, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1217 = stablehlo.add %1208, %1216 : tensor<16x!pf_babybear_mont>
    %1218 = call @permute(%1217) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1219 = stablehlo.slice %1218 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1220 = stablehlo.reshape %1219 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1221 = stablehlo.broadcast_in_dim %1220, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16384x!pf_babybear_mont>
    %1222 = stablehlo.multiply %1136, %1221 : tensor<4x16384x!pf_babybear_mont>
    %1223 = stablehlo.add %1222, %1128 : tensor<4x16384x!pf_babybear_mont>
    %1224 = stablehlo.iota dim = 0 : tensor<8192xi32>
    %c_254 = stablehlo.constant dense<2> : tensor<i32>
    %1225 = stablehlo.broadcast_in_dim %c_254, dims = [] : (tensor<i32>) -> tensor<8192xi32>
    %1226 = stablehlo.multiply %1225, %1224 : tensor<8192xi32>
    %c_255 = stablehlo.constant dense<0> : tensor<i32>
    %1227 = stablehlo.broadcast_in_dim %c_255, dims = [] : (tensor<i32>) -> tensor<8192xi32>
    %1228 = stablehlo.add %1227, %1226 : tensor<8192xi32>
    %1229 = stablehlo.broadcast_in_dim %1228, dims = [0] : (tensor<8192xi32>) -> tensor<8192x1xi32>
    %1230 = "stablehlo.gather"(%1223, %1229) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x16384x!pf_babybear_mont>, tensor<8192x1xi32>) -> tensor<4x8192x!pf_babybear_mont>
    %1231 = stablehlo.iota dim = 0 : tensor<8192xi32>
    %c_256 = stablehlo.constant dense<2> : tensor<i32>
    %1232 = stablehlo.broadcast_in_dim %c_256, dims = [] : (tensor<i32>) -> tensor<8192xi32>
    %1233 = stablehlo.multiply %1232, %1231 : tensor<8192xi32>
    %c_257 = stablehlo.constant dense<1> : tensor<i32>
    %1234 = stablehlo.broadcast_in_dim %c_257, dims = [] : (tensor<i32>) -> tensor<8192xi32>
    %1235 = stablehlo.add %1234, %1233 : tensor<8192xi32>
    %1236 = stablehlo.broadcast_in_dim %1235, dims = [0] : (tensor<8192xi32>) -> tensor<8192x1xi32>
    %1237 = "stablehlo.gather"(%1223, %1236) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x16384x!pf_babybear_mont>, tensor<8192x1xi32>) -> tensor<4x8192x!pf_babybear_mont>
    %1238 = stablehlo.subtract %1237, %1230 : tensor<4x8192x!pf_babybear_mont>
    %cst_258 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1239 = stablehlo.broadcast_in_dim %cst_258, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8192x!pf_babybear_mont>
    %1240 = stablehlo.multiply %1238, %1239 : tensor<4x8192x!pf_babybear_mont>
    %1241 = stablehlo.add %1240, %1230 : tensor<4x8192x!pf_babybear_mont>
    %1242 = stablehlo.slice %1241 [3:4, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1243 = stablehlo.reshape %1242 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1244 = stablehlo.slice %1241 [0:1, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1245 = stablehlo.reshape %1244 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1246 = stablehlo.slice %1241 [1:2, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1247 = stablehlo.reshape %1246 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1248 = stablehlo.multiply %1245, %1247 : tensor<8192x!pf_babybear_mont>
    %1249 = stablehlo.slice %1241 [2:3, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1250 = stablehlo.reshape %1249 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1251 = stablehlo.subtract %1248, %1250 : tensor<8192x!pf_babybear_mont>
    %1252 = stablehlo.multiply %1243, %1251 : tensor<8192x!pf_babybear_mont>
    %cst_259 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1253 = stablehlo.reduce(%1252 init: %cst_259) applies stablehlo.add across dimensions = [0] : (tensor<8192x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_260 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1254 = stablehlo.broadcast_in_dim %cst_260, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8192x!pf_babybear_mont>
    %1255 = stablehlo.multiply %1238, %1254 : tensor<4x8192x!pf_babybear_mont>
    %1256 = stablehlo.add %1255, %1230 : tensor<4x8192x!pf_babybear_mont>
    %1257 = stablehlo.slice %1256 [3:4, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1258 = stablehlo.reshape %1257 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1259 = stablehlo.slice %1256 [0:1, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1260 = stablehlo.reshape %1259 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1261 = stablehlo.slice %1256 [1:2, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1262 = stablehlo.reshape %1261 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1263 = stablehlo.multiply %1260, %1262 : tensor<8192x!pf_babybear_mont>
    %1264 = stablehlo.slice %1256 [2:3, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1265 = stablehlo.reshape %1264 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1266 = stablehlo.subtract %1263, %1265 : tensor<8192x!pf_babybear_mont>
    %1267 = stablehlo.multiply %1258, %1266 : tensor<8192x!pf_babybear_mont>
    %cst_261 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1268 = stablehlo.reduce(%1267 init: %cst_261) applies stablehlo.add across dimensions = [0] : (tensor<8192x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_262 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1269 = stablehlo.broadcast_in_dim %cst_262, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8192x!pf_babybear_mont>
    %1270 = stablehlo.multiply %1238, %1269 : tensor<4x8192x!pf_babybear_mont>
    %1271 = stablehlo.add %1270, %1230 : tensor<4x8192x!pf_babybear_mont>
    %1272 = stablehlo.slice %1271 [3:4, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1273 = stablehlo.reshape %1272 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1274 = stablehlo.slice %1271 [0:1, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1275 = stablehlo.reshape %1274 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1276 = stablehlo.slice %1271 [1:2, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1277 = stablehlo.reshape %1276 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1278 = stablehlo.multiply %1275, %1277 : tensor<8192x!pf_babybear_mont>
    %1279 = stablehlo.slice %1271 [2:3, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1280 = stablehlo.reshape %1279 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1281 = stablehlo.subtract %1278, %1280 : tensor<8192x!pf_babybear_mont>
    %1282 = stablehlo.multiply %1273, %1281 : tensor<8192x!pf_babybear_mont>
    %cst_263 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1283 = stablehlo.reduce(%1282 init: %cst_263) applies stablehlo.add across dimensions = [0] : (tensor<8192x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_264 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1284 = stablehlo.broadcast_in_dim %cst_264, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8192x!pf_babybear_mont>
    %1285 = stablehlo.multiply %1238, %1284 : tensor<4x8192x!pf_babybear_mont>
    %1286 = stablehlo.add %1285, %1230 : tensor<4x8192x!pf_babybear_mont>
    %1287 = stablehlo.slice %1286 [3:4, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1288 = stablehlo.reshape %1287 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1289 = stablehlo.slice %1286 [0:1, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1290 = stablehlo.reshape %1289 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1291 = stablehlo.slice %1286 [1:2, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1292 = stablehlo.reshape %1291 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1293 = stablehlo.multiply %1290, %1292 : tensor<8192x!pf_babybear_mont>
    %1294 = stablehlo.slice %1286 [2:3, 0:8192] : (tensor<4x8192x!pf_babybear_mont>) -> tensor<1x8192x!pf_babybear_mont>
    %1295 = stablehlo.reshape %1294 : (tensor<1x8192x!pf_babybear_mont>) -> tensor<8192x!pf_babybear_mont>
    %1296 = stablehlo.subtract %1293, %1295 : tensor<8192x!pf_babybear_mont>
    %1297 = stablehlo.multiply %1288, %1296 : tensor<8192x!pf_babybear_mont>
    %cst_265 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1298 = stablehlo.reduce(%1297 init: %cst_265) applies stablehlo.add across dimensions = [0] : (tensor<8192x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1299 = stablehlo.broadcast_in_dim %1253, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1300 = stablehlo.broadcast_in_dim %1268, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1301 = stablehlo.broadcast_in_dim %1283, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1302 = stablehlo.broadcast_in_dim %1298, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1303 = stablehlo.concatenate %1299, %1300, %1301, %1302, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1304 = stablehlo.reshape %1220 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1305 = stablehlo.concatenate %cst, %1304, %cst_13, %1303, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_266 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1306 = stablehlo.broadcast_in_dim %cst_266, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1307 = stablehlo.slice %1305 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1308 = stablehlo.reshape %1307 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_267 = stablehlo.constant dense<0> : tensor<i32>
    %1309 = stablehlo.broadcast_in_dim %c_267, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1310 = "stablehlo.scatter"(%1306, %1309, %1308) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_268 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1311 = stablehlo.broadcast_in_dim %cst_268, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1312 = stablehlo.slice %1305 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_269 = stablehlo.constant dense<0> : tensor<i32>
    %1313 = stablehlo.broadcast_in_dim %c_269, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1314 = "stablehlo.scatter"(%1311, %1313, %1312) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_270 = stablehlo.constant dense<8> : tensor<i32>
    %1315 = stablehlo.broadcast_in_dim %c_270, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_271 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1316 = "stablehlo.scatter"(%1314, %1315, %cst_271) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_272 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1317 = stablehlo.broadcast_in_dim %cst_272, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1318 = stablehlo.concatenate %1317, %1316, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1319 = stablehlo.add %1310, %1318 : tensor<16x!pf_babybear_mont>
    %1320 = call @permute(%1319) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1321 = stablehlo.slice %1320 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1322 = stablehlo.reshape %1321 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1323 = stablehlo.broadcast_in_dim %1322, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8192x!pf_babybear_mont>
    %1324 = stablehlo.multiply %1238, %1323 : tensor<4x8192x!pf_babybear_mont>
    %1325 = stablehlo.add %1324, %1230 : tensor<4x8192x!pf_babybear_mont>
    %1326 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %c_273 = stablehlo.constant dense<2> : tensor<i32>
    %1327 = stablehlo.broadcast_in_dim %c_273, dims = [] : (tensor<i32>) -> tensor<4096xi32>
    %1328 = stablehlo.multiply %1327, %1326 : tensor<4096xi32>
    %c_274 = stablehlo.constant dense<0> : tensor<i32>
    %1329 = stablehlo.broadcast_in_dim %c_274, dims = [] : (tensor<i32>) -> tensor<4096xi32>
    %1330 = stablehlo.add %1329, %1328 : tensor<4096xi32>
    %1331 = stablehlo.broadcast_in_dim %1330, dims = [0] : (tensor<4096xi32>) -> tensor<4096x1xi32>
    %1332 = "stablehlo.gather"(%1325, %1331) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x8192x!pf_babybear_mont>, tensor<4096x1xi32>) -> tensor<4x4096x!pf_babybear_mont>
    %1333 = stablehlo.iota dim = 0 : tensor<4096xi32>
    %c_275 = stablehlo.constant dense<2> : tensor<i32>
    %1334 = stablehlo.broadcast_in_dim %c_275, dims = [] : (tensor<i32>) -> tensor<4096xi32>
    %1335 = stablehlo.multiply %1334, %1333 : tensor<4096xi32>
    %c_276 = stablehlo.constant dense<1> : tensor<i32>
    %1336 = stablehlo.broadcast_in_dim %c_276, dims = [] : (tensor<i32>) -> tensor<4096xi32>
    %1337 = stablehlo.add %1336, %1335 : tensor<4096xi32>
    %1338 = stablehlo.broadcast_in_dim %1337, dims = [0] : (tensor<4096xi32>) -> tensor<4096x1xi32>
    %1339 = "stablehlo.gather"(%1325, %1338) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x8192x!pf_babybear_mont>, tensor<4096x1xi32>) -> tensor<4x4096x!pf_babybear_mont>
    %1340 = stablehlo.subtract %1339, %1332 : tensor<4x4096x!pf_babybear_mont>
    %cst_277 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1341 = stablehlo.broadcast_in_dim %cst_277, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4096x!pf_babybear_mont>
    %1342 = stablehlo.multiply %1340, %1341 : tensor<4x4096x!pf_babybear_mont>
    %1343 = stablehlo.add %1342, %1332 : tensor<4x4096x!pf_babybear_mont>
    %1344 = stablehlo.slice %1343 [3:4, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1345 = stablehlo.reshape %1344 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1346 = stablehlo.slice %1343 [0:1, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1347 = stablehlo.reshape %1346 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1348 = stablehlo.slice %1343 [1:2, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1349 = stablehlo.reshape %1348 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1350 = stablehlo.multiply %1347, %1349 : tensor<4096x!pf_babybear_mont>
    %1351 = stablehlo.slice %1343 [2:3, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1352 = stablehlo.reshape %1351 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1353 = stablehlo.subtract %1350, %1352 : tensor<4096x!pf_babybear_mont>
    %1354 = stablehlo.multiply %1345, %1353 : tensor<4096x!pf_babybear_mont>
    %cst_278 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1355 = stablehlo.reduce(%1354 init: %cst_278) applies stablehlo.add across dimensions = [0] : (tensor<4096x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_279 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1356 = stablehlo.broadcast_in_dim %cst_279, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4096x!pf_babybear_mont>
    %1357 = stablehlo.multiply %1340, %1356 : tensor<4x4096x!pf_babybear_mont>
    %1358 = stablehlo.add %1357, %1332 : tensor<4x4096x!pf_babybear_mont>
    %1359 = stablehlo.slice %1358 [3:4, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1360 = stablehlo.reshape %1359 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1361 = stablehlo.slice %1358 [0:1, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1362 = stablehlo.reshape %1361 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1363 = stablehlo.slice %1358 [1:2, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1364 = stablehlo.reshape %1363 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1365 = stablehlo.multiply %1362, %1364 : tensor<4096x!pf_babybear_mont>
    %1366 = stablehlo.slice %1358 [2:3, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1367 = stablehlo.reshape %1366 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1368 = stablehlo.subtract %1365, %1367 : tensor<4096x!pf_babybear_mont>
    %1369 = stablehlo.multiply %1360, %1368 : tensor<4096x!pf_babybear_mont>
    %cst_280 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1370 = stablehlo.reduce(%1369 init: %cst_280) applies stablehlo.add across dimensions = [0] : (tensor<4096x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_281 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1371 = stablehlo.broadcast_in_dim %cst_281, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4096x!pf_babybear_mont>
    %1372 = stablehlo.multiply %1340, %1371 : tensor<4x4096x!pf_babybear_mont>
    %1373 = stablehlo.add %1372, %1332 : tensor<4x4096x!pf_babybear_mont>
    %1374 = stablehlo.slice %1373 [3:4, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1375 = stablehlo.reshape %1374 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1376 = stablehlo.slice %1373 [0:1, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1377 = stablehlo.reshape %1376 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1378 = stablehlo.slice %1373 [1:2, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1379 = stablehlo.reshape %1378 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1380 = stablehlo.multiply %1377, %1379 : tensor<4096x!pf_babybear_mont>
    %1381 = stablehlo.slice %1373 [2:3, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1382 = stablehlo.reshape %1381 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1383 = stablehlo.subtract %1380, %1382 : tensor<4096x!pf_babybear_mont>
    %1384 = stablehlo.multiply %1375, %1383 : tensor<4096x!pf_babybear_mont>
    %cst_282 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1385 = stablehlo.reduce(%1384 init: %cst_282) applies stablehlo.add across dimensions = [0] : (tensor<4096x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_283 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1386 = stablehlo.broadcast_in_dim %cst_283, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4096x!pf_babybear_mont>
    %1387 = stablehlo.multiply %1340, %1386 : tensor<4x4096x!pf_babybear_mont>
    %1388 = stablehlo.add %1387, %1332 : tensor<4x4096x!pf_babybear_mont>
    %1389 = stablehlo.slice %1388 [3:4, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1390 = stablehlo.reshape %1389 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1391 = stablehlo.slice %1388 [0:1, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1392 = stablehlo.reshape %1391 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1393 = stablehlo.slice %1388 [1:2, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1394 = stablehlo.reshape %1393 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1395 = stablehlo.multiply %1392, %1394 : tensor<4096x!pf_babybear_mont>
    %1396 = stablehlo.slice %1388 [2:3, 0:4096] : (tensor<4x4096x!pf_babybear_mont>) -> tensor<1x4096x!pf_babybear_mont>
    %1397 = stablehlo.reshape %1396 : (tensor<1x4096x!pf_babybear_mont>) -> tensor<4096x!pf_babybear_mont>
    %1398 = stablehlo.subtract %1395, %1397 : tensor<4096x!pf_babybear_mont>
    %1399 = stablehlo.multiply %1390, %1398 : tensor<4096x!pf_babybear_mont>
    %cst_284 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1400 = stablehlo.reduce(%1399 init: %cst_284) applies stablehlo.add across dimensions = [0] : (tensor<4096x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1401 = stablehlo.broadcast_in_dim %1355, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1402 = stablehlo.broadcast_in_dim %1370, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1403 = stablehlo.broadcast_in_dim %1385, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1404 = stablehlo.broadcast_in_dim %1400, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1405 = stablehlo.concatenate %1401, %1402, %1403, %1404, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1406 = stablehlo.reshape %1322 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1407 = stablehlo.concatenate %cst, %1406, %cst_14, %1405, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_285 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1408 = stablehlo.broadcast_in_dim %cst_285, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1409 = stablehlo.slice %1407 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1410 = stablehlo.reshape %1409 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_286 = stablehlo.constant dense<0> : tensor<i32>
    %1411 = stablehlo.broadcast_in_dim %c_286, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1412 = "stablehlo.scatter"(%1408, %1411, %1410) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_287 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1413 = stablehlo.broadcast_in_dim %cst_287, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1414 = stablehlo.slice %1407 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_288 = stablehlo.constant dense<0> : tensor<i32>
    %1415 = stablehlo.broadcast_in_dim %c_288, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1416 = "stablehlo.scatter"(%1413, %1415, %1414) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_289 = stablehlo.constant dense<8> : tensor<i32>
    %1417 = stablehlo.broadcast_in_dim %c_289, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_290 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1418 = "stablehlo.scatter"(%1416, %1417, %cst_290) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_291 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1419 = stablehlo.broadcast_in_dim %cst_291, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1420 = stablehlo.concatenate %1419, %1418, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1421 = stablehlo.add %1412, %1420 : tensor<16x!pf_babybear_mont>
    %1422 = call @permute(%1421) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1423 = stablehlo.slice %1422 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1424 = stablehlo.reshape %1423 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1425 = stablehlo.broadcast_in_dim %1424, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4096x!pf_babybear_mont>
    %1426 = stablehlo.multiply %1340, %1425 : tensor<4x4096x!pf_babybear_mont>
    %1427 = stablehlo.add %1426, %1332 : tensor<4x4096x!pf_babybear_mont>
    %1428 = stablehlo.iota dim = 0 : tensor<2048xi32>
    %c_292 = stablehlo.constant dense<2> : tensor<i32>
    %1429 = stablehlo.broadcast_in_dim %c_292, dims = [] : (tensor<i32>) -> tensor<2048xi32>
    %1430 = stablehlo.multiply %1429, %1428 : tensor<2048xi32>
    %c_293 = stablehlo.constant dense<0> : tensor<i32>
    %1431 = stablehlo.broadcast_in_dim %c_293, dims = [] : (tensor<i32>) -> tensor<2048xi32>
    %1432 = stablehlo.add %1431, %1430 : tensor<2048xi32>
    %1433 = stablehlo.broadcast_in_dim %1432, dims = [0] : (tensor<2048xi32>) -> tensor<2048x1xi32>
    %1434 = "stablehlo.gather"(%1427, %1433) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x4096x!pf_babybear_mont>, tensor<2048x1xi32>) -> tensor<4x2048x!pf_babybear_mont>
    %1435 = stablehlo.iota dim = 0 : tensor<2048xi32>
    %c_294 = stablehlo.constant dense<2> : tensor<i32>
    %1436 = stablehlo.broadcast_in_dim %c_294, dims = [] : (tensor<i32>) -> tensor<2048xi32>
    %1437 = stablehlo.multiply %1436, %1435 : tensor<2048xi32>
    %c_295 = stablehlo.constant dense<1> : tensor<i32>
    %1438 = stablehlo.broadcast_in_dim %c_295, dims = [] : (tensor<i32>) -> tensor<2048xi32>
    %1439 = stablehlo.add %1438, %1437 : tensor<2048xi32>
    %1440 = stablehlo.broadcast_in_dim %1439, dims = [0] : (tensor<2048xi32>) -> tensor<2048x1xi32>
    %1441 = "stablehlo.gather"(%1427, %1440) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x4096x!pf_babybear_mont>, tensor<2048x1xi32>) -> tensor<4x2048x!pf_babybear_mont>
    %1442 = stablehlo.subtract %1441, %1434 : tensor<4x2048x!pf_babybear_mont>
    %cst_296 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1443 = stablehlo.broadcast_in_dim %cst_296, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2048x!pf_babybear_mont>
    %1444 = stablehlo.multiply %1442, %1443 : tensor<4x2048x!pf_babybear_mont>
    %1445 = stablehlo.add %1444, %1434 : tensor<4x2048x!pf_babybear_mont>
    %1446 = stablehlo.slice %1445 [3:4, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1447 = stablehlo.reshape %1446 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1448 = stablehlo.slice %1445 [0:1, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1449 = stablehlo.reshape %1448 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1450 = stablehlo.slice %1445 [1:2, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1451 = stablehlo.reshape %1450 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1452 = stablehlo.multiply %1449, %1451 : tensor<2048x!pf_babybear_mont>
    %1453 = stablehlo.slice %1445 [2:3, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1454 = stablehlo.reshape %1453 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1455 = stablehlo.subtract %1452, %1454 : tensor<2048x!pf_babybear_mont>
    %1456 = stablehlo.multiply %1447, %1455 : tensor<2048x!pf_babybear_mont>
    %cst_297 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1457 = stablehlo.reduce(%1456 init: %cst_297) applies stablehlo.add across dimensions = [0] : (tensor<2048x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_298 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1458 = stablehlo.broadcast_in_dim %cst_298, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2048x!pf_babybear_mont>
    %1459 = stablehlo.multiply %1442, %1458 : tensor<4x2048x!pf_babybear_mont>
    %1460 = stablehlo.add %1459, %1434 : tensor<4x2048x!pf_babybear_mont>
    %1461 = stablehlo.slice %1460 [3:4, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1462 = stablehlo.reshape %1461 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1463 = stablehlo.slice %1460 [0:1, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1464 = stablehlo.reshape %1463 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1465 = stablehlo.slice %1460 [1:2, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1466 = stablehlo.reshape %1465 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1467 = stablehlo.multiply %1464, %1466 : tensor<2048x!pf_babybear_mont>
    %1468 = stablehlo.slice %1460 [2:3, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1469 = stablehlo.reshape %1468 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1470 = stablehlo.subtract %1467, %1469 : tensor<2048x!pf_babybear_mont>
    %1471 = stablehlo.multiply %1462, %1470 : tensor<2048x!pf_babybear_mont>
    %cst_299 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1472 = stablehlo.reduce(%1471 init: %cst_299) applies stablehlo.add across dimensions = [0] : (tensor<2048x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_300 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1473 = stablehlo.broadcast_in_dim %cst_300, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2048x!pf_babybear_mont>
    %1474 = stablehlo.multiply %1442, %1473 : tensor<4x2048x!pf_babybear_mont>
    %1475 = stablehlo.add %1474, %1434 : tensor<4x2048x!pf_babybear_mont>
    %1476 = stablehlo.slice %1475 [3:4, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1477 = stablehlo.reshape %1476 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1478 = stablehlo.slice %1475 [0:1, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1479 = stablehlo.reshape %1478 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1480 = stablehlo.slice %1475 [1:2, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1481 = stablehlo.reshape %1480 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1482 = stablehlo.multiply %1479, %1481 : tensor<2048x!pf_babybear_mont>
    %1483 = stablehlo.slice %1475 [2:3, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1484 = stablehlo.reshape %1483 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1485 = stablehlo.subtract %1482, %1484 : tensor<2048x!pf_babybear_mont>
    %1486 = stablehlo.multiply %1477, %1485 : tensor<2048x!pf_babybear_mont>
    %cst_301 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1487 = stablehlo.reduce(%1486 init: %cst_301) applies stablehlo.add across dimensions = [0] : (tensor<2048x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_302 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1488 = stablehlo.broadcast_in_dim %cst_302, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2048x!pf_babybear_mont>
    %1489 = stablehlo.multiply %1442, %1488 : tensor<4x2048x!pf_babybear_mont>
    %1490 = stablehlo.add %1489, %1434 : tensor<4x2048x!pf_babybear_mont>
    %1491 = stablehlo.slice %1490 [3:4, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1492 = stablehlo.reshape %1491 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1493 = stablehlo.slice %1490 [0:1, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1494 = stablehlo.reshape %1493 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1495 = stablehlo.slice %1490 [1:2, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1496 = stablehlo.reshape %1495 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1497 = stablehlo.multiply %1494, %1496 : tensor<2048x!pf_babybear_mont>
    %1498 = stablehlo.slice %1490 [2:3, 0:2048] : (tensor<4x2048x!pf_babybear_mont>) -> tensor<1x2048x!pf_babybear_mont>
    %1499 = stablehlo.reshape %1498 : (tensor<1x2048x!pf_babybear_mont>) -> tensor<2048x!pf_babybear_mont>
    %1500 = stablehlo.subtract %1497, %1499 : tensor<2048x!pf_babybear_mont>
    %1501 = stablehlo.multiply %1492, %1500 : tensor<2048x!pf_babybear_mont>
    %cst_303 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1502 = stablehlo.reduce(%1501 init: %cst_303) applies stablehlo.add across dimensions = [0] : (tensor<2048x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1503 = stablehlo.broadcast_in_dim %1457, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1504 = stablehlo.broadcast_in_dim %1472, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1505 = stablehlo.broadcast_in_dim %1487, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1506 = stablehlo.broadcast_in_dim %1502, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1507 = stablehlo.concatenate %1503, %1504, %1505, %1506, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1508 = stablehlo.reshape %1424 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1509 = stablehlo.concatenate %cst, %1508, %cst_15, %1507, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_304 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1510 = stablehlo.broadcast_in_dim %cst_304, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1511 = stablehlo.slice %1509 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1512 = stablehlo.reshape %1511 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_305 = stablehlo.constant dense<0> : tensor<i32>
    %1513 = stablehlo.broadcast_in_dim %c_305, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1514 = "stablehlo.scatter"(%1510, %1513, %1512) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_306 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1515 = stablehlo.broadcast_in_dim %cst_306, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1516 = stablehlo.slice %1509 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_307 = stablehlo.constant dense<0> : tensor<i32>
    %1517 = stablehlo.broadcast_in_dim %c_307, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1518 = "stablehlo.scatter"(%1515, %1517, %1516) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_308 = stablehlo.constant dense<8> : tensor<i32>
    %1519 = stablehlo.broadcast_in_dim %c_308, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_309 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1520 = "stablehlo.scatter"(%1518, %1519, %cst_309) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_310 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1521 = stablehlo.broadcast_in_dim %cst_310, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1522 = stablehlo.concatenate %1521, %1520, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1523 = stablehlo.add %1514, %1522 : tensor<16x!pf_babybear_mont>
    %1524 = call @permute(%1523) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1525 = stablehlo.slice %1524 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1526 = stablehlo.reshape %1525 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1527 = stablehlo.broadcast_in_dim %1526, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2048x!pf_babybear_mont>
    %1528 = stablehlo.multiply %1442, %1527 : tensor<4x2048x!pf_babybear_mont>
    %1529 = stablehlo.add %1528, %1434 : tensor<4x2048x!pf_babybear_mont>
    %1530 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %c_311 = stablehlo.constant dense<2> : tensor<i32>
    %1531 = stablehlo.broadcast_in_dim %c_311, dims = [] : (tensor<i32>) -> tensor<1024xi32>
    %1532 = stablehlo.multiply %1531, %1530 : tensor<1024xi32>
    %c_312 = stablehlo.constant dense<0> : tensor<i32>
    %1533 = stablehlo.broadcast_in_dim %c_312, dims = [] : (tensor<i32>) -> tensor<1024xi32>
    %1534 = stablehlo.add %1533, %1532 : tensor<1024xi32>
    %1535 = stablehlo.broadcast_in_dim %1534, dims = [0] : (tensor<1024xi32>) -> tensor<1024x1xi32>
    %1536 = "stablehlo.gather"(%1529, %1535) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x2048x!pf_babybear_mont>, tensor<1024x1xi32>) -> tensor<4x1024x!pf_babybear_mont>
    %1537 = stablehlo.iota dim = 0 : tensor<1024xi32>
    %c_313 = stablehlo.constant dense<2> : tensor<i32>
    %1538 = stablehlo.broadcast_in_dim %c_313, dims = [] : (tensor<i32>) -> tensor<1024xi32>
    %1539 = stablehlo.multiply %1538, %1537 : tensor<1024xi32>
    %c_314 = stablehlo.constant dense<1> : tensor<i32>
    %1540 = stablehlo.broadcast_in_dim %c_314, dims = [] : (tensor<i32>) -> tensor<1024xi32>
    %1541 = stablehlo.add %1540, %1539 : tensor<1024xi32>
    %1542 = stablehlo.broadcast_in_dim %1541, dims = [0] : (tensor<1024xi32>) -> tensor<1024x1xi32>
    %1543 = "stablehlo.gather"(%1529, %1542) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x2048x!pf_babybear_mont>, tensor<1024x1xi32>) -> tensor<4x1024x!pf_babybear_mont>
    %1544 = stablehlo.subtract %1543, %1536 : tensor<4x1024x!pf_babybear_mont>
    %cst_315 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1545 = stablehlo.broadcast_in_dim %cst_315, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1024x!pf_babybear_mont>
    %1546 = stablehlo.multiply %1544, %1545 : tensor<4x1024x!pf_babybear_mont>
    %1547 = stablehlo.add %1546, %1536 : tensor<4x1024x!pf_babybear_mont>
    %1548 = stablehlo.slice %1547 [3:4, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1549 = stablehlo.reshape %1548 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1550 = stablehlo.slice %1547 [0:1, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1551 = stablehlo.reshape %1550 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1552 = stablehlo.slice %1547 [1:2, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1553 = stablehlo.reshape %1552 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1554 = stablehlo.multiply %1551, %1553 : tensor<1024x!pf_babybear_mont>
    %1555 = stablehlo.slice %1547 [2:3, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1556 = stablehlo.reshape %1555 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1557 = stablehlo.subtract %1554, %1556 : tensor<1024x!pf_babybear_mont>
    %1558 = stablehlo.multiply %1549, %1557 : tensor<1024x!pf_babybear_mont>
    %cst_316 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1559 = stablehlo.reduce(%1558 init: %cst_316) applies stablehlo.add across dimensions = [0] : (tensor<1024x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_317 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1560 = stablehlo.broadcast_in_dim %cst_317, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1024x!pf_babybear_mont>
    %1561 = stablehlo.multiply %1544, %1560 : tensor<4x1024x!pf_babybear_mont>
    %1562 = stablehlo.add %1561, %1536 : tensor<4x1024x!pf_babybear_mont>
    %1563 = stablehlo.slice %1562 [3:4, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1564 = stablehlo.reshape %1563 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1565 = stablehlo.slice %1562 [0:1, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1566 = stablehlo.reshape %1565 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1567 = stablehlo.slice %1562 [1:2, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1568 = stablehlo.reshape %1567 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1569 = stablehlo.multiply %1566, %1568 : tensor<1024x!pf_babybear_mont>
    %1570 = stablehlo.slice %1562 [2:3, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1571 = stablehlo.reshape %1570 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1572 = stablehlo.subtract %1569, %1571 : tensor<1024x!pf_babybear_mont>
    %1573 = stablehlo.multiply %1564, %1572 : tensor<1024x!pf_babybear_mont>
    %cst_318 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1574 = stablehlo.reduce(%1573 init: %cst_318) applies stablehlo.add across dimensions = [0] : (tensor<1024x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_319 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1575 = stablehlo.broadcast_in_dim %cst_319, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1024x!pf_babybear_mont>
    %1576 = stablehlo.multiply %1544, %1575 : tensor<4x1024x!pf_babybear_mont>
    %1577 = stablehlo.add %1576, %1536 : tensor<4x1024x!pf_babybear_mont>
    %1578 = stablehlo.slice %1577 [3:4, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1579 = stablehlo.reshape %1578 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1580 = stablehlo.slice %1577 [0:1, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1581 = stablehlo.reshape %1580 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1582 = stablehlo.slice %1577 [1:2, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1583 = stablehlo.reshape %1582 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1584 = stablehlo.multiply %1581, %1583 : tensor<1024x!pf_babybear_mont>
    %1585 = stablehlo.slice %1577 [2:3, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1586 = stablehlo.reshape %1585 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1587 = stablehlo.subtract %1584, %1586 : tensor<1024x!pf_babybear_mont>
    %1588 = stablehlo.multiply %1579, %1587 : tensor<1024x!pf_babybear_mont>
    %cst_320 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1589 = stablehlo.reduce(%1588 init: %cst_320) applies stablehlo.add across dimensions = [0] : (tensor<1024x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_321 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1590 = stablehlo.broadcast_in_dim %cst_321, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1024x!pf_babybear_mont>
    %1591 = stablehlo.multiply %1544, %1590 : tensor<4x1024x!pf_babybear_mont>
    %1592 = stablehlo.add %1591, %1536 : tensor<4x1024x!pf_babybear_mont>
    %1593 = stablehlo.slice %1592 [3:4, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1594 = stablehlo.reshape %1593 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1595 = stablehlo.slice %1592 [0:1, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1596 = stablehlo.reshape %1595 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1597 = stablehlo.slice %1592 [1:2, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1598 = stablehlo.reshape %1597 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1599 = stablehlo.multiply %1596, %1598 : tensor<1024x!pf_babybear_mont>
    %1600 = stablehlo.slice %1592 [2:3, 0:1024] : (tensor<4x1024x!pf_babybear_mont>) -> tensor<1x1024x!pf_babybear_mont>
    %1601 = stablehlo.reshape %1600 : (tensor<1x1024x!pf_babybear_mont>) -> tensor<1024x!pf_babybear_mont>
    %1602 = stablehlo.subtract %1599, %1601 : tensor<1024x!pf_babybear_mont>
    %1603 = stablehlo.multiply %1594, %1602 : tensor<1024x!pf_babybear_mont>
    %cst_322 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1604 = stablehlo.reduce(%1603 init: %cst_322) applies stablehlo.add across dimensions = [0] : (tensor<1024x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1605 = stablehlo.broadcast_in_dim %1559, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1606 = stablehlo.broadcast_in_dim %1574, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1607 = stablehlo.broadcast_in_dim %1589, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1608 = stablehlo.broadcast_in_dim %1604, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1609 = stablehlo.concatenate %1605, %1606, %1607, %1608, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1610 = stablehlo.reshape %1526 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1611 = stablehlo.concatenate %cst, %1610, %cst_16, %1609, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_323 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1612 = stablehlo.broadcast_in_dim %cst_323, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1613 = stablehlo.slice %1611 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1614 = stablehlo.reshape %1613 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_324 = stablehlo.constant dense<0> : tensor<i32>
    %1615 = stablehlo.broadcast_in_dim %c_324, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1616 = "stablehlo.scatter"(%1612, %1615, %1614) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_325 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1617 = stablehlo.broadcast_in_dim %cst_325, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1618 = stablehlo.slice %1611 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_326 = stablehlo.constant dense<0> : tensor<i32>
    %1619 = stablehlo.broadcast_in_dim %c_326, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1620 = "stablehlo.scatter"(%1617, %1619, %1618) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_327 = stablehlo.constant dense<8> : tensor<i32>
    %1621 = stablehlo.broadcast_in_dim %c_327, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_328 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1622 = "stablehlo.scatter"(%1620, %1621, %cst_328) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_329 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1623 = stablehlo.broadcast_in_dim %cst_329, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1624 = stablehlo.concatenate %1623, %1622, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1625 = stablehlo.add %1616, %1624 : tensor<16x!pf_babybear_mont>
    %1626 = call @permute(%1625) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1627 = stablehlo.slice %1626 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1628 = stablehlo.reshape %1627 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1629 = stablehlo.broadcast_in_dim %1628, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1024x!pf_babybear_mont>
    %1630 = stablehlo.multiply %1544, %1629 : tensor<4x1024x!pf_babybear_mont>
    %1631 = stablehlo.add %1630, %1536 : tensor<4x1024x!pf_babybear_mont>
    %1632 = stablehlo.iota dim = 0 : tensor<512xi32>
    %c_330 = stablehlo.constant dense<2> : tensor<i32>
    %1633 = stablehlo.broadcast_in_dim %c_330, dims = [] : (tensor<i32>) -> tensor<512xi32>
    %1634 = stablehlo.multiply %1633, %1632 : tensor<512xi32>
    %c_331 = stablehlo.constant dense<0> : tensor<i32>
    %1635 = stablehlo.broadcast_in_dim %c_331, dims = [] : (tensor<i32>) -> tensor<512xi32>
    %1636 = stablehlo.add %1635, %1634 : tensor<512xi32>
    %1637 = stablehlo.broadcast_in_dim %1636, dims = [0] : (tensor<512xi32>) -> tensor<512x1xi32>
    %1638 = "stablehlo.gather"(%1631, %1637) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x1024x!pf_babybear_mont>, tensor<512x1xi32>) -> tensor<4x512x!pf_babybear_mont>
    %1639 = stablehlo.iota dim = 0 : tensor<512xi32>
    %c_332 = stablehlo.constant dense<2> : tensor<i32>
    %1640 = stablehlo.broadcast_in_dim %c_332, dims = [] : (tensor<i32>) -> tensor<512xi32>
    %1641 = stablehlo.multiply %1640, %1639 : tensor<512xi32>
    %c_333 = stablehlo.constant dense<1> : tensor<i32>
    %1642 = stablehlo.broadcast_in_dim %c_333, dims = [] : (tensor<i32>) -> tensor<512xi32>
    %1643 = stablehlo.add %1642, %1641 : tensor<512xi32>
    %1644 = stablehlo.broadcast_in_dim %1643, dims = [0] : (tensor<512xi32>) -> tensor<512x1xi32>
    %1645 = "stablehlo.gather"(%1631, %1644) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x1024x!pf_babybear_mont>, tensor<512x1xi32>) -> tensor<4x512x!pf_babybear_mont>
    %1646 = stablehlo.subtract %1645, %1638 : tensor<4x512x!pf_babybear_mont>
    %cst_334 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1647 = stablehlo.broadcast_in_dim %cst_334, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x512x!pf_babybear_mont>
    %1648 = stablehlo.multiply %1646, %1647 : tensor<4x512x!pf_babybear_mont>
    %1649 = stablehlo.add %1648, %1638 : tensor<4x512x!pf_babybear_mont>
    %1650 = stablehlo.slice %1649 [3:4, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1651 = stablehlo.reshape %1650 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1652 = stablehlo.slice %1649 [0:1, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1653 = stablehlo.reshape %1652 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1654 = stablehlo.slice %1649 [1:2, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1655 = stablehlo.reshape %1654 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1656 = stablehlo.multiply %1653, %1655 : tensor<512x!pf_babybear_mont>
    %1657 = stablehlo.slice %1649 [2:3, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1658 = stablehlo.reshape %1657 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1659 = stablehlo.subtract %1656, %1658 : tensor<512x!pf_babybear_mont>
    %1660 = stablehlo.multiply %1651, %1659 : tensor<512x!pf_babybear_mont>
    %cst_335 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1661 = stablehlo.reduce(%1660 init: %cst_335) applies stablehlo.add across dimensions = [0] : (tensor<512x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_336 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1662 = stablehlo.broadcast_in_dim %cst_336, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x512x!pf_babybear_mont>
    %1663 = stablehlo.multiply %1646, %1662 : tensor<4x512x!pf_babybear_mont>
    %1664 = stablehlo.add %1663, %1638 : tensor<4x512x!pf_babybear_mont>
    %1665 = stablehlo.slice %1664 [3:4, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1666 = stablehlo.reshape %1665 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1667 = stablehlo.slice %1664 [0:1, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1668 = stablehlo.reshape %1667 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1669 = stablehlo.slice %1664 [1:2, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1670 = stablehlo.reshape %1669 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1671 = stablehlo.multiply %1668, %1670 : tensor<512x!pf_babybear_mont>
    %1672 = stablehlo.slice %1664 [2:3, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1673 = stablehlo.reshape %1672 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1674 = stablehlo.subtract %1671, %1673 : tensor<512x!pf_babybear_mont>
    %1675 = stablehlo.multiply %1666, %1674 : tensor<512x!pf_babybear_mont>
    %cst_337 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1676 = stablehlo.reduce(%1675 init: %cst_337) applies stablehlo.add across dimensions = [0] : (tensor<512x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_338 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1677 = stablehlo.broadcast_in_dim %cst_338, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x512x!pf_babybear_mont>
    %1678 = stablehlo.multiply %1646, %1677 : tensor<4x512x!pf_babybear_mont>
    %1679 = stablehlo.add %1678, %1638 : tensor<4x512x!pf_babybear_mont>
    %1680 = stablehlo.slice %1679 [3:4, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1681 = stablehlo.reshape %1680 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1682 = stablehlo.slice %1679 [0:1, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1683 = stablehlo.reshape %1682 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1684 = stablehlo.slice %1679 [1:2, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1685 = stablehlo.reshape %1684 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1686 = stablehlo.multiply %1683, %1685 : tensor<512x!pf_babybear_mont>
    %1687 = stablehlo.slice %1679 [2:3, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1688 = stablehlo.reshape %1687 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1689 = stablehlo.subtract %1686, %1688 : tensor<512x!pf_babybear_mont>
    %1690 = stablehlo.multiply %1681, %1689 : tensor<512x!pf_babybear_mont>
    %cst_339 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1691 = stablehlo.reduce(%1690 init: %cst_339) applies stablehlo.add across dimensions = [0] : (tensor<512x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_340 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1692 = stablehlo.broadcast_in_dim %cst_340, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x512x!pf_babybear_mont>
    %1693 = stablehlo.multiply %1646, %1692 : tensor<4x512x!pf_babybear_mont>
    %1694 = stablehlo.add %1693, %1638 : tensor<4x512x!pf_babybear_mont>
    %1695 = stablehlo.slice %1694 [3:4, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1696 = stablehlo.reshape %1695 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1697 = stablehlo.slice %1694 [0:1, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1698 = stablehlo.reshape %1697 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1699 = stablehlo.slice %1694 [1:2, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1700 = stablehlo.reshape %1699 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1701 = stablehlo.multiply %1698, %1700 : tensor<512x!pf_babybear_mont>
    %1702 = stablehlo.slice %1694 [2:3, 0:512] : (tensor<4x512x!pf_babybear_mont>) -> tensor<1x512x!pf_babybear_mont>
    %1703 = stablehlo.reshape %1702 : (tensor<1x512x!pf_babybear_mont>) -> tensor<512x!pf_babybear_mont>
    %1704 = stablehlo.subtract %1701, %1703 : tensor<512x!pf_babybear_mont>
    %1705 = stablehlo.multiply %1696, %1704 : tensor<512x!pf_babybear_mont>
    %cst_341 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1706 = stablehlo.reduce(%1705 init: %cst_341) applies stablehlo.add across dimensions = [0] : (tensor<512x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1707 = stablehlo.broadcast_in_dim %1661, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1708 = stablehlo.broadcast_in_dim %1676, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1709 = stablehlo.broadcast_in_dim %1691, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1710 = stablehlo.broadcast_in_dim %1706, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1711 = stablehlo.concatenate %1707, %1708, %1709, %1710, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1712 = stablehlo.reshape %1628 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1713 = stablehlo.concatenate %cst, %1712, %cst_17, %1711, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_342 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1714 = stablehlo.broadcast_in_dim %cst_342, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1715 = stablehlo.slice %1713 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1716 = stablehlo.reshape %1715 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_343 = stablehlo.constant dense<0> : tensor<i32>
    %1717 = stablehlo.broadcast_in_dim %c_343, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1718 = "stablehlo.scatter"(%1714, %1717, %1716) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_344 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1719 = stablehlo.broadcast_in_dim %cst_344, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1720 = stablehlo.slice %1713 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_345 = stablehlo.constant dense<0> : tensor<i32>
    %1721 = stablehlo.broadcast_in_dim %c_345, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1722 = "stablehlo.scatter"(%1719, %1721, %1720) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_346 = stablehlo.constant dense<8> : tensor<i32>
    %1723 = stablehlo.broadcast_in_dim %c_346, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_347 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1724 = "stablehlo.scatter"(%1722, %1723, %cst_347) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_348 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1725 = stablehlo.broadcast_in_dim %cst_348, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1726 = stablehlo.concatenate %1725, %1724, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1727 = stablehlo.add %1718, %1726 : tensor<16x!pf_babybear_mont>
    %1728 = call @permute(%1727) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1729 = stablehlo.slice %1728 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1730 = stablehlo.reshape %1729 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1731 = stablehlo.broadcast_in_dim %1730, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x512x!pf_babybear_mont>
    %1732 = stablehlo.multiply %1646, %1731 : tensor<4x512x!pf_babybear_mont>
    %1733 = stablehlo.add %1732, %1638 : tensor<4x512x!pf_babybear_mont>
    %1734 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_349 = stablehlo.constant dense<2> : tensor<i32>
    %1735 = stablehlo.broadcast_in_dim %c_349, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %1736 = stablehlo.multiply %1735, %1734 : tensor<256xi32>
    %c_350 = stablehlo.constant dense<0> : tensor<i32>
    %1737 = stablehlo.broadcast_in_dim %c_350, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %1738 = stablehlo.add %1737, %1736 : tensor<256xi32>
    %1739 = stablehlo.broadcast_in_dim %1738, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %1740 = "stablehlo.gather"(%1733, %1739) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x512x!pf_babybear_mont>, tensor<256x1xi32>) -> tensor<4x256x!pf_babybear_mont>
    %1741 = stablehlo.iota dim = 0 : tensor<256xi32>
    %c_351 = stablehlo.constant dense<2> : tensor<i32>
    %1742 = stablehlo.broadcast_in_dim %c_351, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %1743 = stablehlo.multiply %1742, %1741 : tensor<256xi32>
    %c_352 = stablehlo.constant dense<1> : tensor<i32>
    %1744 = stablehlo.broadcast_in_dim %c_352, dims = [] : (tensor<i32>) -> tensor<256xi32>
    %1745 = stablehlo.add %1744, %1743 : tensor<256xi32>
    %1746 = stablehlo.broadcast_in_dim %1745, dims = [0] : (tensor<256xi32>) -> tensor<256x1xi32>
    %1747 = "stablehlo.gather"(%1733, %1746) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x512x!pf_babybear_mont>, tensor<256x1xi32>) -> tensor<4x256x!pf_babybear_mont>
    %1748 = stablehlo.subtract %1747, %1740 : tensor<4x256x!pf_babybear_mont>
    %cst_353 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1749 = stablehlo.broadcast_in_dim %cst_353, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x256x!pf_babybear_mont>
    %1750 = stablehlo.multiply %1748, %1749 : tensor<4x256x!pf_babybear_mont>
    %1751 = stablehlo.add %1750, %1740 : tensor<4x256x!pf_babybear_mont>
    %1752 = stablehlo.slice %1751 [3:4, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1753 = stablehlo.reshape %1752 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1754 = stablehlo.slice %1751 [0:1, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1755 = stablehlo.reshape %1754 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1756 = stablehlo.slice %1751 [1:2, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1757 = stablehlo.reshape %1756 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1758 = stablehlo.multiply %1755, %1757 : tensor<256x!pf_babybear_mont>
    %1759 = stablehlo.slice %1751 [2:3, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1760 = stablehlo.reshape %1759 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1761 = stablehlo.subtract %1758, %1760 : tensor<256x!pf_babybear_mont>
    %1762 = stablehlo.multiply %1753, %1761 : tensor<256x!pf_babybear_mont>
    %cst_354 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1763 = stablehlo.reduce(%1762 init: %cst_354) applies stablehlo.add across dimensions = [0] : (tensor<256x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_355 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1764 = stablehlo.broadcast_in_dim %cst_355, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x256x!pf_babybear_mont>
    %1765 = stablehlo.multiply %1748, %1764 : tensor<4x256x!pf_babybear_mont>
    %1766 = stablehlo.add %1765, %1740 : tensor<4x256x!pf_babybear_mont>
    %1767 = stablehlo.slice %1766 [3:4, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1768 = stablehlo.reshape %1767 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1769 = stablehlo.slice %1766 [0:1, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1770 = stablehlo.reshape %1769 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1771 = stablehlo.slice %1766 [1:2, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1772 = stablehlo.reshape %1771 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1773 = stablehlo.multiply %1770, %1772 : tensor<256x!pf_babybear_mont>
    %1774 = stablehlo.slice %1766 [2:3, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1775 = stablehlo.reshape %1774 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1776 = stablehlo.subtract %1773, %1775 : tensor<256x!pf_babybear_mont>
    %1777 = stablehlo.multiply %1768, %1776 : tensor<256x!pf_babybear_mont>
    %cst_356 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1778 = stablehlo.reduce(%1777 init: %cst_356) applies stablehlo.add across dimensions = [0] : (tensor<256x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_357 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1779 = stablehlo.broadcast_in_dim %cst_357, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x256x!pf_babybear_mont>
    %1780 = stablehlo.multiply %1748, %1779 : tensor<4x256x!pf_babybear_mont>
    %1781 = stablehlo.add %1780, %1740 : tensor<4x256x!pf_babybear_mont>
    %1782 = stablehlo.slice %1781 [3:4, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1783 = stablehlo.reshape %1782 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1784 = stablehlo.slice %1781 [0:1, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1785 = stablehlo.reshape %1784 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1786 = stablehlo.slice %1781 [1:2, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1787 = stablehlo.reshape %1786 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1788 = stablehlo.multiply %1785, %1787 : tensor<256x!pf_babybear_mont>
    %1789 = stablehlo.slice %1781 [2:3, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1790 = stablehlo.reshape %1789 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1791 = stablehlo.subtract %1788, %1790 : tensor<256x!pf_babybear_mont>
    %1792 = stablehlo.multiply %1783, %1791 : tensor<256x!pf_babybear_mont>
    %cst_358 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1793 = stablehlo.reduce(%1792 init: %cst_358) applies stablehlo.add across dimensions = [0] : (tensor<256x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_359 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1794 = stablehlo.broadcast_in_dim %cst_359, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x256x!pf_babybear_mont>
    %1795 = stablehlo.multiply %1748, %1794 : tensor<4x256x!pf_babybear_mont>
    %1796 = stablehlo.add %1795, %1740 : tensor<4x256x!pf_babybear_mont>
    %1797 = stablehlo.slice %1796 [3:4, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1798 = stablehlo.reshape %1797 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1799 = stablehlo.slice %1796 [0:1, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1800 = stablehlo.reshape %1799 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1801 = stablehlo.slice %1796 [1:2, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1802 = stablehlo.reshape %1801 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1803 = stablehlo.multiply %1800, %1802 : tensor<256x!pf_babybear_mont>
    %1804 = stablehlo.slice %1796 [2:3, 0:256] : (tensor<4x256x!pf_babybear_mont>) -> tensor<1x256x!pf_babybear_mont>
    %1805 = stablehlo.reshape %1804 : (tensor<1x256x!pf_babybear_mont>) -> tensor<256x!pf_babybear_mont>
    %1806 = stablehlo.subtract %1803, %1805 : tensor<256x!pf_babybear_mont>
    %1807 = stablehlo.multiply %1798, %1806 : tensor<256x!pf_babybear_mont>
    %cst_360 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1808 = stablehlo.reduce(%1807 init: %cst_360) applies stablehlo.add across dimensions = [0] : (tensor<256x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1809 = stablehlo.broadcast_in_dim %1763, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1810 = stablehlo.broadcast_in_dim %1778, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1811 = stablehlo.broadcast_in_dim %1793, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1812 = stablehlo.broadcast_in_dim %1808, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1813 = stablehlo.concatenate %1809, %1810, %1811, %1812, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1814 = stablehlo.reshape %1730 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1815 = stablehlo.concatenate %cst, %1814, %cst_18, %1813, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_361 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1816 = stablehlo.broadcast_in_dim %cst_361, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1817 = stablehlo.slice %1815 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1818 = stablehlo.reshape %1817 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_362 = stablehlo.constant dense<0> : tensor<i32>
    %1819 = stablehlo.broadcast_in_dim %c_362, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1820 = "stablehlo.scatter"(%1816, %1819, %1818) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_363 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1821 = stablehlo.broadcast_in_dim %cst_363, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1822 = stablehlo.slice %1815 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_364 = stablehlo.constant dense<0> : tensor<i32>
    %1823 = stablehlo.broadcast_in_dim %c_364, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1824 = "stablehlo.scatter"(%1821, %1823, %1822) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_365 = stablehlo.constant dense<8> : tensor<i32>
    %1825 = stablehlo.broadcast_in_dim %c_365, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_366 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1826 = "stablehlo.scatter"(%1824, %1825, %cst_366) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_367 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1827 = stablehlo.broadcast_in_dim %cst_367, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1828 = stablehlo.concatenate %1827, %1826, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1829 = stablehlo.add %1820, %1828 : tensor<16x!pf_babybear_mont>
    %1830 = call @permute(%1829) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1831 = stablehlo.slice %1830 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1832 = stablehlo.reshape %1831 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1833 = stablehlo.broadcast_in_dim %1832, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x256x!pf_babybear_mont>
    %1834 = stablehlo.multiply %1748, %1833 : tensor<4x256x!pf_babybear_mont>
    %1835 = stablehlo.add %1834, %1740 : tensor<4x256x!pf_babybear_mont>
    %1836 = stablehlo.iota dim = 0 : tensor<128xi32>
    %c_368 = stablehlo.constant dense<2> : tensor<i32>
    %1837 = stablehlo.broadcast_in_dim %c_368, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %1838 = stablehlo.multiply %1837, %1836 : tensor<128xi32>
    %c_369 = stablehlo.constant dense<0> : tensor<i32>
    %1839 = stablehlo.broadcast_in_dim %c_369, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %1840 = stablehlo.add %1839, %1838 : tensor<128xi32>
    %1841 = stablehlo.broadcast_in_dim %1840, dims = [0] : (tensor<128xi32>) -> tensor<128x1xi32>
    %1842 = "stablehlo.gather"(%1835, %1841) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x256x!pf_babybear_mont>, tensor<128x1xi32>) -> tensor<4x128x!pf_babybear_mont>
    %1843 = stablehlo.iota dim = 0 : tensor<128xi32>
    %c_370 = stablehlo.constant dense<2> : tensor<i32>
    %1844 = stablehlo.broadcast_in_dim %c_370, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %1845 = stablehlo.multiply %1844, %1843 : tensor<128xi32>
    %c_371 = stablehlo.constant dense<1> : tensor<i32>
    %1846 = stablehlo.broadcast_in_dim %c_371, dims = [] : (tensor<i32>) -> tensor<128xi32>
    %1847 = stablehlo.add %1846, %1845 : tensor<128xi32>
    %1848 = stablehlo.broadcast_in_dim %1847, dims = [0] : (tensor<128xi32>) -> tensor<128x1xi32>
    %1849 = "stablehlo.gather"(%1835, %1848) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x256x!pf_babybear_mont>, tensor<128x1xi32>) -> tensor<4x128x!pf_babybear_mont>
    %1850 = stablehlo.subtract %1849, %1842 : tensor<4x128x!pf_babybear_mont>
    %cst_372 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1851 = stablehlo.broadcast_in_dim %cst_372, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x128x!pf_babybear_mont>
    %1852 = stablehlo.multiply %1850, %1851 : tensor<4x128x!pf_babybear_mont>
    %1853 = stablehlo.add %1852, %1842 : tensor<4x128x!pf_babybear_mont>
    %1854 = stablehlo.slice %1853 [3:4, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1855 = stablehlo.reshape %1854 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1856 = stablehlo.slice %1853 [0:1, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1857 = stablehlo.reshape %1856 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1858 = stablehlo.slice %1853 [1:2, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1859 = stablehlo.reshape %1858 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1860 = stablehlo.multiply %1857, %1859 : tensor<128x!pf_babybear_mont>
    %1861 = stablehlo.slice %1853 [2:3, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1862 = stablehlo.reshape %1861 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1863 = stablehlo.subtract %1860, %1862 : tensor<128x!pf_babybear_mont>
    %1864 = stablehlo.multiply %1855, %1863 : tensor<128x!pf_babybear_mont>
    %cst_373 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1865 = stablehlo.reduce(%1864 init: %cst_373) applies stablehlo.add across dimensions = [0] : (tensor<128x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_374 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1866 = stablehlo.broadcast_in_dim %cst_374, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x128x!pf_babybear_mont>
    %1867 = stablehlo.multiply %1850, %1866 : tensor<4x128x!pf_babybear_mont>
    %1868 = stablehlo.add %1867, %1842 : tensor<4x128x!pf_babybear_mont>
    %1869 = stablehlo.slice %1868 [3:4, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1870 = stablehlo.reshape %1869 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1871 = stablehlo.slice %1868 [0:1, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1872 = stablehlo.reshape %1871 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1873 = stablehlo.slice %1868 [1:2, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1874 = stablehlo.reshape %1873 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1875 = stablehlo.multiply %1872, %1874 : tensor<128x!pf_babybear_mont>
    %1876 = stablehlo.slice %1868 [2:3, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1877 = stablehlo.reshape %1876 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1878 = stablehlo.subtract %1875, %1877 : tensor<128x!pf_babybear_mont>
    %1879 = stablehlo.multiply %1870, %1878 : tensor<128x!pf_babybear_mont>
    %cst_375 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1880 = stablehlo.reduce(%1879 init: %cst_375) applies stablehlo.add across dimensions = [0] : (tensor<128x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_376 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1881 = stablehlo.broadcast_in_dim %cst_376, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x128x!pf_babybear_mont>
    %1882 = stablehlo.multiply %1850, %1881 : tensor<4x128x!pf_babybear_mont>
    %1883 = stablehlo.add %1882, %1842 : tensor<4x128x!pf_babybear_mont>
    %1884 = stablehlo.slice %1883 [3:4, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1885 = stablehlo.reshape %1884 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1886 = stablehlo.slice %1883 [0:1, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1887 = stablehlo.reshape %1886 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1888 = stablehlo.slice %1883 [1:2, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1889 = stablehlo.reshape %1888 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1890 = stablehlo.multiply %1887, %1889 : tensor<128x!pf_babybear_mont>
    %1891 = stablehlo.slice %1883 [2:3, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1892 = stablehlo.reshape %1891 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1893 = stablehlo.subtract %1890, %1892 : tensor<128x!pf_babybear_mont>
    %1894 = stablehlo.multiply %1885, %1893 : tensor<128x!pf_babybear_mont>
    %cst_377 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1895 = stablehlo.reduce(%1894 init: %cst_377) applies stablehlo.add across dimensions = [0] : (tensor<128x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_378 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1896 = stablehlo.broadcast_in_dim %cst_378, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x128x!pf_babybear_mont>
    %1897 = stablehlo.multiply %1850, %1896 : tensor<4x128x!pf_babybear_mont>
    %1898 = stablehlo.add %1897, %1842 : tensor<4x128x!pf_babybear_mont>
    %1899 = stablehlo.slice %1898 [3:4, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1900 = stablehlo.reshape %1899 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1901 = stablehlo.slice %1898 [0:1, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1902 = stablehlo.reshape %1901 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1903 = stablehlo.slice %1898 [1:2, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1904 = stablehlo.reshape %1903 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1905 = stablehlo.multiply %1902, %1904 : tensor<128x!pf_babybear_mont>
    %1906 = stablehlo.slice %1898 [2:3, 0:128] : (tensor<4x128x!pf_babybear_mont>) -> tensor<1x128x!pf_babybear_mont>
    %1907 = stablehlo.reshape %1906 : (tensor<1x128x!pf_babybear_mont>) -> tensor<128x!pf_babybear_mont>
    %1908 = stablehlo.subtract %1905, %1907 : tensor<128x!pf_babybear_mont>
    %1909 = stablehlo.multiply %1900, %1908 : tensor<128x!pf_babybear_mont>
    %cst_379 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1910 = stablehlo.reduce(%1909 init: %cst_379) applies stablehlo.add across dimensions = [0] : (tensor<128x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1911 = stablehlo.broadcast_in_dim %1865, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1912 = stablehlo.broadcast_in_dim %1880, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1913 = stablehlo.broadcast_in_dim %1895, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1914 = stablehlo.broadcast_in_dim %1910, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1915 = stablehlo.concatenate %1911, %1912, %1913, %1914, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %1916 = stablehlo.reshape %1832 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1917 = stablehlo.concatenate %cst, %1916, %cst_19, %1915, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_380 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1918 = stablehlo.broadcast_in_dim %cst_380, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1919 = stablehlo.slice %1917 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1920 = stablehlo.reshape %1919 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_381 = stablehlo.constant dense<0> : tensor<i32>
    %1921 = stablehlo.broadcast_in_dim %c_381, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1922 = "stablehlo.scatter"(%1918, %1921, %1920) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_382 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1923 = stablehlo.broadcast_in_dim %cst_382, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1924 = stablehlo.slice %1917 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_383 = stablehlo.constant dense<0> : tensor<i32>
    %1925 = stablehlo.broadcast_in_dim %c_383, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %1926 = "stablehlo.scatter"(%1923, %1925, %1924) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_384 = stablehlo.constant dense<8> : tensor<i32>
    %1927 = stablehlo.broadcast_in_dim %c_384, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_385 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1928 = "stablehlo.scatter"(%1926, %1927, %cst_385) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_386 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1929 = stablehlo.broadcast_in_dim %cst_386, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1930 = stablehlo.concatenate %1929, %1928, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1931 = stablehlo.add %1922, %1930 : tensor<16x!pf_babybear_mont>
    %1932 = call @permute(%1931) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1933 = stablehlo.slice %1932 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1934 = stablehlo.reshape %1933 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %1935 = stablehlo.broadcast_in_dim %1934, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x128x!pf_babybear_mont>
    %1936 = stablehlo.multiply %1850, %1935 : tensor<4x128x!pf_babybear_mont>
    %1937 = stablehlo.add %1936, %1842 : tensor<4x128x!pf_babybear_mont>
    %1938 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_387 = stablehlo.constant dense<2> : tensor<i32>
    %1939 = stablehlo.broadcast_in_dim %c_387, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1940 = stablehlo.multiply %1939, %1938 : tensor<64xi32>
    %c_388 = stablehlo.constant dense<0> : tensor<i32>
    %1941 = stablehlo.broadcast_in_dim %c_388, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1942 = stablehlo.add %1941, %1940 : tensor<64xi32>
    %1943 = stablehlo.broadcast_in_dim %1942, dims = [0] : (tensor<64xi32>) -> tensor<64x1xi32>
    %1944 = "stablehlo.gather"(%1937, %1943) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x128x!pf_babybear_mont>, tensor<64x1xi32>) -> tensor<4x64x!pf_babybear_mont>
    %1945 = stablehlo.iota dim = 0 : tensor<64xi32>
    %c_389 = stablehlo.constant dense<2> : tensor<i32>
    %1946 = stablehlo.broadcast_in_dim %c_389, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1947 = stablehlo.multiply %1946, %1945 : tensor<64xi32>
    %c_390 = stablehlo.constant dense<1> : tensor<i32>
    %1948 = stablehlo.broadcast_in_dim %c_390, dims = [] : (tensor<i32>) -> tensor<64xi32>
    %1949 = stablehlo.add %1948, %1947 : tensor<64xi32>
    %1950 = stablehlo.broadcast_in_dim %1949, dims = [0] : (tensor<64xi32>) -> tensor<64x1xi32>
    %1951 = "stablehlo.gather"(%1937, %1950) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x128x!pf_babybear_mont>, tensor<64x1xi32>) -> tensor<4x64x!pf_babybear_mont>
    %1952 = stablehlo.subtract %1951, %1944 : tensor<4x64x!pf_babybear_mont>
    %cst_391 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1953 = stablehlo.broadcast_in_dim %cst_391, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x64x!pf_babybear_mont>
    %1954 = stablehlo.multiply %1952, %1953 : tensor<4x64x!pf_babybear_mont>
    %1955 = stablehlo.add %1954, %1944 : tensor<4x64x!pf_babybear_mont>
    %1956 = stablehlo.slice %1955 [3:4, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1957 = stablehlo.reshape %1956 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1958 = stablehlo.slice %1955 [0:1, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1959 = stablehlo.reshape %1958 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1960 = stablehlo.slice %1955 [1:2, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1961 = stablehlo.reshape %1960 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1962 = stablehlo.multiply %1959, %1961 : tensor<64x!pf_babybear_mont>
    %1963 = stablehlo.slice %1955 [2:3, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1964 = stablehlo.reshape %1963 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1965 = stablehlo.subtract %1962, %1964 : tensor<64x!pf_babybear_mont>
    %1966 = stablehlo.multiply %1957, %1965 : tensor<64x!pf_babybear_mont>
    %cst_392 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1967 = stablehlo.reduce(%1966 init: %cst_392) applies stablehlo.add across dimensions = [0] : (tensor<64x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_393 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1968 = stablehlo.broadcast_in_dim %cst_393, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x64x!pf_babybear_mont>
    %1969 = stablehlo.multiply %1952, %1968 : tensor<4x64x!pf_babybear_mont>
    %1970 = stablehlo.add %1969, %1944 : tensor<4x64x!pf_babybear_mont>
    %1971 = stablehlo.slice %1970 [3:4, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1972 = stablehlo.reshape %1971 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1973 = stablehlo.slice %1970 [0:1, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1974 = stablehlo.reshape %1973 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1975 = stablehlo.slice %1970 [1:2, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1976 = stablehlo.reshape %1975 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1977 = stablehlo.multiply %1974, %1976 : tensor<64x!pf_babybear_mont>
    %1978 = stablehlo.slice %1970 [2:3, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1979 = stablehlo.reshape %1978 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1980 = stablehlo.subtract %1977, %1979 : tensor<64x!pf_babybear_mont>
    %1981 = stablehlo.multiply %1972, %1980 : tensor<64x!pf_babybear_mont>
    %cst_394 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1982 = stablehlo.reduce(%1981 init: %cst_394) applies stablehlo.add across dimensions = [0] : (tensor<64x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_395 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1983 = stablehlo.broadcast_in_dim %cst_395, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x64x!pf_babybear_mont>
    %1984 = stablehlo.multiply %1952, %1983 : tensor<4x64x!pf_babybear_mont>
    %1985 = stablehlo.add %1984, %1944 : tensor<4x64x!pf_babybear_mont>
    %1986 = stablehlo.slice %1985 [3:4, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1987 = stablehlo.reshape %1986 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1988 = stablehlo.slice %1985 [0:1, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1989 = stablehlo.reshape %1988 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1990 = stablehlo.slice %1985 [1:2, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1991 = stablehlo.reshape %1990 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1992 = stablehlo.multiply %1989, %1991 : tensor<64x!pf_babybear_mont>
    %1993 = stablehlo.slice %1985 [2:3, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %1994 = stablehlo.reshape %1993 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %1995 = stablehlo.subtract %1992, %1994 : tensor<64x!pf_babybear_mont>
    %1996 = stablehlo.multiply %1987, %1995 : tensor<64x!pf_babybear_mont>
    %cst_396 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1997 = stablehlo.reduce(%1996 init: %cst_396) applies stablehlo.add across dimensions = [0] : (tensor<64x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_397 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1998 = stablehlo.broadcast_in_dim %cst_397, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x64x!pf_babybear_mont>
    %1999 = stablehlo.multiply %1952, %1998 : tensor<4x64x!pf_babybear_mont>
    %2000 = stablehlo.add %1999, %1944 : tensor<4x64x!pf_babybear_mont>
    %2001 = stablehlo.slice %2000 [3:4, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %2002 = stablehlo.reshape %2001 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %2003 = stablehlo.slice %2000 [0:1, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %2004 = stablehlo.reshape %2003 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %2005 = stablehlo.slice %2000 [1:2, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %2006 = stablehlo.reshape %2005 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %2007 = stablehlo.multiply %2004, %2006 : tensor<64x!pf_babybear_mont>
    %2008 = stablehlo.slice %2000 [2:3, 0:64] : (tensor<4x64x!pf_babybear_mont>) -> tensor<1x64x!pf_babybear_mont>
    %2009 = stablehlo.reshape %2008 : (tensor<1x64x!pf_babybear_mont>) -> tensor<64x!pf_babybear_mont>
    %2010 = stablehlo.subtract %2007, %2009 : tensor<64x!pf_babybear_mont>
    %2011 = stablehlo.multiply %2002, %2010 : tensor<64x!pf_babybear_mont>
    %cst_398 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2012 = stablehlo.reduce(%2011 init: %cst_398) applies stablehlo.add across dimensions = [0] : (tensor<64x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2013 = stablehlo.broadcast_in_dim %1967, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2014 = stablehlo.broadcast_in_dim %1982, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2015 = stablehlo.broadcast_in_dim %1997, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2016 = stablehlo.broadcast_in_dim %2012, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2017 = stablehlo.concatenate %2013, %2014, %2015, %2016, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2018 = stablehlo.reshape %1934 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2019 = stablehlo.concatenate %cst, %2018, %cst_20, %2017, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_399 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2020 = stablehlo.broadcast_in_dim %cst_399, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2021 = stablehlo.slice %2019 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2022 = stablehlo.reshape %2021 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_400 = stablehlo.constant dense<0> : tensor<i32>
    %2023 = stablehlo.broadcast_in_dim %c_400, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2024 = "stablehlo.scatter"(%2020, %2023, %2022) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_401 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2025 = stablehlo.broadcast_in_dim %cst_401, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %2026 = stablehlo.slice %2019 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_402 = stablehlo.constant dense<0> : tensor<i32>
    %2027 = stablehlo.broadcast_in_dim %c_402, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2028 = "stablehlo.scatter"(%2025, %2027, %2026) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_403 = stablehlo.constant dense<8> : tensor<i32>
    %2029 = stablehlo.broadcast_in_dim %c_403, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_404 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2030 = "stablehlo.scatter"(%2028, %2029, %cst_404) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_405 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2031 = stablehlo.broadcast_in_dim %cst_405, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2032 = stablehlo.concatenate %2031, %2030, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2033 = stablehlo.add %2024, %2032 : tensor<16x!pf_babybear_mont>
    %2034 = call @permute(%2033) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2035 = stablehlo.slice %2034 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2036 = stablehlo.reshape %2035 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2037 = stablehlo.broadcast_in_dim %2036, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x64x!pf_babybear_mont>
    %2038 = stablehlo.multiply %1952, %2037 : tensor<4x64x!pf_babybear_mont>
    %2039 = stablehlo.add %2038, %1944 : tensor<4x64x!pf_babybear_mont>
    %2040 = stablehlo.iota dim = 0 : tensor<32xi32>
    %c_406 = stablehlo.constant dense<2> : tensor<i32>
    %2041 = stablehlo.broadcast_in_dim %c_406, dims = [] : (tensor<i32>) -> tensor<32xi32>
    %2042 = stablehlo.multiply %2041, %2040 : tensor<32xi32>
    %c_407 = stablehlo.constant dense<0> : tensor<i32>
    %2043 = stablehlo.broadcast_in_dim %c_407, dims = [] : (tensor<i32>) -> tensor<32xi32>
    %2044 = stablehlo.add %2043, %2042 : tensor<32xi32>
    %2045 = stablehlo.broadcast_in_dim %2044, dims = [0] : (tensor<32xi32>) -> tensor<32x1xi32>
    %2046 = "stablehlo.gather"(%2039, %2045) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x64x!pf_babybear_mont>, tensor<32x1xi32>) -> tensor<4x32x!pf_babybear_mont>
    %2047 = stablehlo.iota dim = 0 : tensor<32xi32>
    %c_408 = stablehlo.constant dense<2> : tensor<i32>
    %2048 = stablehlo.broadcast_in_dim %c_408, dims = [] : (tensor<i32>) -> tensor<32xi32>
    %2049 = stablehlo.multiply %2048, %2047 : tensor<32xi32>
    %c_409 = stablehlo.constant dense<1> : tensor<i32>
    %2050 = stablehlo.broadcast_in_dim %c_409, dims = [] : (tensor<i32>) -> tensor<32xi32>
    %2051 = stablehlo.add %2050, %2049 : tensor<32xi32>
    %2052 = stablehlo.broadcast_in_dim %2051, dims = [0] : (tensor<32xi32>) -> tensor<32x1xi32>
    %2053 = "stablehlo.gather"(%2039, %2052) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x64x!pf_babybear_mont>, tensor<32x1xi32>) -> tensor<4x32x!pf_babybear_mont>
    %2054 = stablehlo.subtract %2053, %2046 : tensor<4x32x!pf_babybear_mont>
    %cst_410 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2055 = stablehlo.broadcast_in_dim %cst_410, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32x!pf_babybear_mont>
    %2056 = stablehlo.multiply %2054, %2055 : tensor<4x32x!pf_babybear_mont>
    %2057 = stablehlo.add %2056, %2046 : tensor<4x32x!pf_babybear_mont>
    %2058 = stablehlo.slice %2057 [3:4, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2059 = stablehlo.reshape %2058 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2060 = stablehlo.slice %2057 [0:1, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2061 = stablehlo.reshape %2060 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2062 = stablehlo.slice %2057 [1:2, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2063 = stablehlo.reshape %2062 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2064 = stablehlo.multiply %2061, %2063 : tensor<32x!pf_babybear_mont>
    %2065 = stablehlo.slice %2057 [2:3, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2066 = stablehlo.reshape %2065 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2067 = stablehlo.subtract %2064, %2066 : tensor<32x!pf_babybear_mont>
    %2068 = stablehlo.multiply %2059, %2067 : tensor<32x!pf_babybear_mont>
    %cst_411 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2069 = stablehlo.reduce(%2068 init: %cst_411) applies stablehlo.add across dimensions = [0] : (tensor<32x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_412 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2070 = stablehlo.broadcast_in_dim %cst_412, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32x!pf_babybear_mont>
    %2071 = stablehlo.multiply %2054, %2070 : tensor<4x32x!pf_babybear_mont>
    %2072 = stablehlo.add %2071, %2046 : tensor<4x32x!pf_babybear_mont>
    %2073 = stablehlo.slice %2072 [3:4, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2074 = stablehlo.reshape %2073 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2075 = stablehlo.slice %2072 [0:1, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2076 = stablehlo.reshape %2075 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2077 = stablehlo.slice %2072 [1:2, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2078 = stablehlo.reshape %2077 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2079 = stablehlo.multiply %2076, %2078 : tensor<32x!pf_babybear_mont>
    %2080 = stablehlo.slice %2072 [2:3, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2081 = stablehlo.reshape %2080 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2082 = stablehlo.subtract %2079, %2081 : tensor<32x!pf_babybear_mont>
    %2083 = stablehlo.multiply %2074, %2082 : tensor<32x!pf_babybear_mont>
    %cst_413 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2084 = stablehlo.reduce(%2083 init: %cst_413) applies stablehlo.add across dimensions = [0] : (tensor<32x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_414 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2085 = stablehlo.broadcast_in_dim %cst_414, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32x!pf_babybear_mont>
    %2086 = stablehlo.multiply %2054, %2085 : tensor<4x32x!pf_babybear_mont>
    %2087 = stablehlo.add %2086, %2046 : tensor<4x32x!pf_babybear_mont>
    %2088 = stablehlo.slice %2087 [3:4, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2089 = stablehlo.reshape %2088 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2090 = stablehlo.slice %2087 [0:1, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2091 = stablehlo.reshape %2090 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2092 = stablehlo.slice %2087 [1:2, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2093 = stablehlo.reshape %2092 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2094 = stablehlo.multiply %2091, %2093 : tensor<32x!pf_babybear_mont>
    %2095 = stablehlo.slice %2087 [2:3, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2096 = stablehlo.reshape %2095 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2097 = stablehlo.subtract %2094, %2096 : tensor<32x!pf_babybear_mont>
    %2098 = stablehlo.multiply %2089, %2097 : tensor<32x!pf_babybear_mont>
    %cst_415 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2099 = stablehlo.reduce(%2098 init: %cst_415) applies stablehlo.add across dimensions = [0] : (tensor<32x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_416 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2100 = stablehlo.broadcast_in_dim %cst_416, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32x!pf_babybear_mont>
    %2101 = stablehlo.multiply %2054, %2100 : tensor<4x32x!pf_babybear_mont>
    %2102 = stablehlo.add %2101, %2046 : tensor<4x32x!pf_babybear_mont>
    %2103 = stablehlo.slice %2102 [3:4, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2104 = stablehlo.reshape %2103 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2105 = stablehlo.slice %2102 [0:1, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2106 = stablehlo.reshape %2105 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2107 = stablehlo.slice %2102 [1:2, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2108 = stablehlo.reshape %2107 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2109 = stablehlo.multiply %2106, %2108 : tensor<32x!pf_babybear_mont>
    %2110 = stablehlo.slice %2102 [2:3, 0:32] : (tensor<4x32x!pf_babybear_mont>) -> tensor<1x32x!pf_babybear_mont>
    %2111 = stablehlo.reshape %2110 : (tensor<1x32x!pf_babybear_mont>) -> tensor<32x!pf_babybear_mont>
    %2112 = stablehlo.subtract %2109, %2111 : tensor<32x!pf_babybear_mont>
    %2113 = stablehlo.multiply %2104, %2112 : tensor<32x!pf_babybear_mont>
    %cst_417 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2114 = stablehlo.reduce(%2113 init: %cst_417) applies stablehlo.add across dimensions = [0] : (tensor<32x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2115 = stablehlo.broadcast_in_dim %2069, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2116 = stablehlo.broadcast_in_dim %2084, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2117 = stablehlo.broadcast_in_dim %2099, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2118 = stablehlo.broadcast_in_dim %2114, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2119 = stablehlo.concatenate %2115, %2116, %2117, %2118, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2120 = stablehlo.reshape %2036 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2121 = stablehlo.concatenate %cst, %2120, %cst_21, %2119, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_418 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2122 = stablehlo.broadcast_in_dim %cst_418, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2123 = stablehlo.slice %2121 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2124 = stablehlo.reshape %2123 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_419 = stablehlo.constant dense<0> : tensor<i32>
    %2125 = stablehlo.broadcast_in_dim %c_419, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2126 = "stablehlo.scatter"(%2122, %2125, %2124) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_420 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2127 = stablehlo.broadcast_in_dim %cst_420, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %2128 = stablehlo.slice %2121 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_421 = stablehlo.constant dense<0> : tensor<i32>
    %2129 = stablehlo.broadcast_in_dim %c_421, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2130 = "stablehlo.scatter"(%2127, %2129, %2128) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_422 = stablehlo.constant dense<8> : tensor<i32>
    %2131 = stablehlo.broadcast_in_dim %c_422, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_423 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2132 = "stablehlo.scatter"(%2130, %2131, %cst_423) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_424 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2133 = stablehlo.broadcast_in_dim %cst_424, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2134 = stablehlo.concatenate %2133, %2132, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2135 = stablehlo.add %2126, %2134 : tensor<16x!pf_babybear_mont>
    %2136 = call @permute(%2135) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2137 = stablehlo.slice %2136 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2138 = stablehlo.reshape %2137 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2139 = stablehlo.broadcast_in_dim %2138, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x32x!pf_babybear_mont>
    %2140 = stablehlo.multiply %2054, %2139 : tensor<4x32x!pf_babybear_mont>
    %2141 = stablehlo.add %2140, %2046 : tensor<4x32x!pf_babybear_mont>
    %2142 = stablehlo.iota dim = 0 : tensor<16xi32>
    %c_425 = stablehlo.constant dense<2> : tensor<i32>
    %2143 = stablehlo.broadcast_in_dim %c_425, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2144 = stablehlo.multiply %2143, %2142 : tensor<16xi32>
    %c_426 = stablehlo.constant dense<0> : tensor<i32>
    %2145 = stablehlo.broadcast_in_dim %c_426, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2146 = stablehlo.add %2145, %2144 : tensor<16xi32>
    %2147 = stablehlo.broadcast_in_dim %2146, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %2148 = "stablehlo.gather"(%2141, %2147) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x32x!pf_babybear_mont>, tensor<16x1xi32>) -> tensor<4x16x!pf_babybear_mont>
    %2149 = stablehlo.iota dim = 0 : tensor<16xi32>
    %c_427 = stablehlo.constant dense<2> : tensor<i32>
    %2150 = stablehlo.broadcast_in_dim %c_427, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2151 = stablehlo.multiply %2150, %2149 : tensor<16xi32>
    %c_428 = stablehlo.constant dense<1> : tensor<i32>
    %2152 = stablehlo.broadcast_in_dim %c_428, dims = [] : (tensor<i32>) -> tensor<16xi32>
    %2153 = stablehlo.add %2152, %2151 : tensor<16xi32>
    %2154 = stablehlo.broadcast_in_dim %2153, dims = [0] : (tensor<16xi32>) -> tensor<16x1xi32>
    %2155 = "stablehlo.gather"(%2141, %2154) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x32x!pf_babybear_mont>, tensor<16x1xi32>) -> tensor<4x16x!pf_babybear_mont>
    %2156 = stablehlo.subtract %2155, %2148 : tensor<4x16x!pf_babybear_mont>
    %cst_429 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2157 = stablehlo.broadcast_in_dim %cst_429, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16x!pf_babybear_mont>
    %2158 = stablehlo.multiply %2156, %2157 : tensor<4x16x!pf_babybear_mont>
    %2159 = stablehlo.add %2158, %2148 : tensor<4x16x!pf_babybear_mont>
    %2160 = stablehlo.slice %2159 [3:4, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2161 = stablehlo.reshape %2160 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2162 = stablehlo.slice %2159 [0:1, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2163 = stablehlo.reshape %2162 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2164 = stablehlo.slice %2159 [1:2, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2165 = stablehlo.reshape %2164 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2166 = stablehlo.multiply %2163, %2165 : tensor<16x!pf_babybear_mont>
    %2167 = stablehlo.slice %2159 [2:3, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2168 = stablehlo.reshape %2167 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2169 = stablehlo.subtract %2166, %2168 : tensor<16x!pf_babybear_mont>
    %2170 = stablehlo.multiply %2161, %2169 : tensor<16x!pf_babybear_mont>
    %cst_430 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2171 = stablehlo.reduce(%2170 init: %cst_430) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_431 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2172 = stablehlo.broadcast_in_dim %cst_431, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16x!pf_babybear_mont>
    %2173 = stablehlo.multiply %2156, %2172 : tensor<4x16x!pf_babybear_mont>
    %2174 = stablehlo.add %2173, %2148 : tensor<4x16x!pf_babybear_mont>
    %2175 = stablehlo.slice %2174 [3:4, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2176 = stablehlo.reshape %2175 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2177 = stablehlo.slice %2174 [0:1, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2178 = stablehlo.reshape %2177 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2179 = stablehlo.slice %2174 [1:2, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2180 = stablehlo.reshape %2179 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2181 = stablehlo.multiply %2178, %2180 : tensor<16x!pf_babybear_mont>
    %2182 = stablehlo.slice %2174 [2:3, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2183 = stablehlo.reshape %2182 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2184 = stablehlo.subtract %2181, %2183 : tensor<16x!pf_babybear_mont>
    %2185 = stablehlo.multiply %2176, %2184 : tensor<16x!pf_babybear_mont>
    %cst_432 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2186 = stablehlo.reduce(%2185 init: %cst_432) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_433 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2187 = stablehlo.broadcast_in_dim %cst_433, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16x!pf_babybear_mont>
    %2188 = stablehlo.multiply %2156, %2187 : tensor<4x16x!pf_babybear_mont>
    %2189 = stablehlo.add %2188, %2148 : tensor<4x16x!pf_babybear_mont>
    %2190 = stablehlo.slice %2189 [3:4, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2191 = stablehlo.reshape %2190 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2192 = stablehlo.slice %2189 [0:1, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2193 = stablehlo.reshape %2192 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2194 = stablehlo.slice %2189 [1:2, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2195 = stablehlo.reshape %2194 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2196 = stablehlo.multiply %2193, %2195 : tensor<16x!pf_babybear_mont>
    %2197 = stablehlo.slice %2189 [2:3, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2198 = stablehlo.reshape %2197 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2199 = stablehlo.subtract %2196, %2198 : tensor<16x!pf_babybear_mont>
    %2200 = stablehlo.multiply %2191, %2199 : tensor<16x!pf_babybear_mont>
    %cst_434 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2201 = stablehlo.reduce(%2200 init: %cst_434) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_435 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2202 = stablehlo.broadcast_in_dim %cst_435, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16x!pf_babybear_mont>
    %2203 = stablehlo.multiply %2156, %2202 : tensor<4x16x!pf_babybear_mont>
    %2204 = stablehlo.add %2203, %2148 : tensor<4x16x!pf_babybear_mont>
    %2205 = stablehlo.slice %2204 [3:4, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2206 = stablehlo.reshape %2205 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2207 = stablehlo.slice %2204 [0:1, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2208 = stablehlo.reshape %2207 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2209 = stablehlo.slice %2204 [1:2, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2210 = stablehlo.reshape %2209 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2211 = stablehlo.multiply %2208, %2210 : tensor<16x!pf_babybear_mont>
    %2212 = stablehlo.slice %2204 [2:3, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %2213 = stablehlo.reshape %2212 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2214 = stablehlo.subtract %2211, %2213 : tensor<16x!pf_babybear_mont>
    %2215 = stablehlo.multiply %2206, %2214 : tensor<16x!pf_babybear_mont>
    %cst_436 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2216 = stablehlo.reduce(%2215 init: %cst_436) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2217 = stablehlo.broadcast_in_dim %2171, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2218 = stablehlo.broadcast_in_dim %2186, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2219 = stablehlo.broadcast_in_dim %2201, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2220 = stablehlo.broadcast_in_dim %2216, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2221 = stablehlo.concatenate %2217, %2218, %2219, %2220, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2222 = stablehlo.reshape %2138 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2223 = stablehlo.concatenate %cst, %2222, %cst_22, %2221, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_437 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2224 = stablehlo.broadcast_in_dim %cst_437, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2225 = stablehlo.slice %2223 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2226 = stablehlo.reshape %2225 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_438 = stablehlo.constant dense<0> : tensor<i32>
    %2227 = stablehlo.broadcast_in_dim %c_438, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2228 = "stablehlo.scatter"(%2224, %2227, %2226) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_439 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2229 = stablehlo.broadcast_in_dim %cst_439, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %2230 = stablehlo.slice %2223 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_440 = stablehlo.constant dense<0> : tensor<i32>
    %2231 = stablehlo.broadcast_in_dim %c_440, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2232 = "stablehlo.scatter"(%2229, %2231, %2230) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_441 = stablehlo.constant dense<8> : tensor<i32>
    %2233 = stablehlo.broadcast_in_dim %c_441, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_442 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2234 = "stablehlo.scatter"(%2232, %2233, %cst_442) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_443 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2235 = stablehlo.broadcast_in_dim %cst_443, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2236 = stablehlo.concatenate %2235, %2234, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2237 = stablehlo.add %2228, %2236 : tensor<16x!pf_babybear_mont>
    %2238 = call @permute(%2237) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2239 = stablehlo.slice %2238 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2240 = stablehlo.reshape %2239 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2241 = stablehlo.broadcast_in_dim %2240, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x16x!pf_babybear_mont>
    %2242 = stablehlo.multiply %2156, %2241 : tensor<4x16x!pf_babybear_mont>
    %2243 = stablehlo.add %2242, %2148 : tensor<4x16x!pf_babybear_mont>
    %2244 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c_444 = stablehlo.constant dense<2> : tensor<i32>
    %2245 = stablehlo.broadcast_in_dim %c_444, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2246 = stablehlo.multiply %2245, %2244 : tensor<8xi32>
    %c_445 = stablehlo.constant dense<0> : tensor<i32>
    %2247 = stablehlo.broadcast_in_dim %c_445, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2248 = stablehlo.add %2247, %2246 : tensor<8xi32>
    %2249 = stablehlo.broadcast_in_dim %2248, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %2250 = "stablehlo.gather"(%2243, %2249) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x16x!pf_babybear_mont>, tensor<8x1xi32>) -> tensor<4x8x!pf_babybear_mont>
    %2251 = stablehlo.iota dim = 0 : tensor<8xi32>
    %c_446 = stablehlo.constant dense<2> : tensor<i32>
    %2252 = stablehlo.broadcast_in_dim %c_446, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2253 = stablehlo.multiply %2252, %2251 : tensor<8xi32>
    %c_447 = stablehlo.constant dense<1> : tensor<i32>
    %2254 = stablehlo.broadcast_in_dim %c_447, dims = [] : (tensor<i32>) -> tensor<8xi32>
    %2255 = stablehlo.add %2254, %2253 : tensor<8xi32>
    %2256 = stablehlo.broadcast_in_dim %2255, dims = [0] : (tensor<8xi32>) -> tensor<8x1xi32>
    %2257 = "stablehlo.gather"(%2243, %2256) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x16x!pf_babybear_mont>, tensor<8x1xi32>) -> tensor<4x8x!pf_babybear_mont>
    %2258 = stablehlo.subtract %2257, %2250 : tensor<4x8x!pf_babybear_mont>
    %cst_448 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2259 = stablehlo.broadcast_in_dim %cst_448, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8x!pf_babybear_mont>
    %2260 = stablehlo.multiply %2258, %2259 : tensor<4x8x!pf_babybear_mont>
    %2261 = stablehlo.add %2260, %2250 : tensor<4x8x!pf_babybear_mont>
    %2262 = stablehlo.slice %2261 [3:4, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2263 = stablehlo.reshape %2262 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2264 = stablehlo.slice %2261 [0:1, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2265 = stablehlo.reshape %2264 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2266 = stablehlo.slice %2261 [1:2, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2267 = stablehlo.reshape %2266 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2268 = stablehlo.multiply %2265, %2267 : tensor<8x!pf_babybear_mont>
    %2269 = stablehlo.slice %2261 [2:3, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2270 = stablehlo.reshape %2269 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2271 = stablehlo.subtract %2268, %2270 : tensor<8x!pf_babybear_mont>
    %2272 = stablehlo.multiply %2263, %2271 : tensor<8x!pf_babybear_mont>
    %cst_449 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2273 = stablehlo.reduce(%2272 init: %cst_449) applies stablehlo.add across dimensions = [0] : (tensor<8x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_450 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2274 = stablehlo.broadcast_in_dim %cst_450, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8x!pf_babybear_mont>
    %2275 = stablehlo.multiply %2258, %2274 : tensor<4x8x!pf_babybear_mont>
    %2276 = stablehlo.add %2275, %2250 : tensor<4x8x!pf_babybear_mont>
    %2277 = stablehlo.slice %2276 [3:4, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2278 = stablehlo.reshape %2277 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2279 = stablehlo.slice %2276 [0:1, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2280 = stablehlo.reshape %2279 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2281 = stablehlo.slice %2276 [1:2, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2282 = stablehlo.reshape %2281 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2283 = stablehlo.multiply %2280, %2282 : tensor<8x!pf_babybear_mont>
    %2284 = stablehlo.slice %2276 [2:3, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2285 = stablehlo.reshape %2284 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2286 = stablehlo.subtract %2283, %2285 : tensor<8x!pf_babybear_mont>
    %2287 = stablehlo.multiply %2278, %2286 : tensor<8x!pf_babybear_mont>
    %cst_451 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2288 = stablehlo.reduce(%2287 init: %cst_451) applies stablehlo.add across dimensions = [0] : (tensor<8x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_452 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2289 = stablehlo.broadcast_in_dim %cst_452, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8x!pf_babybear_mont>
    %2290 = stablehlo.multiply %2258, %2289 : tensor<4x8x!pf_babybear_mont>
    %2291 = stablehlo.add %2290, %2250 : tensor<4x8x!pf_babybear_mont>
    %2292 = stablehlo.slice %2291 [3:4, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2293 = stablehlo.reshape %2292 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2294 = stablehlo.slice %2291 [0:1, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2295 = stablehlo.reshape %2294 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2296 = stablehlo.slice %2291 [1:2, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2297 = stablehlo.reshape %2296 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2298 = stablehlo.multiply %2295, %2297 : tensor<8x!pf_babybear_mont>
    %2299 = stablehlo.slice %2291 [2:3, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2300 = stablehlo.reshape %2299 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2301 = stablehlo.subtract %2298, %2300 : tensor<8x!pf_babybear_mont>
    %2302 = stablehlo.multiply %2293, %2301 : tensor<8x!pf_babybear_mont>
    %cst_453 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2303 = stablehlo.reduce(%2302 init: %cst_453) applies stablehlo.add across dimensions = [0] : (tensor<8x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_454 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2304 = stablehlo.broadcast_in_dim %cst_454, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8x!pf_babybear_mont>
    %2305 = stablehlo.multiply %2258, %2304 : tensor<4x8x!pf_babybear_mont>
    %2306 = stablehlo.add %2305, %2250 : tensor<4x8x!pf_babybear_mont>
    %2307 = stablehlo.slice %2306 [3:4, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2308 = stablehlo.reshape %2307 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2309 = stablehlo.slice %2306 [0:1, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2310 = stablehlo.reshape %2309 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2311 = stablehlo.slice %2306 [1:2, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2312 = stablehlo.reshape %2311 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2313 = stablehlo.multiply %2310, %2312 : tensor<8x!pf_babybear_mont>
    %2314 = stablehlo.slice %2306 [2:3, 0:8] : (tensor<4x8x!pf_babybear_mont>) -> tensor<1x8x!pf_babybear_mont>
    %2315 = stablehlo.reshape %2314 : (tensor<1x8x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %2316 = stablehlo.subtract %2313, %2315 : tensor<8x!pf_babybear_mont>
    %2317 = stablehlo.multiply %2308, %2316 : tensor<8x!pf_babybear_mont>
    %cst_455 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2318 = stablehlo.reduce(%2317 init: %cst_455) applies stablehlo.add across dimensions = [0] : (tensor<8x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2319 = stablehlo.broadcast_in_dim %2273, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2320 = stablehlo.broadcast_in_dim %2288, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2321 = stablehlo.broadcast_in_dim %2303, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2322 = stablehlo.broadcast_in_dim %2318, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2323 = stablehlo.concatenate %2319, %2320, %2321, %2322, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2324 = stablehlo.reshape %2240 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2325 = stablehlo.concatenate %cst, %2324, %cst_23, %2323, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_456 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2326 = stablehlo.broadcast_in_dim %cst_456, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2327 = stablehlo.slice %2325 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2328 = stablehlo.reshape %2327 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_457 = stablehlo.constant dense<0> : tensor<i32>
    %2329 = stablehlo.broadcast_in_dim %c_457, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2330 = "stablehlo.scatter"(%2326, %2329, %2328) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_458 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2331 = stablehlo.broadcast_in_dim %cst_458, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %2332 = stablehlo.slice %2325 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_459 = stablehlo.constant dense<0> : tensor<i32>
    %2333 = stablehlo.broadcast_in_dim %c_459, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2334 = "stablehlo.scatter"(%2331, %2333, %2332) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_460 = stablehlo.constant dense<8> : tensor<i32>
    %2335 = stablehlo.broadcast_in_dim %c_460, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_461 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2336 = "stablehlo.scatter"(%2334, %2335, %cst_461) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_462 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2337 = stablehlo.broadcast_in_dim %cst_462, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2338 = stablehlo.concatenate %2337, %2336, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2339 = stablehlo.add %2330, %2338 : tensor<16x!pf_babybear_mont>
    %2340 = call @permute(%2339) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2341 = stablehlo.slice %2340 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2342 = stablehlo.reshape %2341 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2343 = stablehlo.broadcast_in_dim %2342, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x8x!pf_babybear_mont>
    %2344 = stablehlo.multiply %2258, %2343 : tensor<4x8x!pf_babybear_mont>
    %2345 = stablehlo.add %2344, %2250 : tensor<4x8x!pf_babybear_mont>
    %2346 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c_463 = stablehlo.constant dense<2> : tensor<i32>
    %2347 = stablehlo.broadcast_in_dim %c_463, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2348 = stablehlo.multiply %2347, %2346 : tensor<4xi32>
    %c_464 = stablehlo.constant dense<0> : tensor<i32>
    %2349 = stablehlo.broadcast_in_dim %c_464, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2350 = stablehlo.add %2349, %2348 : tensor<4xi32>
    %2351 = stablehlo.broadcast_in_dim %2350, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %2352 = "stablehlo.gather"(%2345, %2351) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x8x!pf_babybear_mont>, tensor<4x1xi32>) -> tensor<4x4x!pf_babybear_mont>
    %2353 = stablehlo.iota dim = 0 : tensor<4xi32>
    %c_465 = stablehlo.constant dense<2> : tensor<i32>
    %2354 = stablehlo.broadcast_in_dim %c_465, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2355 = stablehlo.multiply %2354, %2353 : tensor<4xi32>
    %c_466 = stablehlo.constant dense<1> : tensor<i32>
    %2356 = stablehlo.broadcast_in_dim %c_466, dims = [] : (tensor<i32>) -> tensor<4xi32>
    %2357 = stablehlo.add %2356, %2355 : tensor<4xi32>
    %2358 = stablehlo.broadcast_in_dim %2357, dims = [0] : (tensor<4xi32>) -> tensor<4x1xi32>
    %2359 = "stablehlo.gather"(%2345, %2358) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x8x!pf_babybear_mont>, tensor<4x1xi32>) -> tensor<4x4x!pf_babybear_mont>
    %2360 = stablehlo.subtract %2359, %2352 : tensor<4x4x!pf_babybear_mont>
    %cst_467 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2361 = stablehlo.broadcast_in_dim %cst_467, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %2362 = stablehlo.multiply %2360, %2361 : tensor<4x4x!pf_babybear_mont>
    %2363 = stablehlo.add %2362, %2352 : tensor<4x4x!pf_babybear_mont>
    %2364 = stablehlo.slice %2363 [3:4, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2365 = stablehlo.reshape %2364 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2366 = stablehlo.slice %2363 [0:1, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2367 = stablehlo.reshape %2366 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2368 = stablehlo.slice %2363 [1:2, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2369 = stablehlo.reshape %2368 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2370 = stablehlo.multiply %2367, %2369 : tensor<4x!pf_babybear_mont>
    %2371 = stablehlo.slice %2363 [2:3, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2372 = stablehlo.reshape %2371 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2373 = stablehlo.subtract %2370, %2372 : tensor<4x!pf_babybear_mont>
    %2374 = stablehlo.multiply %2365, %2373 : tensor<4x!pf_babybear_mont>
    %cst_468 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2375 = stablehlo.reduce(%2374 init: %cst_468) applies stablehlo.add across dimensions = [0] : (tensor<4x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_469 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2376 = stablehlo.broadcast_in_dim %cst_469, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %2377 = stablehlo.multiply %2360, %2376 : tensor<4x4x!pf_babybear_mont>
    %2378 = stablehlo.add %2377, %2352 : tensor<4x4x!pf_babybear_mont>
    %2379 = stablehlo.slice %2378 [3:4, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2380 = stablehlo.reshape %2379 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2381 = stablehlo.slice %2378 [0:1, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2382 = stablehlo.reshape %2381 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2383 = stablehlo.slice %2378 [1:2, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2384 = stablehlo.reshape %2383 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2385 = stablehlo.multiply %2382, %2384 : tensor<4x!pf_babybear_mont>
    %2386 = stablehlo.slice %2378 [2:3, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2387 = stablehlo.reshape %2386 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2388 = stablehlo.subtract %2385, %2387 : tensor<4x!pf_babybear_mont>
    %2389 = stablehlo.multiply %2380, %2388 : tensor<4x!pf_babybear_mont>
    %cst_470 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2390 = stablehlo.reduce(%2389 init: %cst_470) applies stablehlo.add across dimensions = [0] : (tensor<4x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_471 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2391 = stablehlo.broadcast_in_dim %cst_471, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %2392 = stablehlo.multiply %2360, %2391 : tensor<4x4x!pf_babybear_mont>
    %2393 = stablehlo.add %2392, %2352 : tensor<4x4x!pf_babybear_mont>
    %2394 = stablehlo.slice %2393 [3:4, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2395 = stablehlo.reshape %2394 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2396 = stablehlo.slice %2393 [0:1, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2397 = stablehlo.reshape %2396 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2398 = stablehlo.slice %2393 [1:2, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2399 = stablehlo.reshape %2398 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2400 = stablehlo.multiply %2397, %2399 : tensor<4x!pf_babybear_mont>
    %2401 = stablehlo.slice %2393 [2:3, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2402 = stablehlo.reshape %2401 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2403 = stablehlo.subtract %2400, %2402 : tensor<4x!pf_babybear_mont>
    %2404 = stablehlo.multiply %2395, %2403 : tensor<4x!pf_babybear_mont>
    %cst_472 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2405 = stablehlo.reduce(%2404 init: %cst_472) applies stablehlo.add across dimensions = [0] : (tensor<4x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_473 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2406 = stablehlo.broadcast_in_dim %cst_473, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %2407 = stablehlo.multiply %2360, %2406 : tensor<4x4x!pf_babybear_mont>
    %2408 = stablehlo.add %2407, %2352 : tensor<4x4x!pf_babybear_mont>
    %2409 = stablehlo.slice %2408 [3:4, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2410 = stablehlo.reshape %2409 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2411 = stablehlo.slice %2408 [0:1, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2412 = stablehlo.reshape %2411 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2413 = stablehlo.slice %2408 [1:2, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2414 = stablehlo.reshape %2413 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2415 = stablehlo.multiply %2412, %2414 : tensor<4x!pf_babybear_mont>
    %2416 = stablehlo.slice %2408 [2:3, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2417 = stablehlo.reshape %2416 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2418 = stablehlo.subtract %2415, %2417 : tensor<4x!pf_babybear_mont>
    %2419 = stablehlo.multiply %2410, %2418 : tensor<4x!pf_babybear_mont>
    %cst_474 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2420 = stablehlo.reduce(%2419 init: %cst_474) applies stablehlo.add across dimensions = [0] : (tensor<4x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2421 = stablehlo.broadcast_in_dim %2375, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2422 = stablehlo.broadcast_in_dim %2390, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2423 = stablehlo.broadcast_in_dim %2405, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2424 = stablehlo.broadcast_in_dim %2420, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2425 = stablehlo.concatenate %2421, %2422, %2423, %2424, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2426 = stablehlo.reshape %2342 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2427 = stablehlo.concatenate %cst, %2426, %cst_24, %2425, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_475 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2428 = stablehlo.broadcast_in_dim %cst_475, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2429 = stablehlo.slice %2427 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2430 = stablehlo.reshape %2429 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_476 = stablehlo.constant dense<0> : tensor<i32>
    %2431 = stablehlo.broadcast_in_dim %c_476, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2432 = "stablehlo.scatter"(%2428, %2431, %2430) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_477 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2433 = stablehlo.broadcast_in_dim %cst_477, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %2434 = stablehlo.slice %2427 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_478 = stablehlo.constant dense<0> : tensor<i32>
    %2435 = stablehlo.broadcast_in_dim %c_478, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2436 = "stablehlo.scatter"(%2433, %2435, %2434) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_479 = stablehlo.constant dense<8> : tensor<i32>
    %2437 = stablehlo.broadcast_in_dim %c_479, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_480 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2438 = "stablehlo.scatter"(%2436, %2437, %cst_480) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_481 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2439 = stablehlo.broadcast_in_dim %cst_481, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2440 = stablehlo.concatenate %2439, %2438, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2441 = stablehlo.add %2432, %2440 : tensor<16x!pf_babybear_mont>
    %2442 = call @permute(%2441) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2443 = stablehlo.slice %2442 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2444 = stablehlo.reshape %2443 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2445 = stablehlo.broadcast_in_dim %2444, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %2446 = stablehlo.multiply %2360, %2445 : tensor<4x4x!pf_babybear_mont>
    %2447 = stablehlo.add %2446, %2352 : tensor<4x4x!pf_babybear_mont>
    %2448 = stablehlo.iota dim = 0 : tensor<2xi32>
    %c_482 = stablehlo.constant dense<2> : tensor<i32>
    %2449 = stablehlo.broadcast_in_dim %c_482, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2450 = stablehlo.multiply %2449, %2448 : tensor<2xi32>
    %c_483 = stablehlo.constant dense<0> : tensor<i32>
    %2451 = stablehlo.broadcast_in_dim %c_483, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2452 = stablehlo.add %2451, %2450 : tensor<2xi32>
    %2453 = stablehlo.broadcast_in_dim %2452, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %2454 = "stablehlo.gather"(%2447, %2453) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x4x!pf_babybear_mont>, tensor<2x1xi32>) -> tensor<4x2x!pf_babybear_mont>
    %2455 = stablehlo.iota dim = 0 : tensor<2xi32>
    %c_484 = stablehlo.constant dense<2> : tensor<i32>
    %2456 = stablehlo.broadcast_in_dim %c_484, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2457 = stablehlo.multiply %2456, %2455 : tensor<2xi32>
    %c_485 = stablehlo.constant dense<1> : tensor<i32>
    %2458 = stablehlo.broadcast_in_dim %c_485, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %2459 = stablehlo.add %2458, %2457 : tensor<2xi32>
    %2460 = stablehlo.broadcast_in_dim %2459, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    %2461 = "stablehlo.gather"(%2447, %2460) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x4x!pf_babybear_mont>, tensor<2x1xi32>) -> tensor<4x2x!pf_babybear_mont>
    %2462 = stablehlo.subtract %2461, %2454 : tensor<4x2x!pf_babybear_mont>
    %cst_486 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2463 = stablehlo.broadcast_in_dim %cst_486, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2x!pf_babybear_mont>
    %2464 = stablehlo.multiply %2462, %2463 : tensor<4x2x!pf_babybear_mont>
    %2465 = stablehlo.add %2464, %2454 : tensor<4x2x!pf_babybear_mont>
    %2466 = stablehlo.slice %2465 [3:4, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2467 = stablehlo.reshape %2466 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2468 = stablehlo.slice %2465 [0:1, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2469 = stablehlo.reshape %2468 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2470 = stablehlo.slice %2465 [1:2, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2471 = stablehlo.reshape %2470 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2472 = stablehlo.multiply %2469, %2471 : tensor<2x!pf_babybear_mont>
    %2473 = stablehlo.slice %2465 [2:3, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2474 = stablehlo.reshape %2473 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2475 = stablehlo.subtract %2472, %2474 : tensor<2x!pf_babybear_mont>
    %2476 = stablehlo.multiply %2467, %2475 : tensor<2x!pf_babybear_mont>
    %cst_487 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2477 = stablehlo.reduce(%2476 init: %cst_487) applies stablehlo.add across dimensions = [0] : (tensor<2x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_488 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2478 = stablehlo.broadcast_in_dim %cst_488, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2x!pf_babybear_mont>
    %2479 = stablehlo.multiply %2462, %2478 : tensor<4x2x!pf_babybear_mont>
    %2480 = stablehlo.add %2479, %2454 : tensor<4x2x!pf_babybear_mont>
    %2481 = stablehlo.slice %2480 [3:4, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2482 = stablehlo.reshape %2481 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2483 = stablehlo.slice %2480 [0:1, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2484 = stablehlo.reshape %2483 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2485 = stablehlo.slice %2480 [1:2, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2486 = stablehlo.reshape %2485 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2487 = stablehlo.multiply %2484, %2486 : tensor<2x!pf_babybear_mont>
    %2488 = stablehlo.slice %2480 [2:3, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2489 = stablehlo.reshape %2488 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2490 = stablehlo.subtract %2487, %2489 : tensor<2x!pf_babybear_mont>
    %2491 = stablehlo.multiply %2482, %2490 : tensor<2x!pf_babybear_mont>
    %cst_489 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2492 = stablehlo.reduce(%2491 init: %cst_489) applies stablehlo.add across dimensions = [0] : (tensor<2x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_490 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2493 = stablehlo.broadcast_in_dim %cst_490, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2x!pf_babybear_mont>
    %2494 = stablehlo.multiply %2462, %2493 : tensor<4x2x!pf_babybear_mont>
    %2495 = stablehlo.add %2494, %2454 : tensor<4x2x!pf_babybear_mont>
    %2496 = stablehlo.slice %2495 [3:4, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2497 = stablehlo.reshape %2496 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2498 = stablehlo.slice %2495 [0:1, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2499 = stablehlo.reshape %2498 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2500 = stablehlo.slice %2495 [1:2, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2501 = stablehlo.reshape %2500 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2502 = stablehlo.multiply %2499, %2501 : tensor<2x!pf_babybear_mont>
    %2503 = stablehlo.slice %2495 [2:3, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2504 = stablehlo.reshape %2503 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2505 = stablehlo.subtract %2502, %2504 : tensor<2x!pf_babybear_mont>
    %2506 = stablehlo.multiply %2497, %2505 : tensor<2x!pf_babybear_mont>
    %cst_491 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2507 = stablehlo.reduce(%2506 init: %cst_491) applies stablehlo.add across dimensions = [0] : (tensor<2x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_492 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2508 = stablehlo.broadcast_in_dim %cst_492, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2x!pf_babybear_mont>
    %2509 = stablehlo.multiply %2462, %2508 : tensor<4x2x!pf_babybear_mont>
    %2510 = stablehlo.add %2509, %2454 : tensor<4x2x!pf_babybear_mont>
    %2511 = stablehlo.slice %2510 [3:4, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2512 = stablehlo.reshape %2511 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2513 = stablehlo.slice %2510 [0:1, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2514 = stablehlo.reshape %2513 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2515 = stablehlo.slice %2510 [1:2, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2516 = stablehlo.reshape %2515 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2517 = stablehlo.multiply %2514, %2516 : tensor<2x!pf_babybear_mont>
    %2518 = stablehlo.slice %2510 [2:3, 0:2] : (tensor<4x2x!pf_babybear_mont>) -> tensor<1x2x!pf_babybear_mont>
    %2519 = stablehlo.reshape %2518 : (tensor<1x2x!pf_babybear_mont>) -> tensor<2x!pf_babybear_mont>
    %2520 = stablehlo.subtract %2517, %2519 : tensor<2x!pf_babybear_mont>
    %2521 = stablehlo.multiply %2512, %2520 : tensor<2x!pf_babybear_mont>
    %cst_493 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2522 = stablehlo.reduce(%2521 init: %cst_493) applies stablehlo.add across dimensions = [0] : (tensor<2x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2523 = stablehlo.broadcast_in_dim %2477, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2524 = stablehlo.broadcast_in_dim %2492, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2525 = stablehlo.broadcast_in_dim %2507, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2526 = stablehlo.broadcast_in_dim %2522, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2527 = stablehlo.concatenate %2523, %2524, %2525, %2526, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2528 = stablehlo.reshape %2444 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2529 = stablehlo.concatenate %cst, %2528, %cst_25, %2527, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_494 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2530 = stablehlo.broadcast_in_dim %cst_494, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2531 = stablehlo.slice %2529 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2532 = stablehlo.reshape %2531 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_495 = stablehlo.constant dense<0> : tensor<i32>
    %2533 = stablehlo.broadcast_in_dim %c_495, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2534 = "stablehlo.scatter"(%2530, %2533, %2532) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_496 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2535 = stablehlo.broadcast_in_dim %cst_496, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %2536 = stablehlo.slice %2529 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_497 = stablehlo.constant dense<0> : tensor<i32>
    %2537 = stablehlo.broadcast_in_dim %c_497, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2538 = "stablehlo.scatter"(%2535, %2537, %2536) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_498 = stablehlo.constant dense<8> : tensor<i32>
    %2539 = stablehlo.broadcast_in_dim %c_498, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_499 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2540 = "stablehlo.scatter"(%2538, %2539, %cst_499) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_500 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2541 = stablehlo.broadcast_in_dim %cst_500, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2542 = stablehlo.concatenate %2541, %2540, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2543 = stablehlo.add %2534, %2542 : tensor<16x!pf_babybear_mont>
    %2544 = call @permute(%2543) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2545 = stablehlo.slice %2544 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2546 = stablehlo.reshape %2545 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2547 = stablehlo.broadcast_in_dim %2546, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x2x!pf_babybear_mont>
    %2548 = stablehlo.multiply %2462, %2547 : tensor<4x2x!pf_babybear_mont>
    %2549 = stablehlo.add %2548, %2454 : tensor<4x2x!pf_babybear_mont>
    %2550 = stablehlo.iota dim = 0 : tensor<1xi32>
    %c_501 = stablehlo.constant dense<2> : tensor<i32>
    %2551 = stablehlo.broadcast_in_dim %c_501, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2552 = stablehlo.multiply %2551, %2550 : tensor<1xi32>
    %c_502 = stablehlo.constant dense<0> : tensor<i32>
    %2553 = stablehlo.broadcast_in_dim %c_502, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2554 = stablehlo.add %2553, %2552 : tensor<1xi32>
    %2555 = stablehlo.broadcast_in_dim %2554, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %2556 = "stablehlo.gather"(%2549, %2555) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x2x!pf_babybear_mont>, tensor<1x1xi32>) -> tensor<4x1x!pf_babybear_mont>
    %2557 = stablehlo.iota dim = 0 : tensor<1xi32>
    %c_503 = stablehlo.constant dense<2> : tensor<i32>
    %2558 = stablehlo.broadcast_in_dim %c_503, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2559 = stablehlo.multiply %2558, %2557 : tensor<1xi32>
    %c_504 = stablehlo.constant dense<1> : tensor<i32>
    %2560 = stablehlo.broadcast_in_dim %c_504, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2561 = stablehlo.add %2560, %2559 : tensor<1xi32>
    %2562 = stablehlo.broadcast_in_dim %2561, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %2563 = "stablehlo.gather"(%2549, %2562) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 4, 1>}> : (tensor<4x2x!pf_babybear_mont>, tensor<1x1xi32>) -> tensor<4x1x!pf_babybear_mont>
    %2564 = stablehlo.subtract %2563, %2556 : tensor<4x1x!pf_babybear_mont>
    %cst_505 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2565 = stablehlo.broadcast_in_dim %cst_505, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %2566 = stablehlo.multiply %2564, %2565 : tensor<4x1x!pf_babybear_mont>
    %2567 = stablehlo.add %2566, %2556 : tensor<4x1x!pf_babybear_mont>
    %2568 = stablehlo.slice %2567 [3:4, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2569 = stablehlo.reshape %2568 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2570 = stablehlo.slice %2567 [0:1, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2571 = stablehlo.reshape %2570 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2572 = stablehlo.slice %2567 [1:2, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2573 = stablehlo.reshape %2572 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2574 = stablehlo.multiply %2571, %2573 : tensor<1x!pf_babybear_mont>
    %2575 = stablehlo.slice %2567 [2:3, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2576 = stablehlo.reshape %2575 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2577 = stablehlo.subtract %2574, %2576 : tensor<1x!pf_babybear_mont>
    %2578 = stablehlo.multiply %2569, %2577 : tensor<1x!pf_babybear_mont>
    %cst_506 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2579 = stablehlo.reduce(%2578 init: %cst_506) applies stablehlo.add across dimensions = [0] : (tensor<1x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_507 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2580 = stablehlo.broadcast_in_dim %cst_507, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %2581 = stablehlo.multiply %2564, %2580 : tensor<4x1x!pf_babybear_mont>
    %2582 = stablehlo.add %2581, %2556 : tensor<4x1x!pf_babybear_mont>
    %2583 = stablehlo.slice %2582 [3:4, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2584 = stablehlo.reshape %2583 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2585 = stablehlo.slice %2582 [0:1, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2586 = stablehlo.reshape %2585 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2587 = stablehlo.slice %2582 [1:2, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2588 = stablehlo.reshape %2587 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2589 = stablehlo.multiply %2586, %2588 : tensor<1x!pf_babybear_mont>
    %2590 = stablehlo.slice %2582 [2:3, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2591 = stablehlo.reshape %2590 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2592 = stablehlo.subtract %2589, %2591 : tensor<1x!pf_babybear_mont>
    %2593 = stablehlo.multiply %2584, %2592 : tensor<1x!pf_babybear_mont>
    %cst_508 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2594 = stablehlo.reduce(%2593 init: %cst_508) applies stablehlo.add across dimensions = [0] : (tensor<1x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_509 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2595 = stablehlo.broadcast_in_dim %cst_509, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %2596 = stablehlo.multiply %2564, %2595 : tensor<4x1x!pf_babybear_mont>
    %2597 = stablehlo.add %2596, %2556 : tensor<4x1x!pf_babybear_mont>
    %2598 = stablehlo.slice %2597 [3:4, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2599 = stablehlo.reshape %2598 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2600 = stablehlo.slice %2597 [0:1, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2601 = stablehlo.reshape %2600 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2602 = stablehlo.slice %2597 [1:2, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2603 = stablehlo.reshape %2602 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2604 = stablehlo.multiply %2601, %2603 : tensor<1x!pf_babybear_mont>
    %2605 = stablehlo.slice %2597 [2:3, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2606 = stablehlo.reshape %2605 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2607 = stablehlo.subtract %2604, %2606 : tensor<1x!pf_babybear_mont>
    %2608 = stablehlo.multiply %2599, %2607 : tensor<1x!pf_babybear_mont>
    %cst_510 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2609 = stablehlo.reduce(%2608 init: %cst_510) applies stablehlo.add across dimensions = [0] : (tensor<1x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %cst_511 = stablehlo.constant() <{value = dense<805306362> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2610 = stablehlo.broadcast_in_dim %cst_511, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %2611 = stablehlo.multiply %2564, %2610 : tensor<4x1x!pf_babybear_mont>
    %2612 = stablehlo.add %2611, %2556 : tensor<4x1x!pf_babybear_mont>
    %2613 = stablehlo.slice %2612 [3:4, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2614 = stablehlo.reshape %2613 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2615 = stablehlo.slice %2612 [0:1, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2616 = stablehlo.reshape %2615 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2617 = stablehlo.slice %2612 [1:2, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2618 = stablehlo.reshape %2617 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2619 = stablehlo.multiply %2616, %2618 : tensor<1x!pf_babybear_mont>
    %2620 = stablehlo.slice %2612 [2:3, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<1x1x!pf_babybear_mont>
    %2621 = stablehlo.reshape %2620 : (tensor<1x1x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2622 = stablehlo.subtract %2619, %2621 : tensor<1x!pf_babybear_mont>
    %2623 = stablehlo.multiply %2614, %2622 : tensor<1x!pf_babybear_mont>
    %cst_512 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2624 = stablehlo.reduce(%2623 init: %cst_512) applies stablehlo.add across dimensions = [0] : (tensor<1x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2625 = stablehlo.broadcast_in_dim %2579, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2626 = stablehlo.broadcast_in_dim %2594, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2627 = stablehlo.broadcast_in_dim %2609, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2628 = stablehlo.broadcast_in_dim %2624, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2629 = stablehlo.concatenate %2625, %2626, %2627, %2628, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %2630 = stablehlo.reshape %2546 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2631 = stablehlo.concatenate %cst, %2630, %cst_26, %2629, dim = 0 : (tensor<2x!pf_babybear_mont>, tensor<1x!pf_babybear_mont>, tensor<2x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>) -> tensor<9x!pf_babybear_mont>
    %cst_513 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2632 = stablehlo.broadcast_in_dim %cst_513, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2633 = stablehlo.slice %2631 [0:1] : (tensor<9x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2634 = stablehlo.reshape %2633 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %c_514 = stablehlo.constant dense<0> : tensor<i32>
    %2635 = stablehlo.broadcast_in_dim %c_514, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2636 = "stablehlo.scatter"(%2632, %2635, %2634) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<16x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_515 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2637 = stablehlo.broadcast_in_dim %cst_515, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %2638 = stablehlo.slice %2631 [1:9] : (tensor<9x!pf_babybear_mont>) -> tensor<8x!pf_babybear_mont>
    %c_516 = stablehlo.constant dense<0> : tensor<i32>
    %2639 = stablehlo.broadcast_in_dim %c_516, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2640 = "stablehlo.scatter"(%2637, %2639, %2638) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<8x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %c_517 = stablehlo.constant dense<8> : tensor<i32>
    %2641 = stablehlo.broadcast_in_dim %c_517, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %cst_518 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2642 = "stablehlo.scatter"(%2640, %2641, %cst_518) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont>, %arg3: tensor<!pf_babybear_mont>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont>
    }) : (tensor<15x!pf_babybear_mont>, tensor<1xi32>, tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_519 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %2643 = stablehlo.broadcast_in_dim %cst_519, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2644 = stablehlo.concatenate %2643, %2642, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2645 = stablehlo.add %2636, %2644 : tensor<16x!pf_babybear_mont>
    %2646 = call @permute(%2645) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2647 = stablehlo.slice %2646 [1:2] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %2648 = stablehlo.reshape %2647 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2649 = stablehlo.broadcast_in_dim %2648, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %2650 = stablehlo.multiply %2564, %2649 : tensor<4x1x!pf_babybear_mont>
    %2651 = stablehlo.add %2650, %2556 : tensor<4x1x!pf_babybear_mont>
    %2652 = stablehlo.broadcast_in_dim %79, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2653 = stablehlo.broadcast_in_dim %181, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2654 = stablehlo.broadcast_in_dim %283, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2655 = stablehlo.broadcast_in_dim %385, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2656 = stablehlo.broadcast_in_dim %487, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2657 = stablehlo.broadcast_in_dim %589, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2658 = stablehlo.broadcast_in_dim %691, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2659 = stablehlo.broadcast_in_dim %793, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2660 = stablehlo.broadcast_in_dim %895, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2661 = stablehlo.broadcast_in_dim %997, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2662 = stablehlo.broadcast_in_dim %1099, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2663 = stablehlo.broadcast_in_dim %1201, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2664 = stablehlo.broadcast_in_dim %1303, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2665 = stablehlo.broadcast_in_dim %1405, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2666 = stablehlo.broadcast_in_dim %1507, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2667 = stablehlo.broadcast_in_dim %1609, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2668 = stablehlo.broadcast_in_dim %1711, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2669 = stablehlo.broadcast_in_dim %1813, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2670 = stablehlo.broadcast_in_dim %1915, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2671 = stablehlo.broadcast_in_dim %2017, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2672 = stablehlo.broadcast_in_dim %2119, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2673 = stablehlo.broadcast_in_dim %2221, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2674 = stablehlo.broadcast_in_dim %2323, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2675 = stablehlo.broadcast_in_dim %2425, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2676 = stablehlo.broadcast_in_dim %2527, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2677 = stablehlo.broadcast_in_dim %2629, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %2678 = stablehlo.concatenate %2652, %2653, %2654, %2655, %2656, %2657, %2658, %2659, %2660, %2661, %2662, %2663, %2664, %2665, %2666, %2667, dim = 0 : (tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>) -> tensor<16x4x!pf_babybear_mont>
    %2679 = stablehlo.concatenate %2668, %2669, %2670, %2671, %2672, %2673, %2674, %2675, %2676, %2677, dim = 0 : (tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>, tensor<1x4x!pf_babybear_mont>) -> tensor<10x4x!pf_babybear_mont>
    %2680 = stablehlo.concatenate %2678, %2679, dim = 0 : (tensor<16x4x!pf_babybear_mont>, tensor<10x4x!pf_babybear_mont>) -> tensor<26x4x!pf_babybear_mont>
    %2681 = stablehlo.slice %2651 [0:4, 0:1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %2682 = stablehlo.reshape %2681 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    return %2680, %2682 : tensor<26x4x!pf_babybear_mont>, tensor<4x!pf_babybear_mont>
  }
  func.func private @permute(%arg0: tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont> {
    %cst = stablehlo.constant() <{value = dense<"0xEBFFFF27E3FFFF67FCFFFF1FF4FFFF5FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FEFFFFF07E7FFFF47FCFFFF1FFCFFFF1FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FFCFFFF1FF4FFFF5FEBFFFF27E3FFFF67FEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFCFFFF1FFCFFFF1FEFFFFF07E7FFFF47FEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FEBFFFF27E3FFFF67FCFFFF1FF4FFFF5FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FEFFFFF07E7FFFF47FCFFFF1FFCFFFF1FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFCFFFF1FF4FFFF5FEBFFFF27E3FFFF67FEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFCFFFF1FFCFFFF1FEFFFFF07E7FFFF47FEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FEBFFFF27E3FFFF67FCFFFF1FF4FFFF5FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FEFFFFF07E7FFFF47FCFFFF1FFCFFFF1FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFCFFFF1FF4FFFF5FEBFFFF27E3FFFF67FEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFCFFFF1FFCFFFF1FEFFFFF07E7FFFF47FEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FEBFFFF27E3FFFF67FCFFFF1FF4FFFF5FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FEFFFFF07E7FFFF47FCFFFF1FFCFFFF1FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFEFFFF0FFAFFFF2FF6FFFF4FF2FFFF6FFCFFFF1FF4FFFF5FEBFFFF27E3FFFF67FEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFEFFFF0FFEFFFF0FF8FFFF3FF4FFFF5FFCFFFF1FFCFFFF1FEFFFFF07E7FFFF47"> : tensor<16x16xi32>}> : () -> tensor<16x16x!pf_babybear_mont>
    %cst_0 = stablehlo.constant() <{value = dense<[[1582131512, 1899519471, 1641921850, 462688640, 1293997949, 1380417575, 1932416963, 283521298, 1016708647, 35751290, 1270782647, 851730739, 795004022, 929571430, 523703523, 1593957757], [895976710, 1742343460, 917700746, 1516725708, 1170237629, 785693164, 613651155, 352999196, 678775274, 1005433272, 1704854670, 1174551920, 508930349, 530338447, 1327158816, 1417652352], [1153538870, 583201050, 397833841, 1440603828, 454600685, 174490638, 171758601, 1998476616, 1403697810, 1807736944, 450348306, 1458895865, 787037868, 1063762964, 1987002214, 481645916], [1231767638, 1323639433, 238360103, 2012412459, 1024945356, 1108359895, 1284135849, 606928406, 1021455954, 719347978, 659671051, 769588663, 805534062, 592213995, 1752728055, 663410947]]> : tensor<4x16xi32>}> : () -> tensor<4x16x!pf_babybear_mont>
    %cst_1 = stablehlo.constant() <{value = dense<[250494022, 528496384, 1472966118, 977089650, 1885890237, 1094557811, 147492661, 664163003, 398852570, 336233633, 1628648315, 888594966, 586791090]> : tensor<13xi32>}> : () -> tensor<13x!pf_babybear_mont>
    %cst_2 = stablehlo.constant() <{value = dense<[165090876, 1710389270, 1584347757, 1694045778, 842730700, 1672819211, 1452398194, 1206256413, 294912425, 1063530332, 254581617, 133701014, 1340916454, 1276372118, 333256835, 1889983964]> : tensor<16xi32>}> : () -> tensor<16x!pf_babybear_mont>
    %cst_3 = stablehlo.constant() <{value = dense<[[999830298, 304461056, 552699684, 450698925, 667466464, 1736509752, 1327760865, 1153241151, 816675655, 1076172858, 1914832527, 1668723429, 1365579850, 975704528, 1031625628, 1393317533], [1554700828, 1023828605, 1610378860, 347744760, 1909572073, 739227895, 428565985, 633143046, 121797685, 94048546, 1369350241, 1250010422, 114268841, 515033604, 49052844, 1962329907], [1380892638, 1860017417, 64711457, 9758460, 1681838395, 710850601, 1020228997, 1414164790, 1531515535, 36158805, 713604525, 89935127, 1870801994, 395985906, 1122769045, 1760811055], [819787042, 134654834, 1755145179, 18433016, 1701878989, 1782339297, 1483861396, 962480061, 1857590724, 222440409, 63223417, 515206622, 1348364213, 973414686, 1591066884, 705852913]]> : tensor<4x16xi32>}> : () -> tensor<4x16x!pf_babybear_mont>
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %2 = stablehlo.multiply %cst, %1 : tensor<16x16x!pf_babybear_mont>
    %cst_4 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %3 = stablehlo.reduce(%2 init: %cst_4) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %4 = stablehlo.slice %cst_0 [0:1, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %5 = stablehlo.reshape %4 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %6 = stablehlo.add %3, %5 : tensor<16x!pf_babybear_mont>
    %7 = stablehlo.multiply %6, %6 : tensor<16x!pf_babybear_mont>
    %8 = stablehlo.multiply %6, %7 : tensor<16x!pf_babybear_mont>
    %9 = stablehlo.multiply %7, %7 : tensor<16x!pf_babybear_mont>
    %10 = stablehlo.multiply %8, %9 : tensor<16x!pf_babybear_mont>
    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %13 = stablehlo.multiply %cst, %12 : tensor<16x16x!pf_babybear_mont>
    %cst_5 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %14 = stablehlo.reduce(%13 init: %cst_5) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %15 = stablehlo.slice %cst_0 [1:2, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %16 = stablehlo.reshape %15 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %17 = stablehlo.add %14, %16 : tensor<16x!pf_babybear_mont>
    %18 = stablehlo.multiply %17, %17 : tensor<16x!pf_babybear_mont>
    %19 = stablehlo.multiply %17, %18 : tensor<16x!pf_babybear_mont>
    %20 = stablehlo.multiply %18, %18 : tensor<16x!pf_babybear_mont>
    %21 = stablehlo.multiply %19, %20 : tensor<16x!pf_babybear_mont>
    %22 = stablehlo.broadcast_in_dim %21, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %24 = stablehlo.multiply %cst, %23 : tensor<16x16x!pf_babybear_mont>
    %cst_6 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %25 = stablehlo.reduce(%24 init: %cst_6) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %26 = stablehlo.slice %cst_0 [2:3, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %27 = stablehlo.reshape %26 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %28 = stablehlo.add %25, %27 : tensor<16x!pf_babybear_mont>
    %29 = stablehlo.multiply %28, %28 : tensor<16x!pf_babybear_mont>
    %30 = stablehlo.multiply %28, %29 : tensor<16x!pf_babybear_mont>
    %31 = stablehlo.multiply %29, %29 : tensor<16x!pf_babybear_mont>
    %32 = stablehlo.multiply %30, %31 : tensor<16x!pf_babybear_mont>
    %33 = stablehlo.broadcast_in_dim %32, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %35 = stablehlo.multiply %cst, %34 : tensor<16x16x!pf_babybear_mont>
    %cst_7 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %36 = stablehlo.reduce(%35 init: %cst_7) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %37 = stablehlo.slice %cst_0 [3:4, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %38 = stablehlo.reshape %37 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %39 = stablehlo.add %36, %38 : tensor<16x!pf_babybear_mont>
    %40 = stablehlo.multiply %39, %39 : tensor<16x!pf_babybear_mont>
    %41 = stablehlo.multiply %39, %40 : tensor<16x!pf_babybear_mont>
    %42 = stablehlo.multiply %40, %40 : tensor<16x!pf_babybear_mont>
    %43 = stablehlo.multiply %41, %42 : tensor<16x!pf_babybear_mont>
    %44 = stablehlo.broadcast_in_dim %43, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %46 = stablehlo.multiply %cst, %45 : tensor<16x16x!pf_babybear_mont>
    %cst_8 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %47 = stablehlo.reduce(%46 init: %cst_8) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %48 = stablehlo.slice %cst_1 [0:1] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %49 = stablehlo.reshape %48 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %50 = stablehlo.slice %47 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %51 = stablehlo.reshape %50 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %52 = stablehlo.add %51, %49 : tensor<!pf_babybear_mont>
    %53 = stablehlo.multiply %52, %52 : tensor<!pf_babybear_mont>
    %54 = stablehlo.multiply %52, %53 : tensor<!pf_babybear_mont>
    %55 = stablehlo.multiply %53, %53 : tensor<!pf_babybear_mont>
    %56 = stablehlo.multiply %54, %55 : tensor<!pf_babybear_mont>
    %57 = stablehlo.reshape %56 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %58 = stablehlo.slice %47 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %59 = stablehlo.concatenate %57, %58, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_9 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %60 = stablehlo.reduce(%59 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %61 = stablehlo.multiply %59, %cst_2 : tensor<16x!pf_babybear_mont>
    %62 = stablehlo.broadcast_in_dim %60, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %63 = stablehlo.add %61, %62 : tensor<16x!pf_babybear_mont>
    %64 = stablehlo.slice %cst_1 [1:2] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %65 = stablehlo.reshape %64 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %66 = stablehlo.slice %63 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %67 = stablehlo.reshape %66 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %68 = stablehlo.add %67, %65 : tensor<!pf_babybear_mont>
    %69 = stablehlo.multiply %68, %68 : tensor<!pf_babybear_mont>
    %70 = stablehlo.multiply %68, %69 : tensor<!pf_babybear_mont>
    %71 = stablehlo.multiply %69, %69 : tensor<!pf_babybear_mont>
    %72 = stablehlo.multiply %70, %71 : tensor<!pf_babybear_mont>
    %73 = stablehlo.reshape %72 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %74 = stablehlo.slice %63 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %75 = stablehlo.concatenate %73, %74, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_10 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %76 = stablehlo.reduce(%75 init: %cst_10) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %77 = stablehlo.multiply %75, %cst_2 : tensor<16x!pf_babybear_mont>
    %78 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %79 = stablehlo.add %77, %78 : tensor<16x!pf_babybear_mont>
    %80 = stablehlo.slice %cst_1 [2:3] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %81 = stablehlo.reshape %80 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %82 = stablehlo.slice %79 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %83 = stablehlo.reshape %82 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %84 = stablehlo.add %83, %81 : tensor<!pf_babybear_mont>
    %85 = stablehlo.multiply %84, %84 : tensor<!pf_babybear_mont>
    %86 = stablehlo.multiply %84, %85 : tensor<!pf_babybear_mont>
    %87 = stablehlo.multiply %85, %85 : tensor<!pf_babybear_mont>
    %88 = stablehlo.multiply %86, %87 : tensor<!pf_babybear_mont>
    %89 = stablehlo.reshape %88 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %90 = stablehlo.slice %79 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %91 = stablehlo.concatenate %89, %90, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_11 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %92 = stablehlo.reduce(%91 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %93 = stablehlo.multiply %91, %cst_2 : tensor<16x!pf_babybear_mont>
    %94 = stablehlo.broadcast_in_dim %92, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %95 = stablehlo.add %93, %94 : tensor<16x!pf_babybear_mont>
    %96 = stablehlo.slice %cst_1 [3:4] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %97 = stablehlo.reshape %96 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %98 = stablehlo.slice %95 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %99 = stablehlo.reshape %98 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %100 = stablehlo.add %99, %97 : tensor<!pf_babybear_mont>
    %101 = stablehlo.multiply %100, %100 : tensor<!pf_babybear_mont>
    %102 = stablehlo.multiply %100, %101 : tensor<!pf_babybear_mont>
    %103 = stablehlo.multiply %101, %101 : tensor<!pf_babybear_mont>
    %104 = stablehlo.multiply %102, %103 : tensor<!pf_babybear_mont>
    %105 = stablehlo.reshape %104 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %106 = stablehlo.slice %95 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %107 = stablehlo.concatenate %105, %106, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_12 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %108 = stablehlo.reduce(%107 init: %cst_12) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %109 = stablehlo.multiply %107, %cst_2 : tensor<16x!pf_babybear_mont>
    %110 = stablehlo.broadcast_in_dim %108, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %111 = stablehlo.add %109, %110 : tensor<16x!pf_babybear_mont>
    %112 = stablehlo.slice %cst_1 [4:5] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %113 = stablehlo.reshape %112 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %114 = stablehlo.slice %111 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %115 = stablehlo.reshape %114 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %116 = stablehlo.add %115, %113 : tensor<!pf_babybear_mont>
    %117 = stablehlo.multiply %116, %116 : tensor<!pf_babybear_mont>
    %118 = stablehlo.multiply %116, %117 : tensor<!pf_babybear_mont>
    %119 = stablehlo.multiply %117, %117 : tensor<!pf_babybear_mont>
    %120 = stablehlo.multiply %118, %119 : tensor<!pf_babybear_mont>
    %121 = stablehlo.reshape %120 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %122 = stablehlo.slice %111 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %123 = stablehlo.concatenate %121, %122, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_13 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %124 = stablehlo.reduce(%123 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %125 = stablehlo.multiply %123, %cst_2 : tensor<16x!pf_babybear_mont>
    %126 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %127 = stablehlo.add %125, %126 : tensor<16x!pf_babybear_mont>
    %128 = stablehlo.slice %cst_1 [5:6] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %129 = stablehlo.reshape %128 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %130 = stablehlo.slice %127 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %131 = stablehlo.reshape %130 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %132 = stablehlo.add %131, %129 : tensor<!pf_babybear_mont>
    %133 = stablehlo.multiply %132, %132 : tensor<!pf_babybear_mont>
    %134 = stablehlo.multiply %132, %133 : tensor<!pf_babybear_mont>
    %135 = stablehlo.multiply %133, %133 : tensor<!pf_babybear_mont>
    %136 = stablehlo.multiply %134, %135 : tensor<!pf_babybear_mont>
    %137 = stablehlo.reshape %136 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %138 = stablehlo.slice %127 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %139 = stablehlo.concatenate %137, %138, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_14 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %140 = stablehlo.reduce(%139 init: %cst_14) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %141 = stablehlo.multiply %139, %cst_2 : tensor<16x!pf_babybear_mont>
    %142 = stablehlo.broadcast_in_dim %140, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %143 = stablehlo.add %141, %142 : tensor<16x!pf_babybear_mont>
    %144 = stablehlo.slice %cst_1 [6:7] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %145 = stablehlo.reshape %144 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %146 = stablehlo.slice %143 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %147 = stablehlo.reshape %146 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %148 = stablehlo.add %147, %145 : tensor<!pf_babybear_mont>
    %149 = stablehlo.multiply %148, %148 : tensor<!pf_babybear_mont>
    %150 = stablehlo.multiply %148, %149 : tensor<!pf_babybear_mont>
    %151 = stablehlo.multiply %149, %149 : tensor<!pf_babybear_mont>
    %152 = stablehlo.multiply %150, %151 : tensor<!pf_babybear_mont>
    %153 = stablehlo.reshape %152 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %154 = stablehlo.slice %143 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %155 = stablehlo.concatenate %153, %154, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_15 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %156 = stablehlo.reduce(%155 init: %cst_15) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %157 = stablehlo.multiply %155, %cst_2 : tensor<16x!pf_babybear_mont>
    %158 = stablehlo.broadcast_in_dim %156, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %159 = stablehlo.add %157, %158 : tensor<16x!pf_babybear_mont>
    %160 = stablehlo.slice %cst_1 [7:8] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %161 = stablehlo.reshape %160 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %162 = stablehlo.slice %159 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %163 = stablehlo.reshape %162 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %164 = stablehlo.add %163, %161 : tensor<!pf_babybear_mont>
    %165 = stablehlo.multiply %164, %164 : tensor<!pf_babybear_mont>
    %166 = stablehlo.multiply %164, %165 : tensor<!pf_babybear_mont>
    %167 = stablehlo.multiply %165, %165 : tensor<!pf_babybear_mont>
    %168 = stablehlo.multiply %166, %167 : tensor<!pf_babybear_mont>
    %169 = stablehlo.reshape %168 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %170 = stablehlo.slice %159 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %171 = stablehlo.concatenate %169, %170, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_16 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %172 = stablehlo.reduce(%171 init: %cst_16) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %173 = stablehlo.multiply %171, %cst_2 : tensor<16x!pf_babybear_mont>
    %174 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %175 = stablehlo.add %173, %174 : tensor<16x!pf_babybear_mont>
    %176 = stablehlo.slice %cst_1 [8:9] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %177 = stablehlo.reshape %176 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %178 = stablehlo.slice %175 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %179 = stablehlo.reshape %178 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %180 = stablehlo.add %179, %177 : tensor<!pf_babybear_mont>
    %181 = stablehlo.multiply %180, %180 : tensor<!pf_babybear_mont>
    %182 = stablehlo.multiply %180, %181 : tensor<!pf_babybear_mont>
    %183 = stablehlo.multiply %181, %181 : tensor<!pf_babybear_mont>
    %184 = stablehlo.multiply %182, %183 : tensor<!pf_babybear_mont>
    %185 = stablehlo.reshape %184 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %186 = stablehlo.slice %175 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %187 = stablehlo.concatenate %185, %186, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_17 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %188 = stablehlo.reduce(%187 init: %cst_17) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %189 = stablehlo.multiply %187, %cst_2 : tensor<16x!pf_babybear_mont>
    %190 = stablehlo.broadcast_in_dim %188, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %191 = stablehlo.add %189, %190 : tensor<16x!pf_babybear_mont>
    %192 = stablehlo.slice %cst_1 [9:10] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %193 = stablehlo.reshape %192 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %194 = stablehlo.slice %191 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %195 = stablehlo.reshape %194 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %196 = stablehlo.add %195, %193 : tensor<!pf_babybear_mont>
    %197 = stablehlo.multiply %196, %196 : tensor<!pf_babybear_mont>
    %198 = stablehlo.multiply %196, %197 : tensor<!pf_babybear_mont>
    %199 = stablehlo.multiply %197, %197 : tensor<!pf_babybear_mont>
    %200 = stablehlo.multiply %198, %199 : tensor<!pf_babybear_mont>
    %201 = stablehlo.reshape %200 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %202 = stablehlo.slice %191 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %203 = stablehlo.concatenate %201, %202, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_18 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %204 = stablehlo.reduce(%203 init: %cst_18) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %205 = stablehlo.multiply %203, %cst_2 : tensor<16x!pf_babybear_mont>
    %206 = stablehlo.broadcast_in_dim %204, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %207 = stablehlo.add %205, %206 : tensor<16x!pf_babybear_mont>
    %208 = stablehlo.slice %cst_1 [10:11] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %209 = stablehlo.reshape %208 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %210 = stablehlo.slice %207 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %211 = stablehlo.reshape %210 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %212 = stablehlo.add %211, %209 : tensor<!pf_babybear_mont>
    %213 = stablehlo.multiply %212, %212 : tensor<!pf_babybear_mont>
    %214 = stablehlo.multiply %212, %213 : tensor<!pf_babybear_mont>
    %215 = stablehlo.multiply %213, %213 : tensor<!pf_babybear_mont>
    %216 = stablehlo.multiply %214, %215 : tensor<!pf_babybear_mont>
    %217 = stablehlo.reshape %216 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %218 = stablehlo.slice %207 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %219 = stablehlo.concatenate %217, %218, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_19 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %220 = stablehlo.reduce(%219 init: %cst_19) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %221 = stablehlo.multiply %219, %cst_2 : tensor<16x!pf_babybear_mont>
    %222 = stablehlo.broadcast_in_dim %220, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %223 = stablehlo.add %221, %222 : tensor<16x!pf_babybear_mont>
    %224 = stablehlo.slice %cst_1 [11:12] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %225 = stablehlo.reshape %224 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %226 = stablehlo.slice %223 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %227 = stablehlo.reshape %226 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %228 = stablehlo.add %227, %225 : tensor<!pf_babybear_mont>
    %229 = stablehlo.multiply %228, %228 : tensor<!pf_babybear_mont>
    %230 = stablehlo.multiply %228, %229 : tensor<!pf_babybear_mont>
    %231 = stablehlo.multiply %229, %229 : tensor<!pf_babybear_mont>
    %232 = stablehlo.multiply %230, %231 : tensor<!pf_babybear_mont>
    %233 = stablehlo.reshape %232 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %234 = stablehlo.slice %223 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %235 = stablehlo.concatenate %233, %234, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_20 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %236 = stablehlo.reduce(%235 init: %cst_20) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %237 = stablehlo.multiply %235, %cst_2 : tensor<16x!pf_babybear_mont>
    %238 = stablehlo.broadcast_in_dim %236, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %239 = stablehlo.add %237, %238 : tensor<16x!pf_babybear_mont>
    %240 = stablehlo.slice %cst_1 [12:13] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %241 = stablehlo.reshape %240 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %242 = stablehlo.slice %239 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %243 = stablehlo.reshape %242 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %244 = stablehlo.add %243, %241 : tensor<!pf_babybear_mont>
    %245 = stablehlo.multiply %244, %244 : tensor<!pf_babybear_mont>
    %246 = stablehlo.multiply %244, %245 : tensor<!pf_babybear_mont>
    %247 = stablehlo.multiply %245, %245 : tensor<!pf_babybear_mont>
    %248 = stablehlo.multiply %246, %247 : tensor<!pf_babybear_mont>
    %249 = stablehlo.reshape %248 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %250 = stablehlo.slice %239 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %251 = stablehlo.concatenate %249, %250, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_21 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %252 = stablehlo.reduce(%251 init: %cst_21) applies stablehlo.add across dimensions = [0] : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %253 = stablehlo.multiply %251, %cst_2 : tensor<16x!pf_babybear_mont>
    %254 = stablehlo.broadcast_in_dim %252, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %255 = stablehlo.add %253, %254 : tensor<16x!pf_babybear_mont>
    %256 = stablehlo.slice %cst_3 [0:1, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %257 = stablehlo.reshape %256 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %258 = stablehlo.add %255, %257 : tensor<16x!pf_babybear_mont>
    %259 = stablehlo.multiply %258, %258 : tensor<16x!pf_babybear_mont>
    %260 = stablehlo.multiply %258, %259 : tensor<16x!pf_babybear_mont>
    %261 = stablehlo.multiply %259, %259 : tensor<16x!pf_babybear_mont>
    %262 = stablehlo.multiply %260, %261 : tensor<16x!pf_babybear_mont>
    %263 = stablehlo.broadcast_in_dim %262, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %264 = stablehlo.broadcast_in_dim %263, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %265 = stablehlo.multiply %cst, %264 : tensor<16x16x!pf_babybear_mont>
    %cst_22 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %266 = stablehlo.reduce(%265 init: %cst_22) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %267 = stablehlo.slice %cst_3 [1:2, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %268 = stablehlo.reshape %267 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %269 = stablehlo.add %266, %268 : tensor<16x!pf_babybear_mont>
    %270 = stablehlo.multiply %269, %269 : tensor<16x!pf_babybear_mont>
    %271 = stablehlo.multiply %269, %270 : tensor<16x!pf_babybear_mont>
    %272 = stablehlo.multiply %270, %270 : tensor<16x!pf_babybear_mont>
    %273 = stablehlo.multiply %271, %272 : tensor<16x!pf_babybear_mont>
    %274 = stablehlo.broadcast_in_dim %273, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %276 = stablehlo.multiply %cst, %275 : tensor<16x16x!pf_babybear_mont>
    %cst_23 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %277 = stablehlo.reduce(%276 init: %cst_23) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %278 = stablehlo.slice %cst_3 [2:3, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %279 = stablehlo.reshape %278 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %280 = stablehlo.add %277, %279 : tensor<16x!pf_babybear_mont>
    %281 = stablehlo.multiply %280, %280 : tensor<16x!pf_babybear_mont>
    %282 = stablehlo.multiply %280, %281 : tensor<16x!pf_babybear_mont>
    %283 = stablehlo.multiply %281, %281 : tensor<16x!pf_babybear_mont>
    %284 = stablehlo.multiply %282, %283 : tensor<16x!pf_babybear_mont>
    %285 = stablehlo.broadcast_in_dim %284, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %286 = stablehlo.broadcast_in_dim %285, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %287 = stablehlo.multiply %cst, %286 : tensor<16x16x!pf_babybear_mont>
    %cst_24 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %288 = stablehlo.reduce(%287 init: %cst_24) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %289 = stablehlo.slice %cst_3 [3:4, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %290 = stablehlo.reshape %289 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %291 = stablehlo.add %288, %290 : tensor<16x!pf_babybear_mont>
    %292 = stablehlo.multiply %291, %291 : tensor<16x!pf_babybear_mont>
    %293 = stablehlo.multiply %291, %292 : tensor<16x!pf_babybear_mont>
    %294 = stablehlo.multiply %292, %292 : tensor<16x!pf_babybear_mont>
    %295 = stablehlo.multiply %293, %294 : tensor<16x!pf_babybear_mont>
    %296 = stablehlo.broadcast_in_dim %295, dims = [1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %298 = stablehlo.multiply %cst, %297 : tensor<16x16x!pf_babybear_mont>
    %cst_25 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %299 = stablehlo.reduce(%298 init: %cst_25) applies stablehlo.add across dimensions = [1] : (tensor<16x16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    return %299 : tensor<16x!pf_babybear_mont>
  }
}
