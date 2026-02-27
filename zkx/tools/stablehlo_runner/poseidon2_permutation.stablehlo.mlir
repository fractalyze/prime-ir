!pf_babybear_mont = !field.pf<2013265921 : i32, true>
module @jit_permute attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x!pf_babybear_mont>) -> (tensor<16x!pf_babybear_mont> {jax.result_info = "result"}) {
    %0 = call @permute(%arg0) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    return %0 : tensor<16x!pf_babybear_mont>
  }
  func.func private @permute(%arg0: tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont> {
    %cst = stablehlo.constant() <{value = dense<[[999830298, 304461056, 552699684, 450698925, 667466464, 1736509752, 1327760865, 1153241151, 816675655, 1076172858, 1914832527, 1668723429, 1365579850, 975704528, 1031625628, 1393317533], [1554700828, 1023828605, 1610378860, 347744760, 1909572073, 739227895, 428565985, 633143046, 121797685, 94048546, 1369350241, 1250010422, 114268841, 515033604, 49052844, 1962329907], [1380892638, 1860017417, 64711457, 9758460, 1681838395, 710850601, 1020228997, 1414164790, 1531515535, 36158805, 713604525, 89935127, 1870801994, 395985906, 1122769045, 1760811055], [819787042, 134654834, 1755145179, 18433016, 1701878989, 1782339297, 1483861396, 962480061, 1857590724, 222440409, 63223417, 515206622, 1348364213, 973414686, 1591066884, 705852913]]> : tensor<4x16xi32>}> : () -> tensor<4x16x!pf_babybear_mont>
    %0 = call @permute_state_initial(%arg0) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1 = call @permute_state(%0) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2 = call @permute_state_terminal(%1, %cst) : (tensor<16x!pf_babybear_mont>, tensor<4x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    return %2 : tensor<16x!pf_babybear_mont>
  }
  func.func private @permute_state_initial(%arg0: tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont> {
    %cst = stablehlo.constant() <{value = dense<[[1582131512, 1899519471, 1641921850, 462688640, 1293997949, 1380417575, 1932416963, 283521298, 1016708647, 35751290, 1270782647, 851730739, 795004022, 929571430, 523703523, 1593957757], [895976710, 1742343460, 917700746, 1516725708, 1170237629, 785693164, 613651155, 352999196, 678775274, 1005433272, 1704854670, 1174551920, 508930349, 530338447, 1327158816, 1417652352], [1153538870, 583201050, 397833841, 1440603828, 454600685, 174490638, 171758601, 1998476616, 1403697810, 1807736944, 450348306, 1458895865, 787037868, 1063762964, 1987002214, 481645916], [1231767638, 1323639433, 238360103, 2012412459, 1024945356, 1108359895, 1284135849, 606928406, 1021455954, 719347978, 659671051, 769588663, 805534062, 592213995, 1752728055, 663410947]]> : tensor<4x16xi32>}> : () -> tensor<4x16x!pf_babybear_mont>
    %0 = call @mds_light_permutation(%arg0) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %1 = call @permute_state_terminal(%0, %cst) : (tensor<16x!pf_babybear_mont>, tensor<4x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    return %1 : tensor<16x!pf_babybear_mont>
  }
  func.func private @mds_light_permutation(%arg0: tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont> {
    %0 = stablehlo.reshape %arg0 : (tensor<16x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %2 = stablehlo.slice %1 [0:1, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %3 = stablehlo.reshape %2 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %4 = stablehlo.slice %1 [1:2, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %5 = stablehlo.reshape %4 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %6 = stablehlo.slice %1 [2:3, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %7 = stablehlo.reshape %6 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %8 = stablehlo.slice %1 [3:4, 0:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<1x4x!pf_babybear_mont>
    %9 = stablehlo.reshape %8 : (tensor<1x4x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %10 = stablehlo.add %3, %5 : tensor<4x!pf_babybear_mont>
    %11 = stablehlo.add %7, %9 : tensor<4x!pf_babybear_mont>
    %12 = stablehlo.add %10, %11 : tensor<4x!pf_babybear_mont>
    %13 = stablehlo.add %12, %5 : tensor<4x!pf_babybear_mont>
    %14 = stablehlo.add %12, %9 : tensor<4x!pf_babybear_mont>
    %cst = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %15 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %16 = stablehlo.multiply %3, %15 : tensor<4x!pf_babybear_mont>
    %cst_0 = stablehlo.constant() <{value = dense<536870908> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %17 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %18 = stablehlo.multiply %7, %17 : tensor<4x!pf_babybear_mont>
    %19 = stablehlo.add %13, %10 : tensor<4x!pf_babybear_mont>
    %20 = stablehlo.add %13, %18 : tensor<4x!pf_babybear_mont>
    %21 = stablehlo.add %14, %11 : tensor<4x!pf_babybear_mont>
    %22 = stablehlo.add %14, %16 : tensor<4x!pf_babybear_mont>
    %23 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %24 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %25 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %26 = stablehlo.broadcast_in_dim %22, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %27 = stablehlo.concatenate %23, %24, %25, %26, dim = 1 : (tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %28 = stablehlo.reshape %27 : (tensor<4x4x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %29 = stablehlo.slice %28 [0:4] : (tensor<16x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %30 = stablehlo.slice %28 [4:8] : (tensor<16x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %31 = stablehlo.add %29, %30 : tensor<4x!pf_babybear_mont>
    %32 = stablehlo.slice %28 [8:12] : (tensor<16x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %33 = stablehlo.add %31, %32 : tensor<4x!pf_babybear_mont>
    %34 = stablehlo.slice %28 [12:16] : (tensor<16x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %35 = stablehlo.add %33, %34 : tensor<4x!pf_babybear_mont>
    %36 = stablehlo.reshape %28 : (tensor<16x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %37 = stablehlo.transpose %36, dims = [1, 0] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %38 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %40 = stablehlo.add %37, %39 : tensor<4x4x!pf_babybear_mont>
    %41 = stablehlo.transpose %40, dims = [1, 0] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %42 = stablehlo.reshape %41 : (tensor<4x4x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    return %42 : tensor<16x!pf_babybear_mont>
  }
  func.func private @permute_state_terminal(%arg0: tensor<16x!pf_babybear_mont>, %arg1: tensor<4x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont> {
    %0 = stablehlo.slice %arg1 [0:1, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %1 = stablehlo.reshape %0 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %2 = stablehlo.add %arg0, %1 : tensor<16x!pf_babybear_mont>
    %3 = stablehlo.multiply %2, %2 : tensor<16x!pf_babybear_mont>
    %4 = stablehlo.multiply %2, %3 : tensor<16x!pf_babybear_mont>
    %5 = stablehlo.multiply %3, %3 : tensor<16x!pf_babybear_mont>
    %6 = stablehlo.multiply %4, %5 : tensor<16x!pf_babybear_mont>
    %7 = call @mds_light_permutation(%6) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %8 = stablehlo.slice %arg1 [1:2, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %9 = stablehlo.reshape %8 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %10 = stablehlo.add %7, %9 : tensor<16x!pf_babybear_mont>
    %11 = stablehlo.multiply %10, %10 : tensor<16x!pf_babybear_mont>
    %12 = stablehlo.multiply %10, %11 : tensor<16x!pf_babybear_mont>
    %13 = stablehlo.multiply %11, %11 : tensor<16x!pf_babybear_mont>
    %14 = stablehlo.multiply %12, %13 : tensor<16x!pf_babybear_mont>
    %15 = call @mds_light_permutation(%14) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %16 = stablehlo.slice %arg1 [2:3, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %17 = stablehlo.reshape %16 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %18 = stablehlo.add %15, %17 : tensor<16x!pf_babybear_mont>
    %19 = stablehlo.multiply %18, %18 : tensor<16x!pf_babybear_mont>
    %20 = stablehlo.multiply %18, %19 : tensor<16x!pf_babybear_mont>
    %21 = stablehlo.multiply %19, %19 : tensor<16x!pf_babybear_mont>
    %22 = stablehlo.multiply %20, %21 : tensor<16x!pf_babybear_mont>
    %23 = call @mds_light_permutation(%22) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %24 = stablehlo.slice %arg1 [3:4, 0:16] : (tensor<4x16x!pf_babybear_mont>) -> tensor<1x16x!pf_babybear_mont>
    %25 = stablehlo.reshape %24 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %26 = stablehlo.add %23, %25 : tensor<16x!pf_babybear_mont>
    %27 = stablehlo.multiply %26, %26 : tensor<16x!pf_babybear_mont>
    %28 = stablehlo.multiply %26, %27 : tensor<16x!pf_babybear_mont>
    %29 = stablehlo.multiply %27, %27 : tensor<16x!pf_babybear_mont>
    %30 = stablehlo.multiply %28, %29 : tensor<16x!pf_babybear_mont>
    %31 = call @mds_light_permutation(%30) : (tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    return %31 : tensor<16x!pf_babybear_mont>
  }
  func.func private @permute_state(%arg0: tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont> {
    %cst = stablehlo.constant() <{value = dense<[250494022, 528496384, 1472966118, 977089650, 1885890237, 1094557811, 147492661, 664163003, 398852570, 336233633, 1628648315, 888594966, 586791090]> : tensor<13xi32>}> : () -> tensor<13x!pf_babybear_mont>
    %0 = stablehlo.slice %cst [0:1] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %1 = stablehlo.reshape %0 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %3 = stablehlo.reshape %2 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %4 = call @add_rc_and_sbox(%3, %1) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %5 = stablehlo.slice %arg0 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_0 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %7 = stablehlo.add %6, %4 : tensor<!pf_babybear_mont>
    %8 = stablehlo.subtract %6, %4 : tensor<!pf_babybear_mont>
    %9 = stablehlo.reshape %8 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %10 = stablehlo.slice %arg0 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %12 = call @internal_layer_mat_mul(%11, %7) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %13 = stablehlo.slice %cst [1:2] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %14 = stablehlo.reshape %13 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %15 = stablehlo.slice %12 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %16 = stablehlo.reshape %15 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %17 = call @add_rc_and_sbox(%16, %14) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %18 = stablehlo.slice %12 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_1 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %19 = stablehlo.reduce(%18 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %20 = stablehlo.add %19, %17 : tensor<!pf_babybear_mont>
    %21 = stablehlo.subtract %19, %17 : tensor<!pf_babybear_mont>
    %22 = stablehlo.reshape %21 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %23 = stablehlo.slice %12 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %24 = stablehlo.concatenate %22, %23, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %25 = call @internal_layer_mat_mul(%24, %20) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %26 = stablehlo.slice %cst [2:3] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %27 = stablehlo.reshape %26 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %28 = stablehlo.slice %25 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %29 = stablehlo.reshape %28 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %30 = call @add_rc_and_sbox(%29, %27) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %31 = stablehlo.slice %25 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_2 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %32 = stablehlo.reduce(%31 init: %cst_2) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %33 = stablehlo.add %32, %30 : tensor<!pf_babybear_mont>
    %34 = stablehlo.subtract %32, %30 : tensor<!pf_babybear_mont>
    %35 = stablehlo.reshape %34 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %36 = stablehlo.slice %25 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %37 = stablehlo.concatenate %35, %36, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %38 = call @internal_layer_mat_mul(%37, %33) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %39 = stablehlo.slice %cst [3:4] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %40 = stablehlo.reshape %39 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %41 = stablehlo.slice %38 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %42 = stablehlo.reshape %41 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %43 = call @add_rc_and_sbox(%42, %40) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %44 = stablehlo.slice %38 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_3 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %45 = stablehlo.reduce(%44 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %46 = stablehlo.add %45, %43 : tensor<!pf_babybear_mont>
    %47 = stablehlo.subtract %45, %43 : tensor<!pf_babybear_mont>
    %48 = stablehlo.reshape %47 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %49 = stablehlo.slice %38 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %50 = stablehlo.concatenate %48, %49, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %51 = call @internal_layer_mat_mul(%50, %46) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %52 = stablehlo.slice %cst [4:5] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %53 = stablehlo.reshape %52 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %54 = stablehlo.slice %51 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %55 = stablehlo.reshape %54 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %56 = call @add_rc_and_sbox(%55, %53) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %57 = stablehlo.slice %51 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_4 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %58 = stablehlo.reduce(%57 init: %cst_4) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %59 = stablehlo.add %58, %56 : tensor<!pf_babybear_mont>
    %60 = stablehlo.subtract %58, %56 : tensor<!pf_babybear_mont>
    %61 = stablehlo.reshape %60 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %62 = stablehlo.slice %51 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %63 = stablehlo.concatenate %61, %62, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %64 = call @internal_layer_mat_mul(%63, %59) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %65 = stablehlo.slice %cst [5:6] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %66 = stablehlo.reshape %65 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %67 = stablehlo.slice %64 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %68 = stablehlo.reshape %67 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %69 = call @add_rc_and_sbox(%68, %66) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %70 = stablehlo.slice %64 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_5 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %71 = stablehlo.reduce(%70 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %72 = stablehlo.add %71, %69 : tensor<!pf_babybear_mont>
    %73 = stablehlo.subtract %71, %69 : tensor<!pf_babybear_mont>
    %74 = stablehlo.reshape %73 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %75 = stablehlo.slice %64 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %76 = stablehlo.concatenate %74, %75, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %77 = call @internal_layer_mat_mul(%76, %72) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %78 = stablehlo.slice %cst [6:7] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %79 = stablehlo.reshape %78 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %80 = stablehlo.slice %77 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %81 = stablehlo.reshape %80 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %82 = call @add_rc_and_sbox(%81, %79) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %83 = stablehlo.slice %77 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_6 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %84 = stablehlo.reduce(%83 init: %cst_6) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %85 = stablehlo.add %84, %82 : tensor<!pf_babybear_mont>
    %86 = stablehlo.subtract %84, %82 : tensor<!pf_babybear_mont>
    %87 = stablehlo.reshape %86 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %88 = stablehlo.slice %77 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %89 = stablehlo.concatenate %87, %88, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %90 = call @internal_layer_mat_mul(%89, %85) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %91 = stablehlo.slice %cst [7:8] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %92 = stablehlo.reshape %91 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %93 = stablehlo.slice %90 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %94 = stablehlo.reshape %93 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %95 = call @add_rc_and_sbox(%94, %92) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %96 = stablehlo.slice %90 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_7 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %97 = stablehlo.reduce(%96 init: %cst_7) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %98 = stablehlo.add %97, %95 : tensor<!pf_babybear_mont>
    %99 = stablehlo.subtract %97, %95 : tensor<!pf_babybear_mont>
    %100 = stablehlo.reshape %99 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %101 = stablehlo.slice %90 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %102 = stablehlo.concatenate %100, %101, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %103 = call @internal_layer_mat_mul(%102, %98) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %104 = stablehlo.slice %cst [8:9] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %105 = stablehlo.reshape %104 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %106 = stablehlo.slice %103 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %107 = stablehlo.reshape %106 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %108 = call @add_rc_and_sbox(%107, %105) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %109 = stablehlo.slice %103 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_8 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %110 = stablehlo.reduce(%109 init: %cst_8) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %111 = stablehlo.add %110, %108 : tensor<!pf_babybear_mont>
    %112 = stablehlo.subtract %110, %108 : tensor<!pf_babybear_mont>
    %113 = stablehlo.reshape %112 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %114 = stablehlo.slice %103 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %115 = stablehlo.concatenate %113, %114, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %116 = call @internal_layer_mat_mul(%115, %111) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %117 = stablehlo.slice %cst [9:10] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %118 = stablehlo.reshape %117 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %119 = stablehlo.slice %116 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %120 = stablehlo.reshape %119 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %121 = call @add_rc_and_sbox(%120, %118) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %122 = stablehlo.slice %116 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_9 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %123 = stablehlo.reduce(%122 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %124 = stablehlo.add %123, %121 : tensor<!pf_babybear_mont>
    %125 = stablehlo.subtract %123, %121 : tensor<!pf_babybear_mont>
    %126 = stablehlo.reshape %125 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %127 = stablehlo.slice %116 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %128 = stablehlo.concatenate %126, %127, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %129 = call @internal_layer_mat_mul(%128, %124) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %130 = stablehlo.slice %cst [10:11] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %131 = stablehlo.reshape %130 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %132 = stablehlo.slice %129 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %133 = stablehlo.reshape %132 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %134 = call @add_rc_and_sbox(%133, %131) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %135 = stablehlo.slice %129 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_10 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %136 = stablehlo.reduce(%135 init: %cst_10) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %137 = stablehlo.add %136, %134 : tensor<!pf_babybear_mont>
    %138 = stablehlo.subtract %136, %134 : tensor<!pf_babybear_mont>
    %139 = stablehlo.reshape %138 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %140 = stablehlo.slice %129 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %141 = stablehlo.concatenate %139, %140, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %142 = call @internal_layer_mat_mul(%141, %137) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %143 = stablehlo.slice %cst [11:12] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %144 = stablehlo.reshape %143 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %145 = stablehlo.slice %142 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %146 = stablehlo.reshape %145 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %147 = call @add_rc_and_sbox(%146, %144) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %148 = stablehlo.slice %142 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_11 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %149 = stablehlo.reduce(%148 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %150 = stablehlo.add %149, %147 : tensor<!pf_babybear_mont>
    %151 = stablehlo.subtract %149, %147 : tensor<!pf_babybear_mont>
    %152 = stablehlo.reshape %151 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %153 = stablehlo.slice %142 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %154 = stablehlo.concatenate %152, %153, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %155 = call @internal_layer_mat_mul(%154, %150) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %156 = stablehlo.slice %cst [12:13] : (tensor<13x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %157 = stablehlo.reshape %156 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %158 = stablehlo.slice %155 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %159 = stablehlo.reshape %158 : (tensor<1x!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %160 = call @add_rc_and_sbox(%159, %157) : (tensor<!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %161 = stablehlo.slice %155 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %cst_12 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %162 = stablehlo.reduce(%161 init: %cst_12) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont>
    %163 = stablehlo.add %162, %160 : tensor<!pf_babybear_mont>
    %164 = stablehlo.subtract %162, %160 : tensor<!pf_babybear_mont>
    %165 = stablehlo.reshape %164 : (tensor<!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %166 = stablehlo.slice %155 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %167 = stablehlo.concatenate %165, %166, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %168 = call @internal_layer_mat_mul(%167, %163) : (tensor<16x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    return %168 : tensor<16x!pf_babybear_mont>
  }
  func.func private @add_rc_and_sbox(%arg0: tensor<!pf_babybear_mont>, %arg1: tensor<!pf_babybear_mont>) -> tensor<!pf_babybear_mont> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<!pf_babybear_mont>
    %1 = stablehlo.multiply %0, %0 : tensor<!pf_babybear_mont>
    %2 = stablehlo.multiply %0, %1 : tensor<!pf_babybear_mont>
    %3 = stablehlo.multiply %1, %1 : tensor<!pf_babybear_mont>
    %4 = stablehlo.multiply %2, %3 : tensor<!pf_babybear_mont>
    return %4 : tensor<!pf_babybear_mont>
  }
  func.func private @internal_layer_mat_mul(%arg0: tensor<16x!pf_babybear_mont>, %arg1: tensor<!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont> {
    %cst = stablehlo.constant() <{value = dense<[268435454, 536870908, 134217727, 805306362, 1073741816, 1879048194, 1207959559, 939524105, 16777216, 1073741824, 536870912, 32, 1996488705, 1744830465, 2013265889]> : tensor<15xi32>}> : () -> tensor<15x!pf_babybear_mont>
    %0 = stablehlo.slice %arg0 [1:16] : (tensor<16x!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %1 = stablehlo.multiply %0, %cst : tensor<15x!pf_babybear_mont>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<15x!pf_babybear_mont>
    %3 = stablehlo.add %1, %2 : tensor<15x!pf_babybear_mont>
    %4 = stablehlo.slice %arg0 [0:1] : (tensor<16x!pf_babybear_mont>) -> tensor<1x!pf_babybear_mont>
    %5 = stablehlo.concatenate %4, %3, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    return %5 : tensor<16x!pf_babybear_mont>
  }
}
