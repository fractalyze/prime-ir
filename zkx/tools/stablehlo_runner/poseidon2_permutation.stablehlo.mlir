!pf_babybear_mont = !field.pf<2013265921 : i32, true>
module @jit_permute attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x!pf_babybear_mont>) -> (tensor<16x!pf_babybear_mont> {jax.result_info = "result"}) {
    %cst = stablehlo.constant() <{value = dense<1476395013> : tensor<1xi32>}> : () -> tensor<1x!pf_babybear_mont>
    %cst_0 = stablehlo.constant() <{value = dense<[268435454, 536870908, 134217727, 805306362, 1073741816, 1879048194, 1207959559, 939524105, 16777216, 1073741824, 536870912, 32, 1996488705, 1744830465, 2013265889]> : tensor<15xi32>}> : () -> tensor<15x!pf_babybear_mont>
    %cst_1 = stablehlo.constant() <{value = dense<[250494022, 528496384, 1472966118, 977089650, 1885890237, 1094557811, 147492661, 664163003, 398852570, 336233633, 1628648315, 888594966, 586791090]> : tensor<13xi32>}> : () -> tensor<13x!pf_babybear_mont>
    %c = stablehlo.constant dense<[true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]> : tensor<16xi1>
    %cst_2 = stablehlo.constant() <{value = dense<[[1582131512, 1899519471, 1641921850, 462688640, 1293997949, 1380417575, 1932416963, 283521298, 1016708647, 35751290, 1270782647, 851730739, 795004022, 929571430, 523703523, 1593957757], [895976710, 1742343460, 917700746, 1516725708, 1170237629, 785693164, 613651155, 352999196, 678775274, 1005433272, 1704854670, 1174551920, 508930349, 530338447, 1327158816, 1417652352], [1153538870, 583201050, 397833841, 1440603828, 454600685, 174490638, 171758601, 1998476616, 1403697810, 1807736944, 450348306, 1458895865, 787037868, 1063762964, 1987002214, 481645916], [1231767638, 1323639433, 238360103, 2012412459, 1024945356, 1108359895, 1284135849, 606928406, 1021455954, 719347978, 659671051, 769588663, 805534062, 592213995, 1752728055, 663410947]]> : tensor<4x16xi32>}> : () -> tensor<4x16x!pf_babybear_mont>
    %cst_3 = stablehlo.constant() <{value = dense<[[999830298, 304461056, 552699684, 450698925, 667466464, 1736509752, 1327760865, 1153241151, 816675655, 1076172858, 1914832527, 1668723429, 1365579850, 975704528, 1031625628, 1393317533], [1554700828, 1023828605, 1610378860, 347744760, 1909572073, 739227895, 428565985, 633143046, 121797685, 94048546, 1369350241, 1250010422, 114268841, 515033604, 49052844, 1962329907], [1380892638, 1860017417, 64711457, 9758460, 1681838395, 710850601, 1020228997, 1414164790, 1531515535, 36158805, 713604525, 89935127, 1870801994, 395985906, 1122769045, 1760811055], [819787042, 134654834, 1755145179, 18433016, 1701878989, 1782339297, 1483861396, 962480061, 1857590724, 222440409, 63223417, 515206622, 1348364213, 973414686, 1591066884, 705852913]]> : tensor<4x16xi32>}> : () -> tensor<4x16x!pf_babybear_mont>
    %0 = stablehlo.concatenate %cst, %cst_0, dim = 0 : (tensor<1x!pf_babybear_mont>, tensor<15x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %cst_4 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %1 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<13x15x!pf_babybear_mont>
    %2 = stablehlo.reshape %cst_1 : (tensor<13x!pf_babybear_mont>) -> tensor<13x1x!pf_babybear_mont>
    %3 = stablehlo.concatenate %2, %1, dim = 1 : (tensor<13x1x!pf_babybear_mont>, tensor<13x15x!pf_babybear_mont>) -> tensor<13x16x!pf_babybear_mont>
    %cst_5 = stablehlo.constant() <{value = dense<268435454> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %4 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<!pf_babybear_mont>) -> tensor<16x16x!pf_babybear_mont>
    %5 = stablehlo.reshape %arg0 : (tensor<16x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %6 = stablehlo.slice %5 [0:4, 0:1] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %7 = stablehlo.reshape %6 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %8 = stablehlo.slice %5 [0:4, 1:2] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %9 = stablehlo.reshape %8 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %10 = stablehlo.slice %5 [0:4, 2:3] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %11 = stablehlo.reshape %10 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %12 = stablehlo.slice %5 [0:4, 3:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %13 = stablehlo.reshape %12 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %14 = stablehlo.add %7, %9 : tensor<4x!pf_babybear_mont>
    %15 = stablehlo.add %11, %13 : tensor<4x!pf_babybear_mont>
    %16 = stablehlo.add %14, %15 : tensor<4x!pf_babybear_mont>
    %17 = stablehlo.add %16, %9 : tensor<4x!pf_babybear_mont>
    %18 = stablehlo.add %16, %13 : tensor<4x!pf_babybear_mont>
    %19 = stablehlo.add %7, %7 : tensor<4x!pf_babybear_mont>
    %20 = stablehlo.add %11, %11 : tensor<4x!pf_babybear_mont>
    %21 = stablehlo.add %17, %14 : tensor<4x!pf_babybear_mont>
    %22 = stablehlo.add %17, %20 : tensor<4x!pf_babybear_mont>
    %23 = stablehlo.add %18, %15 : tensor<4x!pf_babybear_mont>
    %24 = stablehlo.add %18, %19 : tensor<4x!pf_babybear_mont>
    %25 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %26 = stablehlo.broadcast_in_dim %22, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %27 = stablehlo.broadcast_in_dim %23, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %28 = stablehlo.broadcast_in_dim %24, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
    %29 = stablehlo.concatenate %25, %26, %27, %28, dim = 1 : (tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %30 = stablehlo.reshape %29 : (tensor<4x4x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %31 = stablehlo.reshape %30 : (tensor<16x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %cst_6 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
    %32 = stablehlo.reduce(%31 init: %cst_6) applies stablehlo.add across dimensions = [0] : (tensor<4x4x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
    %33 = stablehlo.broadcast_in_dim %32, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
    %34 = stablehlo.add %31, %33 : tensor<4x4x!pf_babybear_mont>
    %35 = stablehlo.reshape %34 : (tensor<4x4x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %36:3 = stablehlo.while(%iterArg = %cst_2, %iterArg_10 = %c_7, %iterArg_11 = %35) : tensor<4x16x!pf_babybear_mont>, tensor<i32>, tensor<16x!pf_babybear_mont>
     cond {
      %c_12 = stablehlo.constant dense<4> : tensor<i32>
      %39 = stablehlo.compare  LT, %iterArg_10, %c_12 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %39 : tensor<i1>
    } do {
      %c_12 = stablehlo.constant dense<0> : tensor<i32>
      %39 = stablehlo.compare  LT, %iterArg_10, %c_12 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_13 = stablehlo.constant dense<4> : tensor<i32>
      %40 = stablehlo.add %iterArg_10, %c_13 : tensor<i32>
      %41 = stablehlo.select %39, %40, %iterArg_10 : tensor<i1>, tensor<i32>
      %c_14 = stablehlo.constant dense<0> : tensor<i32>
      %42 = stablehlo.dynamic_slice %iterArg, %41, %c_14, sizes = [1, 16] : (tensor<4x16x!pf_babybear_mont>, tensor<i32>, tensor<i32>) -> tensor<1x16x!pf_babybear_mont>
      %43 = stablehlo.reshape %42 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %44 = stablehlo.add %iterArg_11, %43 : tensor<16x!pf_babybear_mont>
      %45 = stablehlo.multiply %44, %44 : tensor<16x!pf_babybear_mont>
      %46 = stablehlo.multiply %44, %45 : tensor<16x!pf_babybear_mont>
      %47 = stablehlo.multiply %45, %45 : tensor<16x!pf_babybear_mont>
      %48 = stablehlo.multiply %46, %47 : tensor<16x!pf_babybear_mont>
      %49 = stablehlo.reshape %48 : (tensor<16x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
      %50 = stablehlo.slice %49 [0:4, 0:1] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %51 = stablehlo.reshape %50 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %52 = stablehlo.slice %49 [0:4, 1:2] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %53 = stablehlo.reshape %52 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %54 = stablehlo.slice %49 [0:4, 2:3] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %55 = stablehlo.reshape %54 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %56 = stablehlo.slice %49 [0:4, 3:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %57 = stablehlo.reshape %56 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %58 = stablehlo.add %51, %53 : tensor<4x!pf_babybear_mont>
      %59 = stablehlo.add %55, %57 : tensor<4x!pf_babybear_mont>
      %60 = stablehlo.add %58, %59 : tensor<4x!pf_babybear_mont>
      %61 = stablehlo.add %60, %53 : tensor<4x!pf_babybear_mont>
      %62 = stablehlo.add %60, %57 : tensor<4x!pf_babybear_mont>
      %63 = stablehlo.add %61, %58 : tensor<4x!pf_babybear_mont>
      %64 = stablehlo.add %55, %55 : tensor<4x!pf_babybear_mont>
      %65 = stablehlo.add %61, %64 : tensor<4x!pf_babybear_mont>
      %66 = stablehlo.add %62, %59 : tensor<4x!pf_babybear_mont>
      %67 = stablehlo.add %51, %51 : tensor<4x!pf_babybear_mont>
      %68 = stablehlo.add %62, %67 : tensor<4x!pf_babybear_mont>
      %69 = stablehlo.broadcast_in_dim %63, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %70 = stablehlo.broadcast_in_dim %65, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %71 = stablehlo.broadcast_in_dim %66, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %72 = stablehlo.broadcast_in_dim %68, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %73 = stablehlo.concatenate %69, %70, %71, %72, dim = 1 : (tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
      %74 = stablehlo.reshape %73 : (tensor<4x4x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %75 = stablehlo.reshape %74 : (tensor<16x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
      %cst_15 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
      %76 = stablehlo.reduce(%75 init: %cst_15) applies stablehlo.add across dimensions = [0] : (tensor<4x4x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %77 = stablehlo.broadcast_in_dim %76, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
      %78 = stablehlo.add %75, %77 : tensor<4x4x!pf_babybear_mont>
      %79 = stablehlo.reshape %78 : (tensor<4x4x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %c_16 = stablehlo.constant dense<1> : tensor<i32>
      %80 = stablehlo.add %iterArg_10, %c_16 : tensor<i32>
      stablehlo.return %iterArg, %80, %79 : tensor<4x16x!pf_babybear_mont>, tensor<i32>, tensor<16x!pf_babybear_mont>
    }
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %37:6 = stablehlo.while(%iterArg = %3, %iterArg_10 = %c, %iterArg_11 = %0, %iterArg_12 = %4, %iterArg_13 = %c_8, %iterArg_14 = %36#2) : tensor<13x16x!pf_babybear_mont>, tensor<16xi1>, tensor<16x!pf_babybear_mont>, tensor<16x16x!pf_babybear_mont>, tensor<i32>, tensor<16x!pf_babybear_mont>
     cond {
      %c_15 = stablehlo.constant dense<13> : tensor<i32>
      %39 = stablehlo.compare  LT, %iterArg_13, %c_15 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %39 : tensor<i1>
    } do {
      %c_15 = stablehlo.constant dense<0> : tensor<i32>
      %39 = stablehlo.compare  LT, %iterArg_13, %c_15 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_16 = stablehlo.constant dense<13> : tensor<i32>
      %40 = stablehlo.add %iterArg_13, %c_16 : tensor<i32>
      %41 = stablehlo.select %39, %40, %iterArg_13 : tensor<i1>, tensor<i32>
      %c_17 = stablehlo.constant dense<0> : tensor<i32>
      %42 = stablehlo.dynamic_slice %iterArg, %41, %c_17, sizes = [1, 16] : (tensor<13x16x!pf_babybear_mont>, tensor<i32>, tensor<i32>) -> tensor<1x16x!pf_babybear_mont>
      %43 = stablehlo.reshape %42 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %44 = stablehlo.add %iterArg_14, %43 : tensor<16x!pf_babybear_mont>
      %45 = stablehlo.multiply %44, %44 : tensor<16x!pf_babybear_mont>
      %46 = stablehlo.multiply %44, %45 : tensor<16x!pf_babybear_mont>
      %47 = stablehlo.multiply %45, %45 : tensor<16x!pf_babybear_mont>
      %48 = stablehlo.multiply %46, %47 : tensor<16x!pf_babybear_mont>
      %49 = func.call @_where(%iterArg_10, %48, %44) : (tensor<16xi1>, tensor<16x!pf_babybear_mont>, tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %50 = stablehlo.multiply %49, %iterArg_11 : tensor<16x!pf_babybear_mont>
      %51 = "stablehlo.dot_general"(%iterArg_12, %49) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<16x16x!pf_babybear_mont>, tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %52 = stablehlo.add %50, %51 : tensor<16x!pf_babybear_mont>
      %c_18 = stablehlo.constant dense<1> : tensor<i32>
      %53 = stablehlo.add %iterArg_13, %c_18 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_10, %iterArg_11, %iterArg_12, %53, %52 : tensor<13x16x!pf_babybear_mont>, tensor<16xi1>, tensor<16x!pf_babybear_mont>, tensor<16x16x!pf_babybear_mont>, tensor<i32>, tensor<16x!pf_babybear_mont>
    }
    %c_9 = stablehlo.constant dense<0> : tensor<i32>
    %38:3 = stablehlo.while(%iterArg = %cst_3, %iterArg_10 = %c_9, %iterArg_11 = %37#5) : tensor<4x16x!pf_babybear_mont>, tensor<i32>, tensor<16x!pf_babybear_mont>
     cond {
      %c_12 = stablehlo.constant dense<4> : tensor<i32>
      %39 = stablehlo.compare  LT, %iterArg_10, %c_12 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %39 : tensor<i1>
    } do {
      %c_12 = stablehlo.constant dense<0> : tensor<i32>
      %39 = stablehlo.compare  LT, %iterArg_10, %c_12 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %c_13 = stablehlo.constant dense<4> : tensor<i32>
      %40 = stablehlo.add %iterArg_10, %c_13 : tensor<i32>
      %41 = stablehlo.select %39, %40, %iterArg_10 : tensor<i1>, tensor<i32>
      %c_14 = stablehlo.constant dense<0> : tensor<i32>
      %42 = stablehlo.dynamic_slice %iterArg, %41, %c_14, sizes = [1, 16] : (tensor<4x16x!pf_babybear_mont>, tensor<i32>, tensor<i32>) -> tensor<1x16x!pf_babybear_mont>
      %43 = stablehlo.reshape %42 : (tensor<1x16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %44 = stablehlo.add %iterArg_11, %43 : tensor<16x!pf_babybear_mont>
      %45 = stablehlo.multiply %44, %44 : tensor<16x!pf_babybear_mont>
      %46 = stablehlo.multiply %44, %45 : tensor<16x!pf_babybear_mont>
      %47 = stablehlo.multiply %45, %45 : tensor<16x!pf_babybear_mont>
      %48 = stablehlo.multiply %46, %47 : tensor<16x!pf_babybear_mont>
      %49 = stablehlo.reshape %48 : (tensor<16x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
      %50 = stablehlo.slice %49 [0:4, 0:1] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %51 = stablehlo.reshape %50 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %52 = stablehlo.slice %49 [0:4, 1:2] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %53 = stablehlo.reshape %52 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %54 = stablehlo.slice %49 [0:4, 2:3] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %55 = stablehlo.reshape %54 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %56 = stablehlo.slice %49 [0:4, 3:4] : (tensor<4x4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %57 = stablehlo.reshape %56 : (tensor<4x1x!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %58 = stablehlo.add %51, %53 : tensor<4x!pf_babybear_mont>
      %59 = stablehlo.add %55, %57 : tensor<4x!pf_babybear_mont>
      %60 = stablehlo.add %58, %59 : tensor<4x!pf_babybear_mont>
      %61 = stablehlo.add %60, %53 : tensor<4x!pf_babybear_mont>
      %62 = stablehlo.add %60, %57 : tensor<4x!pf_babybear_mont>
      %63 = stablehlo.add %61, %58 : tensor<4x!pf_babybear_mont>
      %64 = stablehlo.add %55, %55 : tensor<4x!pf_babybear_mont>
      %65 = stablehlo.add %61, %64 : tensor<4x!pf_babybear_mont>
      %66 = stablehlo.add %62, %59 : tensor<4x!pf_babybear_mont>
      %67 = stablehlo.add %51, %51 : tensor<4x!pf_babybear_mont>
      %68 = stablehlo.add %62, %67 : tensor<4x!pf_babybear_mont>
      %69 = stablehlo.broadcast_in_dim %63, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %70 = stablehlo.broadcast_in_dim %65, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %71 = stablehlo.broadcast_in_dim %66, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %72 = stablehlo.broadcast_in_dim %68, dims = [0] : (tensor<4x!pf_babybear_mont>) -> tensor<4x1x!pf_babybear_mont>
      %73 = stablehlo.concatenate %69, %70, %71, %72, dim = 1 : (tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>, tensor<4x1x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
      %74 = stablehlo.reshape %73 : (tensor<4x4x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %75 = stablehlo.reshape %74 : (tensor<16x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
      %cst_15 = stablehlo.constant() <{value = dense<0> : tensor<i32>}> : () -> tensor<!pf_babybear_mont>
      %76 = stablehlo.reduce(%75 init: %cst_15) applies stablehlo.add across dimensions = [0] : (tensor<4x4x!pf_babybear_mont>, tensor<!pf_babybear_mont>) -> tensor<4x!pf_babybear_mont>
      %77 = stablehlo.broadcast_in_dim %76, dims = [1] : (tensor<4x!pf_babybear_mont>) -> tensor<4x4x!pf_babybear_mont>
      %78 = stablehlo.add %75, %77 : tensor<4x4x!pf_babybear_mont>
      %79 = stablehlo.reshape %78 : (tensor<4x4x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont>
      %c_16 = stablehlo.constant dense<1> : tensor<i32>
      %80 = stablehlo.add %iterArg_10, %c_16 : tensor<i32>
      stablehlo.return %iterArg, %80, %79 : tensor<4x16x!pf_babybear_mont>, tensor<i32>, tensor<16x!pf_babybear_mont>
    }
    return %38#2 : tensor<16x!pf_babybear_mont>
  }
  func.func private @_where(%arg0: tensor<16xi1>, %arg1: tensor<16x!pf_babybear_mont>, %arg2: tensor<16x!pf_babybear_mont>) -> tensor<16x!pf_babybear_mont> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<16xi1>, tensor<16x!pf_babybear_mont>
    return %0 : tensor<16x!pf_babybear_mont>
  }
}
