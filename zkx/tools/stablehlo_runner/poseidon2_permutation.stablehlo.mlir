!pf_babybear_mont_std = !field.pf<2013265921 : i32>
module @jit_permute attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x!pf_babybear_mont_std>) -> (tensor<16x!pf_babybear_mont_std> {jax.result_info = "result"}) {
    %0 = call @permute(%arg0) : (tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %0 : tensor<16x!pf_babybear_mont_std>
  }
  func.func private @permute(%arg0: tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std> {
    %cst = stablehlo.constant() <{value = dense<[[1922082829, 1870549801, 1502529704, 1990744480, 1700391016, 1702593455, 321330495, 528965731, 183414327, 1886297254, 1178602734, 1923111974, 744004766, 549271463, 1781349648, 542259047], [1536158148, 715456982, 503426110, 340311124, 1558555932, 1226350925, 742828095, 1338992758, 1641600456, 1843351545, 301835475, 43203215, 386838401, 1520185679, 1235297680, 904680097], [1491801617, 1581784677, 913384905, 247083962, 532844013, 107190701, 213827818, 1979521776, 1358282574, 1681743681, 1867507480, 1530706910, 507181886, 695185447, 1172395131, 1250800299], [1503161625, 817684387, 498481458, 494676004, 1404253825, 108246855, 59414691, 744214112, 890862029, 1342765939, 1417398904, 1897591937, 1066647396, 1682806907, 1015795079, 1619482808]]> : tensor<4x16xui32>}> : () -> tensor<4x16x!pf_babybear_mont_std>
    %0 = call @permute_state_initial(%arg0) : (tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %1 = call @permute_state(%0) : (tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %2 = call @permute_state_terminal(%1, %cst) : (tensor<16x!pf_babybear_mont_std>, tensor<4x16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %2 : tensor<16x!pf_babybear_mont_std>
  }
  func.func private @permute_state_initial(%arg0: tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std> {
    %cst = stablehlo.constant() <{value = dense<[[1774958255, 1185780729, 1621102414, 1796380621, 588815102, 1932426223, 1925334750, 747903232, 89648862, 360728943, 977184635, 1425273457, 256487465, 1200041953, 572403254, 448208942], [1215789478, 944884184, 953948096, 547326025, 646827752, 889997530, 1536873262, 86189867, 1065944411, 32019634, 333311454, 456061748, 1963448500, 1827584334, 1391160226, 1348741381], [88424255, 104111868, 1763866748, 79691676, 1988915530, 1050669594, 359890076, 573163527, 222820492, 159256268, 669703072, 763177444, 889367200, 256335831, 704371273, 25886717], [51754520, 1833211857, 454499742, 1384520381, 777848065, 1053320300, 1851729162, 344647910, 401996362, 1046925956, 5351995, 1212119315, 754867989, 36972490, 751272725, 506915399]]> : tensor<4x16xui32>}> : () -> tensor<4x16x!pf_babybear_mont_std>
    %0 = call @mds_light_permutation(%arg0) : (tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %1 = call @permute_state_terminal(%0, %cst) : (tensor<16x!pf_babybear_mont_std>, tensor<4x16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %1 : tensor<16x!pf_babybear_mont_std>
  }
  func.func private @mds_light_permutation(%arg0: tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0:3 = stablehlo.while(%iterArg = %c_0, %iterArg_1 = %c, %iterArg_2 = %arg0) : tensor<i32>, tensor<i32>, tensor<16x!pf_babybear_mont_std>
     cond {
      %c_3 = stablehlo.constant dense<4> : tensor<i32>
      %15 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %15 : tensor<i1>
    } do {
      %15:2 = func.call @closed_call(%iterArg_1, %iterArg_2) : (tensor<i32>, tensor<16x!pf_babybear_mont_std>) -> (tensor<i32>, tensor<16x!pf_babybear_mont_std>)
      %c_3 = stablehlo.constant dense<1> : tensor<i32>
      %16 = stablehlo.add %iterArg, %c_3 : tensor<i32>
      stablehlo.return %16, %15#0, %15#1 : tensor<i32>, tensor<i32>, tensor<16x!pf_babybear_mont_std>
    }
    %1 = stablehlo.slice %0#2 [0:4] : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %2 = stablehlo.slice %0#2 [4:8] : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %3 = stablehlo.add %1, %2 : tensor<4x!pf_babybear_mont_std>
    %4 = stablehlo.slice %0#2 [8:12] : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %5 = stablehlo.add %3, %4 : tensor<4x!pf_babybear_mont_std>
    %6 = stablehlo.slice %0#2 [12:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %7 = stablehlo.add %5, %6 : tensor<4x!pf_babybear_mont_std>
    %8 = stablehlo.reshape %0#2 : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %9 = stablehlo.transpose %8, dims = [1, 0] : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %10 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<4x1x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %12 = stablehlo.add %9, %11 : tensor<4x4x!pf_babybear_mont_std>
    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %14 = stablehlo.reshape %13 : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %14 : tensor<16x!pf_babybear_mont_std>
  }
  func.func private @closed_call(%arg0: tensor<i32>, %arg1: tensor<16x!pf_babybear_mont_std>) -> (tensor<i32>, tensor<16x!pf_babybear_mont_std>) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.add %arg0, %c : tensor<i32>
    %c_0 = stablehlo.constant dense<4> : tensor<i32>
    %1 = stablehlo.multiply %arg0, %c_0 : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.compare  LT, %1, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.convert %1 : tensor<i32>
    %c_2 = stablehlo.constant dense<16> : tensor<i32>
    %4 = stablehlo.add %3, %c_2 : tensor<i32>
    %5 = stablehlo.select %2, %4, %1 : tensor<i1>, tensor<i32>
    %6 = stablehlo.dynamic_slice %arg1, %5, sizes = [1] : (tensor<16x!pf_babybear_mont_std>, tensor<i32>) -> tensor<1x!pf_babybear_mont_std>
    %7 = stablehlo.reshape %6 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %c_3 = stablehlo.constant dense<1> : tensor<i32>
    %8 = stablehlo.add %1, %c_3 : tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.compare  LT, %8, %c_4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %10 = stablehlo.convert %8 : tensor<i32>
    %c_5 = stablehlo.constant dense<16> : tensor<i32>
    %11 = stablehlo.add %10, %c_5 : tensor<i32>
    %12 = stablehlo.select %9, %11, %8 : tensor<i1>, tensor<i32>
    %13 = stablehlo.dynamic_slice %arg1, %12, sizes = [1] : (tensor<16x!pf_babybear_mont_std>, tensor<i32>) -> tensor<1x!pf_babybear_mont_std>
    %14 = stablehlo.reshape %13 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %c_6 = stablehlo.constant dense<2> : tensor<i32>
    %15 = stablehlo.add %1, %c_6 : tensor<i32>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.compare  LT, %15, %c_7 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %17 = stablehlo.convert %15 : tensor<i32>
    %c_8 = stablehlo.constant dense<16> : tensor<i32>
    %18 = stablehlo.add %17, %c_8 : tensor<i32>
    %19 = stablehlo.select %16, %18, %15 : tensor<i1>, tensor<i32>
    %20 = stablehlo.dynamic_slice %arg1, %19, sizes = [1] : (tensor<16x!pf_babybear_mont_std>, tensor<i32>) -> tensor<1x!pf_babybear_mont_std>
    %21 = stablehlo.reshape %20 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %c_9 = stablehlo.constant dense<3> : tensor<i32>
    %22 = stablehlo.add %1, %c_9 : tensor<i32>
    %c_10 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.compare  LT, %22, %c_10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %24 = stablehlo.convert %22 : tensor<i32>
    %c_11 = stablehlo.constant dense<16> : tensor<i32>
    %25 = stablehlo.add %24, %c_11 : tensor<i32>
    %26 = stablehlo.select %23, %25, %22 : tensor<i1>, tensor<i32>
    %27 = stablehlo.dynamic_slice %arg1, %26, sizes = [1] : (tensor<16x!pf_babybear_mont_std>, tensor<i32>) -> tensor<1x!pf_babybear_mont_std>
    %28 = stablehlo.reshape %27 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %29 = stablehlo.add %7, %14 : tensor<!pf_babybear_mont_std>
    %30 = stablehlo.add %21, %28 : tensor<!pf_babybear_mont_std>
    %31 = stablehlo.add %29, %30 : tensor<!pf_babybear_mont_std>
    %32 = stablehlo.add %31, %14 : tensor<!pf_babybear_mont_std>
    %33 = stablehlo.add %31, %28 : tensor<!pf_babybear_mont_std>
    %cst = stablehlo.constant() <{value = dense<2> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %34 = stablehlo.multiply %7, %cst : tensor<!pf_babybear_mont_std>
    %cst_12 = stablehlo.constant() <{value = dense<2> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %35 = stablehlo.multiply %21, %cst_12 : tensor<!pf_babybear_mont_std>
    %36 = stablehlo.add %32, %29 : tensor<!pf_babybear_mont_std>
    %37 = stablehlo.add %32, %35 : tensor<!pf_babybear_mont_std>
    %38 = stablehlo.add %33, %30 : tensor<!pf_babybear_mont_std>
    %39 = stablehlo.add %33, %34 : tensor<!pf_babybear_mont_std>
    %c_13 = stablehlo.constant dense<0> : tensor<i32>
    %40 = stablehlo.compare  LT, %1, %c_13 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_14 = stablehlo.constant dense<16> : tensor<i32>
    %41 = stablehlo.add %1, %c_14 : tensor<i32>
    %42 = stablehlo.select %40, %41, %1 : tensor<i1>, tensor<i32>
    %43 = stablehlo.convert %42 : tensor<i32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %45 = "stablehlo.scatter"(%arg1, %44, %36) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont_std>, %arg3: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %c_15 = stablehlo.constant dense<1> : tensor<i32>
    %46 = stablehlo.add %1, %c_15 : tensor<i32>
    %c_16 = stablehlo.constant dense<0> : tensor<i32>
    %47 = stablehlo.compare  LT, %46, %c_16 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_17 = stablehlo.constant dense<16> : tensor<i32>
    %48 = stablehlo.add %46, %c_17 : tensor<i32>
    %49 = stablehlo.select %47, %48, %46 : tensor<i1>, tensor<i32>
    %50 = stablehlo.convert %49 : tensor<i32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %52 = "stablehlo.scatter"(%45, %51, %37) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont_std>, %arg3: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %c_18 = stablehlo.constant dense<2> : tensor<i32>
    %53 = stablehlo.add %1, %c_18 : tensor<i32>
    %c_19 = stablehlo.constant dense<0> : tensor<i32>
    %54 = stablehlo.compare  LT, %53, %c_19 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_20 = stablehlo.constant dense<16> : tensor<i32>
    %55 = stablehlo.add %53, %c_20 : tensor<i32>
    %56 = stablehlo.select %54, %55, %53 : tensor<i1>, tensor<i32>
    %57 = stablehlo.convert %56 : tensor<i32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %59 = "stablehlo.scatter"(%52, %58, %38) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont_std>, %arg3: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %c_21 = stablehlo.constant dense<3> : tensor<i32>
    %60 = stablehlo.add %1, %c_21 : tensor<i32>
    %c_22 = stablehlo.constant dense<0> : tensor<i32>
    %61 = stablehlo.compare  LT, %60, %c_22 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c_23 = stablehlo.constant dense<16> : tensor<i32>
    %62 = stablehlo.add %60, %c_23 : tensor<i32>
    %63 = stablehlo.select %61, %62, %60 : tensor<i1>, tensor<i32>
    %64 = stablehlo.convert %63 : tensor<i32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %66 = "stablehlo.scatter"(%59, %65, %39) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont_std>, %arg3: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %0, %66 : tensor<i32>, tensor<16x!pf_babybear_mont_std>
  }
  func.func private @permute_state_terminal(%arg0: tensor<16x!pf_babybear_mont_std>, %arg1: tensor<4x16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std> {
    %0 = stablehlo.slice %arg1 [0:1, 0:16] : (tensor<4x16x!pf_babybear_mont_std>) -> tensor<1x16x!pf_babybear_mont_std>
    %1 = stablehlo.reshape %0 : (tensor<1x16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %2 = stablehlo.add %arg0, %1 : tensor<16x!pf_babybear_mont_std>
    %3 = stablehlo.multiply %2, %2 : tensor<16x!pf_babybear_mont_std>
    %4 = stablehlo.multiply %2, %3 : tensor<16x!pf_babybear_mont_std>
    %5 = stablehlo.multiply %3, %3 : tensor<16x!pf_babybear_mont_std>
    %6 = stablehlo.multiply %4, %5 : tensor<16x!pf_babybear_mont_std>
    %7 = call @mds_light_permutation(%6) : (tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %8 = stablehlo.slice %arg1 [1:2, 0:16] : (tensor<4x16x!pf_babybear_mont_std>) -> tensor<1x16x!pf_babybear_mont_std>
    %9 = stablehlo.reshape %8 : (tensor<1x16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %10 = stablehlo.add %7, %9 : tensor<16x!pf_babybear_mont_std>
    %11 = stablehlo.multiply %10, %10 : tensor<16x!pf_babybear_mont_std>
    %12 = stablehlo.multiply %10, %11 : tensor<16x!pf_babybear_mont_std>
    %13 = stablehlo.multiply %11, %11 : tensor<16x!pf_babybear_mont_std>
    %14 = stablehlo.multiply %12, %13 : tensor<16x!pf_babybear_mont_std>
    %15 = call @mds_light_permutation(%14) : (tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %16 = stablehlo.slice %arg1 [2:3, 0:16] : (tensor<4x16x!pf_babybear_mont_std>) -> tensor<1x16x!pf_babybear_mont_std>
    %17 = stablehlo.reshape %16 : (tensor<1x16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %18 = stablehlo.add %15, %17 : tensor<16x!pf_babybear_mont_std>
    %19 = stablehlo.multiply %18, %18 : tensor<16x!pf_babybear_mont_std>
    %20 = stablehlo.multiply %18, %19 : tensor<16x!pf_babybear_mont_std>
    %21 = stablehlo.multiply %19, %19 : tensor<16x!pf_babybear_mont_std>
    %22 = stablehlo.multiply %20, %21 : tensor<16x!pf_babybear_mont_std>
    %23 = call @mds_light_permutation(%22) : (tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %24 = stablehlo.slice %arg1 [3:4, 0:16] : (tensor<4x16x!pf_babybear_mont_std>) -> tensor<1x16x!pf_babybear_mont_std>
    %25 = stablehlo.reshape %24 : (tensor<1x16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %26 = stablehlo.add %23, %25 : tensor<16x!pf_babybear_mont_std>
    %27 = stablehlo.multiply %26, %26 : tensor<16x!pf_babybear_mont_std>
    %28 = stablehlo.multiply %26, %27 : tensor<16x!pf_babybear_mont_std>
    %29 = stablehlo.multiply %27, %27 : tensor<16x!pf_babybear_mont_std>
    %30 = stablehlo.multiply %28, %29 : tensor<16x!pf_babybear_mont_std>
    %31 = call @mds_light_permutation(%30) : (tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %31 : tensor<16x!pf_babybear_mont_std>
  }
  func.func private @permute_state(%arg0: tensor<16x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std> {
    %cst = stablehlo.constant() <{value = dense<[1518359488, 1765533241, 945325693, 422793067, 311365592, 1311448267, 1629555936, 1009879353, 190525218, 786108885, 557776863, 212616710, 605745517]> : tensor<13xui32>}> : () -> tensor<13x!pf_babybear_mont_std>
    %0 = stablehlo.slice %cst [0:1] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %1 = stablehlo.reshape %0 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %2 = stablehlo.slice %arg0 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %3 = stablehlo.reshape %2 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %4 = call @add_rc_and_sbox(%3, %1) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %5 = stablehlo.slice %arg0 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_0 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %6 = stablehlo.reduce(%5 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %7 = stablehlo.add %6, %4 : tensor<!pf_babybear_mont_std>
    %8 = stablehlo.subtract %6, %4 : tensor<!pf_babybear_mont_std>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %10 = "stablehlo.scatter"(%arg0, %9, %8) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %11 = call @internal_layer_mat_mul(%10, %7) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %12 = stablehlo.slice %cst [1:2] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %13 = stablehlo.reshape %12 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %14 = stablehlo.slice %11 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %15 = stablehlo.reshape %14 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %16 = call @add_rc_and_sbox(%15, %13) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %17 = stablehlo.slice %11 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_1 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %18 = stablehlo.reduce(%17 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %19 = stablehlo.add %18, %16 : tensor<!pf_babybear_mont_std>
    %20 = stablehlo.subtract %18, %16 : tensor<!pf_babybear_mont_std>
    %c_2 = stablehlo.constant dense<0> : tensor<i32>
    %21 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %22 = "stablehlo.scatter"(%11, %21, %20) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %23 = call @internal_layer_mat_mul(%22, %19) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %24 = stablehlo.slice %cst [2:3] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %25 = stablehlo.reshape %24 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %26 = stablehlo.slice %23 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %27 = stablehlo.reshape %26 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %28 = call @add_rc_and_sbox(%27, %25) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %29 = stablehlo.slice %23 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_3 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %30 = stablehlo.reduce(%29 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %31 = stablehlo.add %30, %28 : tensor<!pf_babybear_mont_std>
    %32 = stablehlo.subtract %30, %28 : tensor<!pf_babybear_mont_std>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %33 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %34 = "stablehlo.scatter"(%23, %33, %32) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %35 = call @internal_layer_mat_mul(%34, %31) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %36 = stablehlo.slice %cst [3:4] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %37 = stablehlo.reshape %36 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %38 = stablehlo.slice %35 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %39 = stablehlo.reshape %38 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %40 = call @add_rc_and_sbox(%39, %37) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %41 = stablehlo.slice %35 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_5 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %42 = stablehlo.reduce(%41 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %43 = stablehlo.add %42, %40 : tensor<!pf_babybear_mont_std>
    %44 = stablehlo.subtract %42, %40 : tensor<!pf_babybear_mont_std>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %45 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %46 = "stablehlo.scatter"(%35, %45, %44) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %47 = call @internal_layer_mat_mul(%46, %43) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %48 = stablehlo.slice %cst [4:5] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %49 = stablehlo.reshape %48 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %50 = stablehlo.slice %47 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %51 = stablehlo.reshape %50 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %52 = call @add_rc_and_sbox(%51, %49) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %53 = stablehlo.slice %47 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_7 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %54 = stablehlo.reduce(%53 init: %cst_7) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %55 = stablehlo.add %54, %52 : tensor<!pf_babybear_mont_std>
    %56 = stablehlo.subtract %54, %52 : tensor<!pf_babybear_mont_std>
    %c_8 = stablehlo.constant dense<0> : tensor<i32>
    %57 = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %58 = "stablehlo.scatter"(%47, %57, %56) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %59 = call @internal_layer_mat_mul(%58, %55) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %60 = stablehlo.slice %cst [5:6] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %61 = stablehlo.reshape %60 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %62 = stablehlo.slice %59 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %63 = stablehlo.reshape %62 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %64 = call @add_rc_and_sbox(%63, %61) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %65 = stablehlo.slice %59 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_9 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %66 = stablehlo.reduce(%65 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %67 = stablehlo.add %66, %64 : tensor<!pf_babybear_mont_std>
    %68 = stablehlo.subtract %66, %64 : tensor<!pf_babybear_mont_std>
    %c_10 = stablehlo.constant dense<0> : tensor<i32>
    %69 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %70 = "stablehlo.scatter"(%59, %69, %68) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %71 = call @internal_layer_mat_mul(%70, %67) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %72 = stablehlo.slice %cst [6:7] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %73 = stablehlo.reshape %72 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %74 = stablehlo.slice %71 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %75 = stablehlo.reshape %74 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %76 = call @add_rc_and_sbox(%75, %73) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %77 = stablehlo.slice %71 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_11 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %78 = stablehlo.reduce(%77 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %79 = stablehlo.add %78, %76 : tensor<!pf_babybear_mont_std>
    %80 = stablehlo.subtract %78, %76 : tensor<!pf_babybear_mont_std>
    %c_12 = stablehlo.constant dense<0> : tensor<i32>
    %81 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %82 = "stablehlo.scatter"(%71, %81, %80) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %83 = call @internal_layer_mat_mul(%82, %79) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %84 = stablehlo.slice %cst [7:8] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %85 = stablehlo.reshape %84 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %86 = stablehlo.slice %83 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %87 = stablehlo.reshape %86 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %88 = call @add_rc_and_sbox(%87, %85) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %89 = stablehlo.slice %83 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_13 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %90 = stablehlo.reduce(%89 init: %cst_13) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %91 = stablehlo.add %90, %88 : tensor<!pf_babybear_mont_std>
    %92 = stablehlo.subtract %90, %88 : tensor<!pf_babybear_mont_std>
    %c_14 = stablehlo.constant dense<0> : tensor<i32>
    %93 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %94 = "stablehlo.scatter"(%83, %93, %92) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %95 = call @internal_layer_mat_mul(%94, %91) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %96 = stablehlo.slice %cst [8:9] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %97 = stablehlo.reshape %96 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %98 = stablehlo.slice %95 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %99 = stablehlo.reshape %98 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %100 = call @add_rc_and_sbox(%99, %97) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %101 = stablehlo.slice %95 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_15 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %102 = stablehlo.reduce(%101 init: %cst_15) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %103 = stablehlo.add %102, %100 : tensor<!pf_babybear_mont_std>
    %104 = stablehlo.subtract %102, %100 : tensor<!pf_babybear_mont_std>
    %c_16 = stablehlo.constant dense<0> : tensor<i32>
    %105 = stablehlo.broadcast_in_dim %c_16, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %106 = "stablehlo.scatter"(%95, %105, %104) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %107 = call @internal_layer_mat_mul(%106, %103) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %108 = stablehlo.slice %cst [9:10] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %109 = stablehlo.reshape %108 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %110 = stablehlo.slice %107 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %111 = stablehlo.reshape %110 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %112 = call @add_rc_and_sbox(%111, %109) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %113 = stablehlo.slice %107 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_17 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %114 = stablehlo.reduce(%113 init: %cst_17) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %115 = stablehlo.add %114, %112 : tensor<!pf_babybear_mont_std>
    %116 = stablehlo.subtract %114, %112 : tensor<!pf_babybear_mont_std>
    %c_18 = stablehlo.constant dense<0> : tensor<i32>
    %117 = stablehlo.broadcast_in_dim %c_18, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %118 = "stablehlo.scatter"(%107, %117, %116) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %119 = call @internal_layer_mat_mul(%118, %115) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %120 = stablehlo.slice %cst [10:11] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %121 = stablehlo.reshape %120 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %122 = stablehlo.slice %119 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %123 = stablehlo.reshape %122 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %124 = call @add_rc_and_sbox(%123, %121) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %125 = stablehlo.slice %119 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_19 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %126 = stablehlo.reduce(%125 init: %cst_19) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %127 = stablehlo.add %126, %124 : tensor<!pf_babybear_mont_std>
    %128 = stablehlo.subtract %126, %124 : tensor<!pf_babybear_mont_std>
    %c_20 = stablehlo.constant dense<0> : tensor<i32>
    %129 = stablehlo.broadcast_in_dim %c_20, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %130 = "stablehlo.scatter"(%119, %129, %128) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %131 = call @internal_layer_mat_mul(%130, %127) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %132 = stablehlo.slice %cst [11:12] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %133 = stablehlo.reshape %132 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %134 = stablehlo.slice %131 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %135 = stablehlo.reshape %134 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %136 = call @add_rc_and_sbox(%135, %133) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %137 = stablehlo.slice %131 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_21 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %138 = stablehlo.reduce(%137 init: %cst_21) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %139 = stablehlo.add %138, %136 : tensor<!pf_babybear_mont_std>
    %140 = stablehlo.subtract %138, %136 : tensor<!pf_babybear_mont_std>
    %c_22 = stablehlo.constant dense<0> : tensor<i32>
    %141 = stablehlo.broadcast_in_dim %c_22, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %142 = "stablehlo.scatter"(%131, %141, %140) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %143 = call @internal_layer_mat_mul(%142, %139) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %144 = stablehlo.slice %cst [12:13] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %145 = stablehlo.reshape %144 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %146 = stablehlo.slice %143 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %147 = stablehlo.reshape %146 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %148 = call @add_rc_and_sbox(%147, %145) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %149 = stablehlo.slice %143 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_23 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %150 = stablehlo.reduce(%149 init: %cst_23) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %151 = stablehlo.add %150, %148 : tensor<!pf_babybear_mont_std>
    %152 = stablehlo.subtract %150, %148 : tensor<!pf_babybear_mont_std>
    %c_24 = stablehlo.constant dense<0> : tensor<i32>
    %153 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %154 = "stablehlo.scatter"(%143, %153, %152) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg1: tensor<!pf_babybear_mont_std>, %arg2: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg2 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %155 = call @internal_layer_mat_mul(%154, %151) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %155 : tensor<16x!pf_babybear_mont_std>
  }
  func.func private @add_rc_and_sbox(%arg0: tensor<!pf_babybear_mont_std>, %arg1: tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<!pf_babybear_mont_std>
    %1 = stablehlo.multiply %0, %0 : tensor<!pf_babybear_mont_std>
    %2 = stablehlo.multiply %0, %1 : tensor<!pf_babybear_mont_std>
    %3 = stablehlo.multiply %1, %1 : tensor<!pf_babybear_mont_std>
    %4 = stablehlo.multiply %2, %3 : tensor<!pf_babybear_mont_std>
    return %4 : tensor<!pf_babybear_mont_std>
  }
  func.func private @internal_layer_mat_mul(%arg0: tensor<16x!pf_babybear_mont_std>, %arg1: tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std> {
    %cst = stablehlo.constant() <{value = dense<[1, 2, 1006632961, 3, 4, 1006632960, 2013265918, 2013265917, 2005401601, 1509949441, 1761607681, 2013265906, 7864320, 125829120, 15]> : tensor<15xui32>}> : () -> tensor<15x!pf_babybear_mont_std>
    %0 = stablehlo.slice %arg0 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %1 = stablehlo.multiply %0, %cst : tensor<15x!pf_babybear_mont_std>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %3 = stablehlo.add %1, %2 : tensor<15x!pf_babybear_mont_std>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.scatter"(%arg0, %4, %3) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg2: tensor<!pf_babybear_mont_std>, %arg3: tensor<!pf_babybear_mont_std>):
      stablehlo.return %arg3 : tensor<!pf_babybear_mont_std>
    }) : (tensor<16x!pf_babybear_mont_std>, tensor<1xi32>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %5 : tensor<16x!pf_babybear_mont_std>
  }
}
