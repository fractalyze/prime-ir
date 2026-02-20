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
    %0 = stablehlo.reshape %arg0 : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %1 = stablehlo.slice %0 [0:4, 0:1] : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %2 = stablehlo.reshape %1 : (tensor<4x1x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %3 = stablehlo.slice %0 [0:4, 1:2] : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %4 = stablehlo.reshape %3 : (tensor<4x1x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %5 = stablehlo.slice %0 [0:4, 2:3] : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %6 = stablehlo.reshape %5 : (tensor<4x1x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %7 = stablehlo.slice %0 [0:4, 3:4] : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %8 = stablehlo.reshape %7 : (tensor<4x1x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %9 = stablehlo.add %2, %4 : tensor<4x!pf_babybear_mont_std>
    %10 = stablehlo.add %6, %8 : tensor<4x!pf_babybear_mont_std>
    %11 = stablehlo.add %9, %10 : tensor<4x!pf_babybear_mont_std>
    %12 = stablehlo.add %11, %4 : tensor<4x!pf_babybear_mont_std>
    %13 = stablehlo.add %11, %8 : tensor<4x!pf_babybear_mont_std>
    %cst = stablehlo.constant() <{value = dense<2> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %14 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %15 = stablehlo.multiply %2, %14 : tensor<4x!pf_babybear_mont_std>
    %cst_0 = stablehlo.constant() <{value = dense<2> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %16 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %17 = stablehlo.multiply %6, %16 : tensor<4x!pf_babybear_mont_std>
    %18 = stablehlo.add %12, %9 : tensor<4x!pf_babybear_mont_std>
    %19 = stablehlo.add %12, %17 : tensor<4x!pf_babybear_mont_std>
    %20 = stablehlo.add %13, %10 : tensor<4x!pf_babybear_mont_std>
    %21 = stablehlo.add %13, %15 : tensor<4x!pf_babybear_mont_std>
    %22 = stablehlo.broadcast_in_dim %18, dims = [0] : (tensor<4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %23 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %24 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %25 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %26 = stablehlo.concatenate %22, %23, %24, %25, dim = 1 : (tensor<4x1x!pf_babybear_mont_std>, tensor<4x1x!pf_babybear_mont_std>, tensor<4x1x!pf_babybear_mont_std>, tensor<4x1x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %27 = stablehlo.reshape %26 : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %28 = stablehlo.slice %27 [0:4] : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %29 = stablehlo.slice %27 [4:8] : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %30 = stablehlo.add %28, %29 : tensor<4x!pf_babybear_mont_std>
    %31 = stablehlo.slice %27 [8:12] : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %32 = stablehlo.add %30, %31 : tensor<4x!pf_babybear_mont_std>
    %33 = stablehlo.slice %27 [12:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x!pf_babybear_mont_std>
    %34 = stablehlo.add %32, %33 : tensor<4x!pf_babybear_mont_std>
    %35 = stablehlo.reshape %27 : (tensor<16x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %36 = stablehlo.transpose %35, dims = [1, 0] : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %37 = stablehlo.broadcast_in_dim %34, dims = [0] : (tensor<4x!pf_babybear_mont_std>) -> tensor<4x1x!pf_babybear_mont_std>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<4x1x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %39 = stablehlo.add %36, %38 : tensor<4x4x!pf_babybear_mont_std>
    %40 = stablehlo.transpose %39, dims = [1, 0] : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<4x4x!pf_babybear_mont_std>
    %41 = stablehlo.reshape %40 : (tensor<4x4x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %41 : tensor<16x!pf_babybear_mont_std>
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
    %9 = stablehlo.reshape %8 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %10 = stablehlo.slice %arg0 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %12 = call @internal_layer_mat_mul(%11, %7) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %13 = stablehlo.slice %cst [1:2] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %14 = stablehlo.reshape %13 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %15 = stablehlo.slice %12 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %16 = stablehlo.reshape %15 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %17 = call @add_rc_and_sbox(%16, %14) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %18 = stablehlo.slice %12 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_1 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %19 = stablehlo.reduce(%18 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %20 = stablehlo.add %19, %17 : tensor<!pf_babybear_mont_std>
    %21 = stablehlo.subtract %19, %17 : tensor<!pf_babybear_mont_std>
    %22 = stablehlo.reshape %21 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %23 = stablehlo.slice %12 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %24 = stablehlo.concatenate %22, %23, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %25 = call @internal_layer_mat_mul(%24, %20) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %26 = stablehlo.slice %cst [2:3] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %27 = stablehlo.reshape %26 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %28 = stablehlo.slice %25 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %29 = stablehlo.reshape %28 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %30 = call @add_rc_and_sbox(%29, %27) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %31 = stablehlo.slice %25 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_2 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %32 = stablehlo.reduce(%31 init: %cst_2) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %33 = stablehlo.add %32, %30 : tensor<!pf_babybear_mont_std>
    %34 = stablehlo.subtract %32, %30 : tensor<!pf_babybear_mont_std>
    %35 = stablehlo.reshape %34 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %36 = stablehlo.slice %25 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %37 = stablehlo.concatenate %35, %36, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %38 = call @internal_layer_mat_mul(%37, %33) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %39 = stablehlo.slice %cst [3:4] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %40 = stablehlo.reshape %39 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %41 = stablehlo.slice %38 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %42 = stablehlo.reshape %41 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %43 = call @add_rc_and_sbox(%42, %40) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %44 = stablehlo.slice %38 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_3 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %45 = stablehlo.reduce(%44 init: %cst_3) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %46 = stablehlo.add %45, %43 : tensor<!pf_babybear_mont_std>
    %47 = stablehlo.subtract %45, %43 : tensor<!pf_babybear_mont_std>
    %48 = stablehlo.reshape %47 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %49 = stablehlo.slice %38 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %50 = stablehlo.concatenate %48, %49, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %51 = call @internal_layer_mat_mul(%50, %46) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %52 = stablehlo.slice %cst [4:5] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %53 = stablehlo.reshape %52 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %54 = stablehlo.slice %51 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %55 = stablehlo.reshape %54 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %56 = call @add_rc_and_sbox(%55, %53) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %57 = stablehlo.slice %51 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_4 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %58 = stablehlo.reduce(%57 init: %cst_4) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %59 = stablehlo.add %58, %56 : tensor<!pf_babybear_mont_std>
    %60 = stablehlo.subtract %58, %56 : tensor<!pf_babybear_mont_std>
    %61 = stablehlo.reshape %60 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %62 = stablehlo.slice %51 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %63 = stablehlo.concatenate %61, %62, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %64 = call @internal_layer_mat_mul(%63, %59) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %65 = stablehlo.slice %cst [5:6] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %66 = stablehlo.reshape %65 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %67 = stablehlo.slice %64 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %68 = stablehlo.reshape %67 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %69 = call @add_rc_and_sbox(%68, %66) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %70 = stablehlo.slice %64 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_5 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %71 = stablehlo.reduce(%70 init: %cst_5) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %72 = stablehlo.add %71, %69 : tensor<!pf_babybear_mont_std>
    %73 = stablehlo.subtract %71, %69 : tensor<!pf_babybear_mont_std>
    %74 = stablehlo.reshape %73 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %75 = stablehlo.slice %64 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %76 = stablehlo.concatenate %74, %75, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %77 = call @internal_layer_mat_mul(%76, %72) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %78 = stablehlo.slice %cst [6:7] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %79 = stablehlo.reshape %78 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %80 = stablehlo.slice %77 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %81 = stablehlo.reshape %80 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %82 = call @add_rc_and_sbox(%81, %79) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %83 = stablehlo.slice %77 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_6 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %84 = stablehlo.reduce(%83 init: %cst_6) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %85 = stablehlo.add %84, %82 : tensor<!pf_babybear_mont_std>
    %86 = stablehlo.subtract %84, %82 : tensor<!pf_babybear_mont_std>
    %87 = stablehlo.reshape %86 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %88 = stablehlo.slice %77 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %89 = stablehlo.concatenate %87, %88, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %90 = call @internal_layer_mat_mul(%89, %85) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %91 = stablehlo.slice %cst [7:8] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %92 = stablehlo.reshape %91 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %93 = stablehlo.slice %90 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %94 = stablehlo.reshape %93 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %95 = call @add_rc_and_sbox(%94, %92) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %96 = stablehlo.slice %90 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_7 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %97 = stablehlo.reduce(%96 init: %cst_7) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %98 = stablehlo.add %97, %95 : tensor<!pf_babybear_mont_std>
    %99 = stablehlo.subtract %97, %95 : tensor<!pf_babybear_mont_std>
    %100 = stablehlo.reshape %99 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %101 = stablehlo.slice %90 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %102 = stablehlo.concatenate %100, %101, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %103 = call @internal_layer_mat_mul(%102, %98) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %104 = stablehlo.slice %cst [8:9] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %105 = stablehlo.reshape %104 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %106 = stablehlo.slice %103 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %107 = stablehlo.reshape %106 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %108 = call @add_rc_and_sbox(%107, %105) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %109 = stablehlo.slice %103 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_8 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %110 = stablehlo.reduce(%109 init: %cst_8) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %111 = stablehlo.add %110, %108 : tensor<!pf_babybear_mont_std>
    %112 = stablehlo.subtract %110, %108 : tensor<!pf_babybear_mont_std>
    %113 = stablehlo.reshape %112 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %114 = stablehlo.slice %103 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %115 = stablehlo.concatenate %113, %114, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %116 = call @internal_layer_mat_mul(%115, %111) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %117 = stablehlo.slice %cst [9:10] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %118 = stablehlo.reshape %117 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %119 = stablehlo.slice %116 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %120 = stablehlo.reshape %119 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %121 = call @add_rc_and_sbox(%120, %118) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %122 = stablehlo.slice %116 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_9 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %123 = stablehlo.reduce(%122 init: %cst_9) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %124 = stablehlo.add %123, %121 : tensor<!pf_babybear_mont_std>
    %125 = stablehlo.subtract %123, %121 : tensor<!pf_babybear_mont_std>
    %126 = stablehlo.reshape %125 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %127 = stablehlo.slice %116 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %128 = stablehlo.concatenate %126, %127, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %129 = call @internal_layer_mat_mul(%128, %124) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %130 = stablehlo.slice %cst [10:11] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %131 = stablehlo.reshape %130 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %132 = stablehlo.slice %129 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %133 = stablehlo.reshape %132 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %134 = call @add_rc_and_sbox(%133, %131) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %135 = stablehlo.slice %129 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_10 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %136 = stablehlo.reduce(%135 init: %cst_10) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %137 = stablehlo.add %136, %134 : tensor<!pf_babybear_mont_std>
    %138 = stablehlo.subtract %136, %134 : tensor<!pf_babybear_mont_std>
    %139 = stablehlo.reshape %138 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %140 = stablehlo.slice %129 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %141 = stablehlo.concatenate %139, %140, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %142 = call @internal_layer_mat_mul(%141, %137) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %143 = stablehlo.slice %cst [11:12] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %144 = stablehlo.reshape %143 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %145 = stablehlo.slice %142 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %146 = stablehlo.reshape %145 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %147 = call @add_rc_and_sbox(%146, %144) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %148 = stablehlo.slice %142 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_11 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %149 = stablehlo.reduce(%148 init: %cst_11) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %150 = stablehlo.add %149, %147 : tensor<!pf_babybear_mont_std>
    %151 = stablehlo.subtract %149, %147 : tensor<!pf_babybear_mont_std>
    %152 = stablehlo.reshape %151 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %153 = stablehlo.slice %142 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %154 = stablehlo.concatenate %152, %153, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %155 = call @internal_layer_mat_mul(%154, %150) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %156 = stablehlo.slice %cst [12:13] : (tensor<13x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %157 = stablehlo.reshape %156 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %158 = stablehlo.slice %155 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %159 = stablehlo.reshape %158 : (tensor<1x!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %160 = call @add_rc_and_sbox(%159, %157) : (tensor<!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %161 = stablehlo.slice %155 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %cst_12 = stablehlo.constant() <{value = dense<0> : tensor<ui32>}> : () -> tensor<!pf_babybear_mont_std>
    %162 = stablehlo.reduce(%161 init: %cst_12) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<!pf_babybear_mont_std>
    %163 = stablehlo.add %162, %160 : tensor<!pf_babybear_mont_std>
    %164 = stablehlo.subtract %162, %160 : tensor<!pf_babybear_mont_std>
    %165 = stablehlo.reshape %164 : (tensor<!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %166 = stablehlo.slice %155 [1:16] : (tensor<16x!pf_babybear_mont_std>) -> tensor<15x!pf_babybear_mont_std>
    %167 = stablehlo.concatenate %165, %166, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    %168 = call @internal_layer_mat_mul(%167, %163) : (tensor<16x!pf_babybear_mont_std>, tensor<!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %168 : tensor<16x!pf_babybear_mont_std>
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
    %4 = stablehlo.slice %arg0 [0:1] : (tensor<16x!pf_babybear_mont_std>) -> tensor<1x!pf_babybear_mont_std>
    %5 = stablehlo.concatenate %4, %3, dim = 0 : (tensor<1x!pf_babybear_mont_std>, tensor<15x!pf_babybear_mont_std>) -> tensor<16x!pf_babybear_mont_std>
    return %5 : tensor<16x!pf_babybear_mont_std>
  }
}
