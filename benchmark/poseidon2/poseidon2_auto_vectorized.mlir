!pf_babybear = !field.pf<2013265921 : i32, true>
!pf_babybear_std = !field.pf<2013265921 : i32>
module {
  func.func @auto_vec_add_rc_and_sbox(%arg0: vector<16x!pf_babybear>, %arg1: vector<16x!pf_babybear>) -> vector<16x!pf_babybear> {
    %c7_i32 = arith.constant 7 : i32
    %0 = field.add %arg0, %arg1 : vector<16x!pf_babybear>
    %1 = field.powui %0, %c7_i32 : vector<16x!pf_babybear>, i32
    return %1 : vector<16x!pf_babybear>
  }
  func.func @auto_vec_apply_mat4(%arg0: memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>) {
    %cst = arith.constant dense<[[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]> : tensor<4x4xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = memref.load %arg0[%c0] : memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>
    %1 = memref.load %arg0[%c1] : memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>
    %2 = memref.load %arg0[%c2] : memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>
    %3 = memref.load %arg0[%c3] : memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>
    %4 = field.add %0, %1 : vector<16x!pf_babybear>
    %5 = field.add %2, %3 : vector<16x!pf_babybear>
    %6 = field.add %4, %5 : vector<16x!pf_babybear>
    %7 = field.add %6, %1 : vector<16x!pf_babybear>
    %8 = field.add %6, %3 : vector<16x!pf_babybear>
    %9 = field.double %0 : vector<16x!pf_babybear>
    %10 = field.double %2 : vector<16x!pf_babybear>
    %11 = field.add %7, %4 : vector<16x!pf_babybear>
    %12 = field.add %7, %10 : vector<16x!pf_babybear>
    %13 = field.add %8, %5 : vector<16x!pf_babybear>
    %14 = field.add %8, %9 : vector<16x!pf_babybear>
    memref.store %11, %arg0[%c0] : memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>
    memref.store %12, %arg0[%c1] : memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>
    memref.store %13, %arg0[%c2] : memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>
    memref.store %14, %arg0[%c3] : memref<4xvector<16x!pf_babybear>, strided<[1], offset: ?>>
    return
  }
  func.func @auto_vec_mds_light_permutation(%arg0: memref<16xvector<16x!pf_babybear>>) {
    affine.for %arg1 = 0 to 4 {
      %0 = affine.load %arg0[%arg1 * 4] : memref<16xvector<16x!pf_babybear>>
      %1 = affine.load %arg0[%arg1 * 4 + 1] : memref<16xvector<16x!pf_babybear>>
      %2 = field.add %0, %1 : vector<16x!pf_babybear>
      %3 = affine.load %arg0[%arg1 * 4 + 2] : memref<16xvector<16x!pf_babybear>>
      %4 = affine.load %arg0[%arg1 * 4 + 3] : memref<16xvector<16x!pf_babybear>>
      %5 = field.add %3, %4 : vector<16x!pf_babybear>
      %6 = field.add %2, %5 : vector<16x!pf_babybear>
      %7 = field.add %6, %1 : vector<16x!pf_babybear>
      %8 = field.add %6, %4 : vector<16x!pf_babybear>
      %9 = field.double %0 : vector<16x!pf_babybear>
      %10 = field.double %3 : vector<16x!pf_babybear>
      %11 = field.add %7, %2 : vector<16x!pf_babybear>
      %12 = field.add %7, %10 : vector<16x!pf_babybear>
      %13 = field.add %8, %5 : vector<16x!pf_babybear>
      %14 = field.add %8, %9 : vector<16x!pf_babybear>
      affine.store %11, %arg0[%arg1 * 4] : memref<16xvector<16x!pf_babybear>>
      affine.store %12, %arg0[%arg1 * 4 + 1] : memref<16xvector<16x!pf_babybear>>
      affine.store %13, %arg0[%arg1 * 4 + 2] : memref<16xvector<16x!pf_babybear>>
      affine.store %14, %arg0[%arg1 * 4 + 3] : memref<16xvector<16x!pf_babybear>>
    }
    %alloca = memref.alloca() : memref<4xvector<16x!pf_babybear>>
    affine.for %arg1 = 0 to 4 {
      %0 = affine.load %arg0[%arg1] : memref<16xvector<16x!pf_babybear>>
      %1 = affine.load %arg0[%arg1 + 4] : memref<16xvector<16x!pf_babybear>>
      %2 = affine.load %arg0[%arg1 + 8] : memref<16xvector<16x!pf_babybear>>
      %3 = affine.load %arg0[%arg1 + 12] : memref<16xvector<16x!pf_babybear>>
      %4 = field.add %0, %1 : vector<16x!pf_babybear>
      %5 = field.add %2, %3 : vector<16x!pf_babybear>
      %6 = field.add %4, %5 : vector<16x!pf_babybear>
      affine.store %6, %alloca[%arg1] : memref<4xvector<16x!pf_babybear>>
    }
    affine.for %arg1 = 0 to 4 {
      %0 = affine.load %arg0[%arg1] : memref<16xvector<16x!pf_babybear>>
      %1 = affine.load %arg0[%arg1 + 4] : memref<16xvector<16x!pf_babybear>>
      %2 = affine.load %arg0[%arg1 + 8] : memref<16xvector<16x!pf_babybear>>
      %3 = affine.load %arg0[%arg1 + 12] : memref<16xvector<16x!pf_babybear>>
      %4 = affine.load %alloca[%arg1] : memref<4xvector<16x!pf_babybear>>
      %5 = field.add %0, %4 : vector<16x!pf_babybear>
      %6 = field.add %1, %4 : vector<16x!pf_babybear>
      %7 = field.add %2, %4 : vector<16x!pf_babybear>
      %8 = field.add %3, %4 : vector<16x!pf_babybear>
      affine.store %5, %arg0[%arg1] : memref<16xvector<16x!pf_babybear>>
      affine.store %6, %arg0[%arg1 + 4] : memref<16xvector<16x!pf_babybear>>
      affine.store %7, %arg0[%arg1 + 8] : memref<16xvector<16x!pf_babybear>>
      affine.store %8, %arg0[%arg1 + 12] : memref<16xvector<16x!pf_babybear>>
    }
    return
  }
  func.func @auto_vec_internal_layer_mat_mul(%arg0: memref<16xvector<16x!pf_babybear>>, %arg1: vector<16x!pf_babybear>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %cst = arith.constant dense<2> : vector<16xi32>
    %0 = field.bitcast %cst : vector<16xi32> -> vector<16x!pf_babybear_std>
    %1 = field.to_mont %0 : vector<16x!pf_babybear>
    %2 = field.inverse %1 : vector<16x!pf_babybear>
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c27_i32 = arith.constant 27 : i32
    %3 = field.powui %2, %c2_i32 : vector<16x!pf_babybear>, i32
    %4 = field.powui %2, %c3_i32 : vector<16x!pf_babybear>, i32
    %5 = field.powui %2, %c4_i32 : vector<16x!pf_babybear>, i32
    %6 = field.square %5 : vector<16x!pf_babybear>
    %7 = field.powui %2, %c27_i32 : vector<16x!pf_babybear>, i32
    %8 = memref.load %arg0[%c1] : memref<16xvector<16x!pf_babybear>>
    %9 = field.add %8, %arg1 : vector<16x!pf_babybear>
    memref.store %9, %arg0[%c1] : memref<16xvector<16x!pf_babybear>>
    %10 = memref.load %arg0[%c2] : memref<16xvector<16x!pf_babybear>>
    %11 = field.double %10 : vector<16x!pf_babybear>
    %12 = field.add %11, %arg1 : vector<16x!pf_babybear>
    memref.store %12, %arg0[%c2] : memref<16xvector<16x!pf_babybear>>
    %13 = memref.load %arg0[%c3] : memref<16xvector<16x!pf_babybear>>
    %14 = field.mul %13, %2 : vector<16x!pf_babybear>
    %15 = field.add %14, %arg1 : vector<16x!pf_babybear>
    memref.store %15, %arg0[%c3] : memref<16xvector<16x!pf_babybear>>
    %16 = memref.load %arg0[%c4] : memref<16xvector<16x!pf_babybear>>
    %17 = field.double %16 : vector<16x!pf_babybear>
    %18 = field.add %17, %16 : vector<16x!pf_babybear>
    %19 = field.add %arg1, %18 : vector<16x!pf_babybear>
    memref.store %19, %arg0[%c4] : memref<16xvector<16x!pf_babybear>>
    %20 = memref.load %arg0[%c5] : memref<16xvector<16x!pf_babybear>>
    %21 = field.double %20 : vector<16x!pf_babybear>
    %22 = field.double %21 : vector<16x!pf_babybear>
    %23 = field.add %arg1, %22 : vector<16x!pf_babybear>
    memref.store %23, %arg0[%c5] : memref<16xvector<16x!pf_babybear>>
    %24 = memref.load %arg0[%c6] : memref<16xvector<16x!pf_babybear>>
    %25 = field.mul %24, %2 : vector<16x!pf_babybear>
    %26 = field.sub %arg1, %25 : vector<16x!pf_babybear>
    memref.store %26, %arg0[%c6] : memref<16xvector<16x!pf_babybear>>
    %27 = memref.load %arg0[%c7] : memref<16xvector<16x!pf_babybear>>
    %28 = field.double %27 : vector<16x!pf_babybear>
    %29 = field.add %28, %27 : vector<16x!pf_babybear>
    %30 = field.sub %arg1, %29 : vector<16x!pf_babybear>
    memref.store %30, %arg0[%c7] : memref<16xvector<16x!pf_babybear>>
    %31 = memref.load %arg0[%c8] : memref<16xvector<16x!pf_babybear>>
    %32 = field.double %31 : vector<16x!pf_babybear>
    %33 = field.double %32 : vector<16x!pf_babybear>
    %34 = field.sub %arg1, %33 : vector<16x!pf_babybear>
    memref.store %34, %arg0[%c8] : memref<16xvector<16x!pf_babybear>>
    %35 = memref.load %arg0[%c9] : memref<16xvector<16x!pf_babybear>>
    %36 = field.mul %35, %6 : vector<16x!pf_babybear>
    %37 = field.add %36, %arg1 : vector<16x!pf_babybear>
    memref.store %37, %arg0[%c9] : memref<16xvector<16x!pf_babybear>>
    %38 = memref.load %arg0[%c10] : memref<16xvector<16x!pf_babybear>>
    %39 = field.mul %38, %3 : vector<16x!pf_babybear>
    %40 = field.add %39, %arg1 : vector<16x!pf_babybear>
    memref.store %40, %arg0[%c10] : memref<16xvector<16x!pf_babybear>>
    %41 = memref.load %arg0[%c11] : memref<16xvector<16x!pf_babybear>>
    %42 = field.mul %41, %4 : vector<16x!pf_babybear>
    %43 = field.add %42, %arg1 : vector<16x!pf_babybear>
    memref.store %43, %arg0[%c11] : memref<16xvector<16x!pf_babybear>>
    %44 = memref.load %arg0[%c12] : memref<16xvector<16x!pf_babybear>>
    %45 = field.mul %44, %7 : vector<16x!pf_babybear>
    %46 = field.add %45, %arg1 : vector<16x!pf_babybear>
    memref.store %46, %arg0[%c12] : memref<16xvector<16x!pf_babybear>>
    %47 = memref.load %arg0[%c13] : memref<16xvector<16x!pf_babybear>>
    %48 = field.mul %47, %6 : vector<16x!pf_babybear>
    %49 = field.sub %arg1, %48 : vector<16x!pf_babybear>
    memref.store %49, %arg0[%c13] : memref<16xvector<16x!pf_babybear>>
    %50 = memref.load %arg0[%c14] : memref<16xvector<16x!pf_babybear>>
    %51 = field.mul %50, %5 : vector<16x!pf_babybear>
    %52 = field.sub %arg1, %51 : vector<16x!pf_babybear>
    memref.store %52, %arg0[%c14] : memref<16xvector<16x!pf_babybear>>
    %53 = memref.load %arg0[%c15] : memref<16xvector<16x!pf_babybear>>
    %54 = field.mul %53, %7 : vector<16x!pf_babybear>
    %55 = field.sub %arg1, %54 : vector<16x!pf_babybear>
    memref.store %55, %arg0[%c15] : memref<16xvector<16x!pf_babybear>>
    return
  }
  func.func @auto_vec_permute_state(%arg0: memref<16xvector<16x!pf_babybear>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %cst = arith.constant dense<[1518359488, 1765533241, 945325693, 422793067, 311365592, 1311448267, 1629555936, 1009879353, 190525218, 786108885, 557776863, 212616710, 605745517]> : tensor<13xi32>
    %0 = field.bitcast %cst : tensor<13xi32> -> tensor<13x!pf_babybear_std>
    %1 = field.to_mont %0 : tensor<13x!pf_babybear>
    affine.for %arg1 = 0 to 13 {
      %extracted = tensor.extract %1[%arg1] : tensor<13x!pf_babybear>
      %2 = vector.splat %extracted : vector<16x!pf_babybear>
      %3 = memref.load %arg0[%c0] : memref<16xvector<16x!pf_babybear>>
      %4 = func.call @auto_vec_add_rc_and_sbox(%3, %2) : (vector<16x!pf_babybear>, vector<16x!pf_babybear>) -> vector<16x!pf_babybear>
      %5 = memref.load %arg0[%c1] : memref<16xvector<16x!pf_babybear>>
      %6 = memref.load %arg0[%c2] : memref<16xvector<16x!pf_babybear>>
      %7 = memref.load %arg0[%c3] : memref<16xvector<16x!pf_babybear>>
      %8 = memref.load %arg0[%c4] : memref<16xvector<16x!pf_babybear>>
      %9 = memref.load %arg0[%c5] : memref<16xvector<16x!pf_babybear>>
      %10 = memref.load %arg0[%c6] : memref<16xvector<16x!pf_babybear>>
      %11 = memref.load %arg0[%c7] : memref<16xvector<16x!pf_babybear>>
      %12 = memref.load %arg0[%c8] : memref<16xvector<16x!pf_babybear>>
      %13 = memref.load %arg0[%c9] : memref<16xvector<16x!pf_babybear>>
      %14 = memref.load %arg0[%c10] : memref<16xvector<16x!pf_babybear>>
      %15 = memref.load %arg0[%c11] : memref<16xvector<16x!pf_babybear>>
      %16 = memref.load %arg0[%c12] : memref<16xvector<16x!pf_babybear>>
      %17 = memref.load %arg0[%c13] : memref<16xvector<16x!pf_babybear>>
      %18 = memref.load %arg0[%c14] : memref<16xvector<16x!pf_babybear>>
      %19 = memref.load %arg0[%c15] : memref<16xvector<16x!pf_babybear>>
      %20 = field.add %6, %7 : vector<16x!pf_babybear>
      %21 = field.add %8, %9 : vector<16x!pf_babybear>
      %22 = field.add %10, %11 : vector<16x!pf_babybear>
      %23 = field.add %12, %13 : vector<16x!pf_babybear>
      %24 = field.add %14, %15 : vector<16x!pf_babybear>
      %25 = field.add %16, %17 : vector<16x!pf_babybear>
      %26 = field.add %18, %19 : vector<16x!pf_babybear>
      %27 = field.add %5, %20 : vector<16x!pf_babybear>
      %28 = field.add %21, %22 : vector<16x!pf_babybear>
      %29 = field.add %23, %24 : vector<16x!pf_babybear>
      %30 = field.add %25, %26 : vector<16x!pf_babybear>
      %31 = field.add %27, %28 : vector<16x!pf_babybear>
      %32 = field.add %29, %30 : vector<16x!pf_babybear>
      %33 = field.add %31, %32 : vector<16x!pf_babybear>
      %34 = field.add %33, %4 : vector<16x!pf_babybear>
      %35 = field.sub %33, %4 : vector<16x!pf_babybear>
      memref.store %35, %arg0[%c0] : memref<16xvector<16x!pf_babybear>>
      func.call @auto_vec_internal_layer_mat_mul(%arg0, %34) : (memref<16xvector<16x!pf_babybear>>, vector<16x!pf_babybear>) -> ()
    }
    return
  }
  func.func @auto_vec_permute_state_terminal(%arg0: memref<16xvector<16x!pf_babybear>>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<[[1922082829, 1870549801, 1502529704, 1990744480, 1700391016, 1702593455, 321330495, 528965731, 183414327, 1886297254, 1178602734, 1923111974, 744004766, 549271463, 1781349648, 542259047], [1536158148, 715456982, 503426110, 340311124, 1558555932, 1226350925, 742828095, 1338992758, 1641600456, 1843351545, 301835475, 43203215, 386838401, 1520185679, 1235297680, 904680097], [1491801617, 1581784677, 913384905, 247083962, 532844013, 107190701, 213827818, 1979521776, 1358282574, 1681743681, 1867507480, 1530706910, 507181886, 695185447, 1172395131, 1250800299], [1503161625, 817684387, 498481458, 494676004, 1404253825, 108246855, 59414691, 744214112, 890862029, 1342765939, 1417398904, 1897591937, 1066647396, 1682806907, 1015795079, 1619482808]]> : tensor<4x16xi32>
    %0 = field.bitcast %cst : tensor<4x16xi32> -> tensor<4x16x!pf_babybear_std>
    %1 = field.to_mont %0 : tensor<4x16x!pf_babybear>
    %2 = bufferization.to_tensor %arg0 restrict : memref<16xvector<16x!pf_babybear>> to tensor<16xvector<16x!pf_babybear>>
    affine.for %arg1 = 0 to 4 {
      affine.for %arg2 = 0 to 16 {
        %3 = memref.load %arg0[%arg2] : memref<16xvector<16x!pf_babybear>>
        %extracted = tensor.extract %1[%arg1, %arg2] : tensor<4x16x!pf_babybear>
        %4 = vector.splat %extracted : vector<16x!pf_babybear>
        %5 = func.call @auto_vec_add_rc_and_sbox(%3, %4) : (vector<16x!pf_babybear>, vector<16x!pf_babybear>) -> vector<16x!pf_babybear>
        affine.store %5, %arg0[%arg2] : memref<16xvector<16x!pf_babybear>>
      }
      func.call @auto_vec_mds_light_permutation(%arg0) : (memref<16xvector<16x!pf_babybear>>) -> ()
    }
    return
  }
  func.func @auto_vec_permute_state_initial(%arg0: memref<16xvector<16x!pf_babybear>>) {
    call @auto_vec_mds_light_permutation(%arg0) : (memref<16xvector<16x!pf_babybear>>) -> ()
    %cst = arith.constant dense<[[1774958255, 1185780729, 1621102414, 1796380621, 588815102, 1932426223, 1925334750, 747903232, 89648862, 360728943, 977184635, 1425273457, 256487465, 1200041953, 572403254, 448208942], [1215789478, 944884184, 953948096, 547326025, 646827752, 889997530, 1536873262, 86189867, 1065944411, 32019634, 333311454, 456061748, 1963448500, 1827584334, 1391160226, 1348741381], [88424255, 104111868, 1763866748, 79691676, 1988915530, 1050669594, 359890076, 573163527, 222820492, 159256268, 669703072, 763177444, 889367200, 256335831, 704371273, 25886717], [51754520, 1833211857, 454499742, 1384520381, 777848065, 1053320300, 1851729162, 344647910, 401996362, 1046925956, 5351995, 1212119315, 754867989, 36972490, 751272725, 506915399]]> : tensor<4x16xi32>
    %0 = field.bitcast %cst : tensor<4x16xi32> -> tensor<4x16x!pf_babybear_std>
    %1 = field.to_mont %0 : tensor<4x16x!pf_babybear>
    %2 = bufferization.to_tensor %arg0 restrict : memref<16xvector<16x!pf_babybear>> to tensor<16xvector<16x!pf_babybear>>
    affine.for %arg1 = 0 to 4 {
      affine.for %arg2 = 0 to 16 {
        %3 = memref.load %arg0[%arg2] : memref<16xvector<16x!pf_babybear>>
        %extracted = tensor.extract %1[%arg1, %arg2] : tensor<4x16x!pf_babybear>
        %4 = vector.splat %extracted : vector<16x!pf_babybear>
        %5 = func.call @auto_vec_add_rc_and_sbox(%3, %4) : (vector<16x!pf_babybear>, vector<16x!pf_babybear>) -> vector<16x!pf_babybear>
        affine.store %5, %arg0[%arg2] : memref<16xvector<16x!pf_babybear>>
      }
      func.call @auto_vec_mds_light_permutation(%arg0) : (memref<16xvector<16x!pf_babybear>>) -> ()
    }
    return
  }
  func.func @auto_vec_poseidon2_permute(%arg0: memref<16xvector<16x!pf_babybear>>) {
    call @auto_vec_permute_state_initial(%arg0) : (memref<16xvector<16x!pf_babybear>>) -> ()
    call @auto_vec_permute_state(%arg0) : (memref<16xvector<16x!pf_babybear>>) -> ()
    call @auto_vec_permute_state_terminal(%arg0) : (memref<16xvector<16x!pf_babybear>>) -> ()
    return
  }
  func.func @auto_vec_permute_10000(%arg0: memref<16xvector<16x!pf_babybear>>) attributes {llvm.emit_c_interface} {
    affine.for %arg1 = 0 to 10000 {
      func.call @auto_vec_poseidon2_permute(%arg0) : (memref<16xvector<16x!pf_babybear>>) -> ()
    }
    return
  }
}

