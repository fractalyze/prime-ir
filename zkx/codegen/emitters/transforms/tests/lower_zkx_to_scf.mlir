// RUN: emitters_opt %s -zkx-lower-zkx-to-scf --split-input-file  \
// RUN: | FileCheck %s

func.func @combiner(%a: i32, %b: i32, %c: i32, %d: i32) -> (i32, i32) {
  return %a, %b : i32, i32
}

func.func @shuffler(%a: i32, %b: i32) -> (i32, i32) {
  %ret:2 = zkx_gpu.shuffle_reduce (%a, %b) to 4 combiner=@combiner: i32, i32
  return %ret#0, %ret#1 : i32, i32
}
// CHECK: @shuffler(%[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[C2:.*]] = arith.constant 2
// CHECK-DAG: %[[C4:.*]] = arith.constant 4
// CHECK-DAG: %[[C32:.*]] = arith.constant 32
// CHECK: %[[A4H:.*]], {{.*}} = gpu.shuffle down %[[A]], %[[C4]], %[[C32]]
// CHECK: %[[B4H:.*]], {{.*}} = gpu.shuffle down %[[B]], %[[C4]], %[[C32]]
// CHECK: %[[AB4_0:.*]], %[[AB4_1:.*]] = zkx.pure_call @combiner(%[[A]], %[[B]], %[[A4H]], %[[B4H]])
// CHECK: %[[A2H:.*]], {{.*}} = gpu.shuffle down %[[AB4_0]], %[[C2]], %[[C32]]
// CHECK: %[[B2H:.*]], {{.*}} = gpu.shuffle down %[[AB4_1]], %[[C2]], %[[C32]]
// CHECK: %[[AB2_0:.*]], %[[AB2_1:.*]] = zkx.pure_call @combiner(%[[AB4_0]], %[[AB4_1]], %[[A2H]], %[[B2H]])
// CHECK: %[[A1H:.*]], {{.*}} = gpu.shuffle down %[[AB2_0]], %[[C1]], %[[C32]]
// CHECK: %[[B1H:.*]], {{.*}} = gpu.shuffle down %[[AB2_1]], %[[C1]], %[[C32]]
// CHECK: %[[AB1_0:.*]], %[[AB1_1:.*]] = zkx.pure_call @combiner(%[[AB2_0]], %[[AB2_1]], %[[A1H]], %[[B1H]])
// CHECK: return %[[AB1_0]], %[[AB1_1]]

// -----

func.func @combiner(%a: i64, %b: i64) -> i64 {
  return %a : i64
}

func.func @shuffler(%a: i64) -> i64 {
  %ret = zkx_gpu.shuffle_reduce(%a) to 1 combiner=@combiner : i64
  return %ret : i64
}
// CHECK: @shuffler(%[[A:.*]]: i64
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-COUNT-2: gpu.shuffle down {{.*}}, %[[C1]], %[[C32]]

// -----

func.func @combiner(%a: i8, %b: i8) -> i8 {
  return %a : i8
}

func.func @shuffler_i8(%a: i8) -> i8 {
  %ret = zkx_gpu.shuffle_reduce (%a) to 1 combiner=@combiner : i8
  return %ret : i8
}
// CHECK: @shuffler_i8(
// CHECK-NOT: vector
// CHECK-COUNT-1: gpu.shuffle down {{.*}}, %[[C1]]

// -----

func.func @predicated_insert(
    %v: i32, %tensor: tensor<2xi32>, %index: index,
    %cond: i1) -> tensor<2xi32> {
  %ret = zkx.predicated_insert %v into %tensor[%index] if %cond
    : tensor<2xi32>
  return %ret : tensor<2xi32>
}
// CHECK: @predicated_insert
// CHECK-SAME: %[[V:.*]]: i32, %[[TENSOR:.*]]: tensor<2xi32>,
// CHECK-SAME: %[[INDEX:.*]]: index, %[[COND:.*]]: i1
// CHECK-NEXT: %[[RET:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   %[[UPD:.*]] = tensor.insert %[[V]] into %[[TENSOR]][%[[INDEX]]]
// CHECK-NEXT:   yield %[[UPD]]
// CHECK-NEXT: else
// CHECK-NEXT:   yield %[[TENSOR]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RET]]

// -----

func.func @predicated_extract(
    %v: i32, %tensor: tensor<2xi32>, %index: index,
    %cond: i1) -> i32 {
  %ret = zkx.predicated_extract %tensor[%index] if %cond else %v
    : tensor<2xi32>
  return %ret : i32
}
// CHECK: @predicated_extract
// CHECK-SAME: %[[V:.*]]: i32, %[[TENSOR:.*]]: tensor<2xi32>,
// CHECK-SAME: %[[INDEX:.*]]: index, %[[COND:.*]]: i1
// CHECK-NEXT: %[[RET:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   %[[VAL:.*]] = tensor.extract  %[[TENSOR]][%[[INDEX]]]
// CHECK-NEXT:   yield %[[VAL]]
// CHECK-NEXT: else
// CHECK-NEXT:   yield %[[V]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RET]]

// -----

func.func private @exp(%p0: tensor<32x64xi32>, %i: index, %j: index) -> i32

#map = #zkx.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]">
#map1 = #zkx.indexing_map<"(d0, d1)[s0, s1] -> (d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1], s1 in [0, 1]">

func.func @materialize(%input: tensor<32x64xi32>, %i: index, %j: index)
    -> !zkx_gpu.indexed_vector<32x2x2xi32, #map1> {
  %0 = zkx_gpu.materialize @exp(%input) at #map(%i, %j)
    : (tensor<32x64xi32>) -> !zkx_gpu.indexed_vector<32x2x2xi32, #map1>
  func.return %0 : !zkx_gpu.indexed_vector<32x2x2xi32, #map1>
}
// CHECK-DAG: #[[$MAP:.*]] = #zkx.indexing_map<"(d0, d1)[s0, s1] -> (d1 * 32 + d0 * 2 + s0, s1)
// CHECK-DAG: #[[$MAP1:.*]] = #zkx.indexing_map<"(d0, d1)[s0, s1] -> (d0 * 2 + s0, s1)

// CHECK: @materialize(%[[INPUT:.*]]: tensor<32x64xi32>, %[[INDEX1:.*]]: index, %[[INDEX2:.*]]: index)

// CHECK:      %[[INIT_VEC:.*]] = arith.constant {{.*}} : vector<2x2xi32>
// CHECK:      zkx.loop (%[[INDEX1]], %[[INDEX2]])[%[[S0:.*]], %[[S1:.*]]]
// CHECK-SAME:   -> (%[[MAP_RESULT1:.*]], %[[MAP_RESULT2:.*]]) in
// CHECK-SAME:   #[[$MAP]] iter_args(%[[ITER_ARG:.*]] = %[[INIT_VEC]])

// CHECK: %[[PURE_CALL:.*]] = zkx.pure_call @exp(%[[INPUT]], %[[MAP_RESULT1]], %[[MAP_RESULT2]])
// CHECK: vector.insert %[[PURE_CALL]], %[[ITER_ARG]] [%[[S0]], %[[S1]]]
// CHECK zkx.yield %{{.*}} : vector<2x2xi32>

// -----

#map = #zkx.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]">
#map1 = #zkx.indexing_map<"(d0, d1) -> (d0 mod 16, d1), domain: d0 in [0, 32], d1 in [0, 2]">

func.func @insert(%input: !zkx_gpu.indexed_vector<32x64xf32, #map>,
    %i: index, %j: index, %output: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = zkx_gpu.insert %input(%i, %j) into %output at #map1
    : !zkx_gpu.indexed_vector<32x64xf32, #map> -> tensor<32x64xf32>
  func.return %0 : tensor<32x64xf32>
}
// CHECK-DAG: #[[$MAP:.*]] = #zkx.indexing_map<"(d0, d1)[s0, s1] -> (d1 * 32 + d0 * 2 + s0, s1)
// CHECK-DAG: #[[$MAP1:.*]] = #zkx.indexing_map<"(d0, d1) -> (d0 mod 16, d1)

// CHECK:      @insert(%[[INPUT:.*]]: !zkx_gpu.indexed_vector<32x64xf32, #[[$MAP]]>,
// CHECK-SAME:   %[[I:.*]]: index, %[[J:.*]]: index,
// CHECK-SAME:   %[[OUTPUT:.*]]: tensor<32x64xf32>)

// CHECK:      zkx.loop (%[[I]], %[[J]])[%[[S0:.*]], %[[S1:.*]]] ->
// CHECK-SAME:   (%[[MAP_RESULT1:.*]], %[[MAP_RESULT2:.*]]) in #[[$MAP]]
// CHECK-SAME:   iter_args(%[[TENSOR:.*]] = %[[OUTPUT]])

// CHECK: %[[SCALAR:.*]] = vector.extract %{{.*}}[%[[S0]], %[[S1]]]
// CHECK-SAME: : f32 from vector<2x2xf32>
// CHECK: %[[MAP1_RESULT:.*]]:2 = zkx.apply_indexing
// CHECK-SAME: #[[$MAP1]](%[[MAP_RESULT1]], %[[MAP_RESULT2]])
// CHECK: %[[NEW_TENSOR:.*]] = tensor.insert %[[SCALAR]]
// CHECK-SAME: into %[[TENSOR]][%[[MAP1_RESULT]]#0, %[[MAP1_RESULT]]#1]
// CHECK: zkx.yield %[[NEW_TENSOR]]

// -----

func.func private @exp(%p0: tensor<32x64xi32>, %i: index, %j: index) -> i32

#map = #zkx.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]">
#map1 = #zkx.indexing_map<"(d0, d1)[s0, s1] -> (d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1], s1 in [0, 1]">
#map2 = #zkx.indexing_map<"(d0, d1) -> (d0, d1), domain: d0 in [0, 32], d1 in [0, 2]">

func.func @materialize_and_insert(%input: tensor<32x64xi32>, %i: index,
    %j: index, %output: tensor<32x64xi32>) -> tensor<32x64xi32> {
  %0 = zkx_gpu.materialize @exp(%input) at #map(%i, %j)
    : (tensor<32x64xi32>) -> !zkx_gpu.indexed_vector<32x2x2xi32, #map1>
  %1 = zkx_gpu.insert %0(%i, %j) into %output at #map2
    : !zkx_gpu.indexed_vector<32x2x2xi32, #map1> -> tensor<32x64xi32>
  func.return %1 : tensor<32x64xi32>
}
// CHECK-NOT: unrealized_conversion_cast

// -----

// shuffle_reduce with field type: cast to i32, shuffle, cast back.
func.func @combiner_pf(%a: !field.pf<2013265921:i32>,
    %b: !field.pf<2013265921:i32>) -> !field.pf<2013265921:i32> {
  return %a : !field.pf<2013265921:i32>
}

func.func @shuffler_pf(%a: !field.pf<2013265921:i32>)
    -> !field.pf<2013265921:i32> {
  %ret = zkx_gpu.shuffle_reduce (%a) to 1 combiner=@combiner_pf
    : !field.pf<2013265921:i32>
  return %ret : !field.pf<2013265921:i32>
}
// CHECK-LABEL: @shuffler_pf
// CHECK:         field.bitcast {{.*}} : !pf_babybear -> i32
// CHECK:         gpu.shuffle down
// CHECK:         field.bitcast {{.*}} : i32 -> !pf_babybear

// -----

// materialize with field type: field.constant for zero init.
func.func private @exp_pf(tensor<8x!field.pf<2013265921:i32>>, index)
    -> !field.pf<2013265921:i32>

#map = #zkx.indexing_map<"(d0)[s0] -> (s0), domain: d0 in [0, 1], s0 in [0, 7]">

func.func @materialize_pf(%input: tensor<8x!field.pf<2013265921:i32>>,
    %i: index)
    -> !zkx_gpu.indexed_vector<1x8x!field.pf<2013265921:i32>, #map> {
  %0 = zkx_gpu.materialize @exp_pf(%input) at #map(%i)
    : (tensor<8x!field.pf<2013265921:i32>>)
    -> !zkx_gpu.indexed_vector<1x8x!field.pf<2013265921:i32>, #map>
  func.return %0
    : !zkx_gpu.indexed_vector<1x8x!field.pf<2013265921:i32>, #map>
}
// CHECK-LABEL: @materialize_pf
// CHECK:         field.constant dense<0> : vector<8x!pf_babybear>
