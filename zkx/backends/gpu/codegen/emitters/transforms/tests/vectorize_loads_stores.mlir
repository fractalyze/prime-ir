// RUN: emitters_opt -allow-unregistered-dialect %s -split-input-file \
// RUN: -zkx-gpu-vectorize-loads-stores -cse -canonicalize \
// RUN: | FileCheck %s

#map = #zkx.indexing_map<"(d0)[s0] -> (d0 * 2 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @simple_read(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK: #[[$MAP:.*]] = #zkx.indexing_map<"(d0) -> (d0 * 2), domain: d0 in [0, 63]">
// CHECK-LABEL: @simple_read
// CHECK-SAME:     (%[[ARG0:.*]]: tensor
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[ITER:.*]] =
// CHECK:         %[[BASE:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]])
// CHECK-NEXT:    %[[V:.*]] = vector.transfer_read %[[ARG0]][%[[BASE]]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]]
// CHECK-NEXT:      vector.extract %[[V]][%[[J]]]
// CHECK-NEXT:      addf

// -----

#map = #zkx.indexing_map<"(d0)[s0] -> (d0 * 4 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 3]">
func.func @simple_read(%arg0: tensor<256xf16>) -> (f16) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f16
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f16 {
    %inner = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter1 = %iter) -> f16 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<256xf16>
      %added = arith.addf %iter1, %extracted : f16
      scf.yield %added : f16
    }
    scf.yield %inner : f16
  }
  return %outer : f16
}
// CHECK: #[[$MAP:.*]] = #zkx.indexing_map<"(d0) -> (d0 * 4), domain: d0 in [0, 63]">
// CHECK-LABEL: @simple_read
// CHECK-SAME:     (%[[ARG0:.*]]: tensor
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[ITER:.*]] =
// CHECK:         %[[BASE:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]])
// CHECK-NEXT:    %[[V:.*]] = vector.transfer_read %[[ARG0]][%[[BASE]]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]]
// CHECK-NEXT:      vector.extract %[[V]][%[[J]]]
// CHECK-NEXT:      addf

// -----

#map = #zkx.indexing_map<"(d0)[s0] -> (d0 * 8 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 7]">
func.func @simple_read(%arg0: tensor<512xi8>) -> (i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0 : i8
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> i8 {
    %inner = scf.for %j = %c0 to %c8 step %c1 iter_args(%iter1 = %iter) -> i8 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<512xi8>
      %added = arith.addi %iter1, %extracted : i8
      scf.yield %added : i8
    }
    scf.yield %inner : i8
  }
  return %outer : i8
}
// CHECK: #[[$MAP:.*]] = #zkx.indexing_map<"(d0) -> (d0 * 8), domain: d0 in [0, 63]">
// CHECK-LABEL: @simple_read
// CHECK-SAME:     (%[[ARG0:.*]]: tensor
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[ITER:.*]] =
// CHECK:         %[[BASE:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]])
// CHECK-NEXT:    %[[V:.*]] = vector.transfer_read %[[ARG0]][%[[BASE]]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]]
// CHECK-NEXT:      vector.extract %[[V]][%[[J]]]
// CHECK-NEXT:      addi

// -----

#map = #zkx.indexing_map<"(d0)[s0] -> (d0 * 2 + s0 + 1),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @misaligned_indexing_map(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @misaligned_indexing_map
// CHECK-NOT: vector.transfer_read

// -----

#map = #zkx.indexing_map<"(d0)[s0] -> (d0 * 3 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @misaligned_indexing_map_2(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @misaligned_indexing_map_2
// CHECK-NOT: vector.transfer_read

// -----

#map = #zkx.indexing_map<"(d0)[s0] -> (3 * d0 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @misaligned_shape(%arg0: tensor<192xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<192xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @misaligned_shape
// CHECK-NOT: vector.transfer_read

// -----

#map = #zkx.indexing_map<"(d0)[s0] -> (d0 + s0 * 2),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @wrong_stride(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @wrong_stride
// CHECK-NOT: vector.transfer_read

// -----

// This is vectorizable, but not currently supported.
func.func @layout(%arg0: tensor<2x64xf32, dense<[0, 1]> : tensor<2xi64>>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %extracted = tensor.extract %arg0[%j, %i]
        : tensor<2x64xf32, dense<[0, 1]> : tensor<2xi64>>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @layout
// CHECK-NOT: vector.transfer_read

// -----

func.func @simple_write(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64xf32> {
    %inserted = tensor.insert %cst into %iter[%j] : tensor<64xf32>
    scf.yield %inserted : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}
// CHECK-LABEL: @simple_write
// CHECK-SAME:     (%[[ARG0:.*]]: tensor{{.*}})
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[V:.*]] = scf.for
// CHECK-SAME:      (vector<4xf32>)
// CHECK-NEXT:    vector.insert
// CHECK-NEXT:    scf.yield
// CHECK:       %[[WRITTEN:.*]] = vector.transfer_write %[[V]], %[[ARG0]][%[[C0]]]
// CHECK-NEXT:  return %[[WRITTEN]]

// -----

func.func @write_with_use(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64xf32> {
    %inserted = tensor.insert %cst into %iter[%j] : tensor<64xf32>
    "dummy.op1"(%inserted) : (tensor<64xf32>) -> ()
    scf.yield %inserted : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}
// CHECK-LABEL: @write_with_use
// CHECK-NOT:   transfer_write

// -----

func.func @write_not_to_iter_arg(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64xf32> {
    %inserted = tensor.insert %cst into %arg0[%j] : tensor<64xf32>
    scf.yield %inserted : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}

// CHECK-LABEL: @write_not_to_iter_arg
// CHECK-NOT:   transfer_write

// -----

func.func @write_not_yielded(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64xf32> {
    %inserted = tensor.insert %cst into %arg0[%j] : tensor<64xf32>
    scf.yield %arg0 : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}
// CHECK-LABEL: @write_not_yielded
// CHECK-NOT:   transfer_write

// -----

#map = #zkx.indexing_map<"(d0, d1)[s0] -> (d1 * 2 + d0 + s0 * 512),"
  "domain: d0 in [0, 7], d1 in [0, 255], s0 in [0, 7]">
#map1 = #zkx.indexing_map<
  "(d0, d1, d2)[s0] -> (d0 * 32 + d2 * 2 + d1 + s0 * 512),"
  "domain: d0 in [0, 7], d1 in [0, 1], d2 in [0, 255], s0 in [0, 7]">
func.func @multiple(%arg0: tensor<131072xf32>, %arg1: tensor<4096xbf16>,
      %arg2: tensor<32xf32>, %arg3: tensor<131072xf32>,
      %arg4: index) -> (tensor<131072xf32>, f32) {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %extracted1 = tensor.extract %arg2[%arg4] : tensor<32xf32>
  %0:2 = scf.for %i = %c0 to %c8 step %c1 iter_args(%iter0 = %arg3, %iter1 = %cst) -> (tensor<131072xf32>, f32) {
    %1:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter2 = %iter0, %iter3 = %iter1) -> (tensor<131072xf32>, f32) {
      %2 = zkx.apply_indexing #map(%j, %arg4)[%i]
      %idx = zkx.apply_indexing #map1(%i, %j, %arg4)[%i]
      %extracted2 = tensor.extract %arg0[%idx] : tensor<131072xf32>
      %extracted3 = tensor.extract %arg1[%2] : tensor<4096xbf16>
      %3 = arith.extf %extracted3 : bf16 to f32
      %4 = arith.addf %extracted2, %3 : f32
      %5 = arith.addf %extracted1, %4 : f32
      %6 = arith.addf %iter3, %5 : f32
      %inserted = tensor.insert %5 into %iter2[%idx] : tensor<131072xf32>
      scf.yield %inserted, %6 : tensor<131072xf32>, f32
    }
    scf.yield %1#0, %1#1 : tensor<131072xf32>, f32
  }
  return %0#0, %0#1 : tensor<131072xf32>, f32
}
// CHECK-DAG: #[[$MAP:.*]] = #zkx.indexing_map<"(d0, d1) -> (d0 * 2 + d1 * 512), domain: d0 in [0, 255], d1 in [0, 7]">
// CHECK-DAG: #[[$MAP1:.*]] = #zkx.indexing_map<"(d0, d1, d2) -> (d0 * 32 + d1 * 2 + d2 * 512), domain: d0 in [0, 7], d1 in [0, 255], d2 in [0, 7]">
// CHECK-LABEL: @multiple
// CHECK-SAME: (%[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}, %[[ARG4:.*]]: index)
// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK:      scf.for %[[I:.*]] = %[[C0]]
// CHECK-DAG:  %[[BASE:.*]] = zkx.apply_indexing #[[$MAP]](%[[ARG4]], %[[I]])
// CHECK-DAG:  %[[IDX:.*]] = zkx.apply_indexing #[[$MAP1]](%[[I]], %[[ARG4]], %[[I]])
// CHECK:      %[[READ1:.*]] = vector.transfer_read %[[ARG1]][%[[BASE]]]
// CHECK:      %[[READ2:.*]] = vector.transfer_read %[[ARG0]][%[[IDX]]]
// CHECK:      %[[INNER:.*]]:2 = scf.for %[[J:.*]] = %[[C0]] {{.*}} iter_args(%[[F:.*]] = {{.*}}, %[[V:.*]] = {{.*}}) -> (f32, vector<2xf32>)
// CHECK-DAG:  vector.extract %[[READ1]][%[[J]]]
// CHECK-DAG:  vector.extract %[[READ2]][%[[J]]]
// CHECK:      extf
// CHECK-NEXT: addf
// CHECK-NEXT: %[[TO_INSERT:.*]] = arith.addf
// CHECK-NEXT: %[[TO_YIELD:.*]] = arith.addf
// CHECK-NEXT: %[[V_NEXT:.*]] = vector.insert %[[TO_INSERT]], %[[V]] [%[[J]]]
// CHECK-NEXT: scf.yield %[[TO_YIELD]], %[[V_NEXT]]
// CHECK:      %[[WRITTEN:.*]] = vector.transfer_write %[[INNER]]#1, %{{.*}}[%[[IDX]]]
// CHECK:      scf.yield %[[WRITTEN]], %[[INNER]]#0

// -----

#map = #zkx.indexing_map<"(d0)[s0] -> ((d0 * 4) mod 64 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @remainder_with_modulo(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK: #[[$MAP:.*]] = #zkx.indexing_map<"(d0) -> ((d0 mod 16) * 4),
// CHECK-LABEL: @remainder_with_modulo
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.for %[[I:.*]] = %[[C0]]
// CHECK: %[[BASE:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]]
// CHECK: vector.transfer_read {{.*}}[%[[BASE]]]

// -----

#map = #zkx.indexing_map<"(d0)[s0] -> ((d0 * 4) mod 65 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @remainder_with_modulo_misaligned(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = zkx.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @remainder_with_modulo_misaligned
// CHECK-NOT: vector.transfer_read

// -----

#map0 = #zkx.indexing_map<"(d0) -> (d0 + 5),"
  "domain: d0 in [0, 63]">
#map1 = #zkx.indexing_map<"(d0)[s0] -> (d0 * 2 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
module {
  func.func @apply_indexing_sequence(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %offset = zkx.apply_indexing #map0(%i)
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %idx = zkx.apply_indexing #map1(%offset)[%j]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK: #[[$MAP0:.*]] = #zkx.indexing_map<"(d0) -> (d0 * 2 + 10),
// CHECK-SAME:                                  domain: d0 in [0, 63]">
// CHECK-LABEL: @apply_indexing_sequence
// CHECK: %[[BASE:.*]] = zkx.apply_indexing #[[$MAP0]]
// CHECK: vector.transfer_read {{.*}}[%[[BASE]]]

// -----


#map0 = #zkx.indexing_map<"(d0) -> (d0 + 5),"
  "domain: d0 in [0, 63]">
#map1 = #zkx.indexing_map<"(d0)[s0] -> (d0 * 2 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
module {
  func.func @apply_indexing_sequence_same_block(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        // Usually, this will be hoisted by LICM or folded, so we do not detect
        // this pattern.
        %offset = zkx.apply_indexing #map0(%i)
        %idx = zkx.apply_indexing #map1(%offset)[%j]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK-LABEL: @apply_indexing_sequence_same_block
// CHECK-NOT: vector.transfer_read

// -----

!pf = !field.pf<2013265921 : i32, true>
func.func @vectorize_store_field_type(%arg0: tensor<64x!pf>, %val: !pf) -> tensor<64x!pf> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64x!pf> {
    %inserted = tensor.insert %val into %iter[%j] : tensor<64x!pf>
    scf.yield %inserted : tensor<64x!pf>
  }
  return %loop : tensor<64x!pf>
}
// CHECK-LABEL: @vectorize_store_field_type
// CHECK-SAME:     (%[[ARG0:.*]]: tensor{{.*}}, %[[VAL:.*]]:
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[ZERO:.*]] = field.constant dense<0> : vector<4x!pf
// CHECK:       %[[V:.*]] = scf.for
// CHECK-SAME:      (vector<4x!pf
// CHECK-NEXT:    vector.insert
// CHECK-NEXT:    scf.yield
// CHECK:       %[[WRITTEN:.*]] = vector.transfer_write %[[V]], %[[ARG0]][%[[C0]]]
// CHECK-NEXT:  return %[[WRITTEN]]

// TODO(jeong0982): Add vectorize_store tests for BinaryFieldType and
// ExtensionFieldType once prime_ir support is complete.
// - BinaryFieldType: requires updating prime_ir pin (bf type not yet available)
// - ExtensionFieldType: getZeroAttr crashes in createConstantAttr
