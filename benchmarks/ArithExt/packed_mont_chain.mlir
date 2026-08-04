// Copyright 2025 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Microbenchmark for SpecializeArithToAVX (see run_bench.sh for how to run).
//
// Each round is one packed Montgomery multiply-accumulate step over BabyBear
// (q = 2013265921, q^-1 mod 2^32 = 2281701377): the
// mului_extended/muli/subi/addi mix and low/high gather chaining that the
// field-to-llvm pipeline emits for absorb-style kernels, on vector<16xi32>.
//
// Two kernels:
// - @chain:  one loop-carried vector, latency-bound (absorb-chain analog).
// - @sweep:  independent rounds over a buffer, throughput-bound. The buffer
//   size selects the cache level (see run_bench.sh).

func.func private @rtclock() -> f64

// One Montgomery REDC round: x <- REDC(x * y) + q (kept unreduced; the
// values wrap mod 2^32, which is fine for measurement purposes).
func.func @round(%x: vector<16xi32>, %y: vector<16xi32>) -> vector<16xi32> {
  // q^-1 mod 2^32 (= 2281701377) as signed i32; the subtraction form of REDC
  // below pairs with q^-1, not -q^-1.
  %qInv = arith.constant dense<-2013265919> : vector<16xi32>
  %q = arith.constant dense<2013265921> : vector<16xi32>
  %t:2 = arith.mului_extended %x, %y : vector<16xi32>
  %m = arith.muli %t#0, %qInv : vector<16xi32>
  %u:2 = arith.mului_extended %m, %q : vector<16xi32>
  %r = arith.subi %t#1, %u#1 : vector<16xi32>
  %x1 = arith.addi %r, %q : vector<16xi32>
  return %x1 : vector<16xi32>
}

func.func @chain(%x0: vector<16xi32>, %y: vector<16xi32>, %reps: index)
    -> vector<16xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %res = scf.for %i = %c0 to %reps step %c1
      iter_args(%x = %x0) -> (vector<16xi32>) {
    %x1 = func.call @round(%x, %y) : (vector<16xi32>, vector<16xi32>)
        -> vector<16xi32>
    scf.yield %x1 : vector<16xi32>
  }
  return %res : vector<16xi32>
}

func.func @sweep(%buf: memref<?xvector<16xi32>>, %y: vector<16xi32>,
                 %outer: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %buf, %c0 : memref<?xvector<16xi32>>
  scf.for %t = %c0 to %outer step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %x = memref.load %buf[%j] : memref<?xvector<16xi32>>
      %x1 = func.call @round(%x, %y) : (vector<16xi32>, vector<16xi32>)
          -> vector<16xi32>
      %x2 = func.call @round(%x1, %y) : (vector<16xi32>, vector<16xi32>)
          -> vector<16xi32>
      %x3 = func.call @round(%x2, %y) : (vector<16xi32>, vector<16xi32>)
          -> vector<16xi32>
      %x4 = func.call @round(%x3, %y) : (vector<16xi32>, vector<16xi32>)
          -> vector<16xi32>
      memref.store %x4, %buf[%j] : memref<?xvector<16xi32>>
    }
  }
  return
}

func.func @init(%buf: memref<?xvector<16xi32>>, %x0: vector<16xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = memref.dim %buf, %c0 : memref<?xvector<16xi32>>
  scf.for %j = %c0 to %n step %c1 {
    memref.store %x0, %buf[%j] : memref<?xvector<16xi32>>
  }
  return
}

// Times @sweep over a buffer of %n vectors; returns seconds per round.
func.func @timed_sweep(%n: index, %outer: index, %x0: vector<16xi32>,
                       %y: vector<16xi32>) -> f64 {
  %c0 = arith.constant 0 : index
  %buf = memref.alloc(%n) : memref<?xvector<16xi32>>
  func.call @init(%buf, %x0)
      : (memref<?xvector<16xi32>>, vector<16xi32>) -> ()
  // Warm-up pass.
  %c1 = arith.constant 1 : index
  func.call @sweep(%buf, %y, %c1)
      : (memref<?xvector<16xi32>>, vector<16xi32>, index) -> ()
  %t0 = func.call @rtclock() : () -> f64
  func.call @sweep(%buf, %y, %outer)
      : (memref<?xvector<16xi32>>, vector<16xi32>, index) -> ()
  %t1 = func.call @rtclock() : () -> f64
  // Keep the result observable; the reductions also let the harness diff
  // results across configs.
  %x = memref.load %buf[%c0] : memref<?xvector<16xi32>>
  %xXor = vector.reduction <xor>, %x : vector<16xi32> into i32
  vector.print %xXor : i32
  %xAdd = vector.reduction <add>, %x : vector<16xi32> into i32
  vector.print %xAdd : i32
  %dt = arith.subf %t1, %t0 : f64
  %nI64 = arith.index_cast %n : index to i64
  %outerI64 = arith.index_cast %outer : index to i64
  %c4 = arith.constant 4 : i64
  %roundsA = arith.muli %nI64, %outerI64 : i64
  %rounds = arith.muli %roundsA, %c4 : i64
  %roundsF = arith.sitofp %rounds : i64 to f64
  %perRound = arith.divf %dt, %roundsF : f64
  memref.dealloc %buf : memref<?xvector<16xi32>>
  return %perRound : f64
}

func.func @main() {
  %x0 = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8,
                              9, 10, 11, 12, 13, 14, 15, 16]> : vector<16xi32>
  %y = arith.constant dense<[901943132, 447872240, 1755702243, 917797396,
                             1663489668, 977253245, 473039071, 100480371,
                             1524823050, 933205805, 67650069, 1893806164,
                             225804680, 1911871336, 1189862080,
                             688018430]> : vector<16xi32>

  // Latency-bound chain: 1<<22 dependent rounds.
  %chain_reps = arith.constant 4194304 : index
  // Warm-up.
  %warm_reps = arith.constant 16384 : index
  %w = func.call @chain(%x0, %y, %warm_reps)
      : (vector<16xi32>, vector<16xi32>, index) -> vector<16xi32>
  %wXor = vector.reduction <xor>, %w : vector<16xi32> into i32
  vector.print %wXor : i32
  %t0 = func.call @rtclock() : () -> f64
  %r = func.call @chain(%x0, %y, %chain_reps)
      : (vector<16xi32>, vector<16xi32>, index) -> vector<16xi32>
  %t1 = func.call @rtclock() : () -> f64
  %rXor = vector.reduction <xor>, %r : vector<16xi32> into i32
  vector.print %rXor : i32
  %rAdd = vector.reduction <add>, %r : vector<16xi32> into i32
  vector.print %rAdd : i32
  %dt = arith.subf %t1, %t0 : f64
  %repsI64 = arith.index_cast %chain_reps : index to i64
  %repsF = arith.sitofp %repsI64 : i64 to f64
  %perRound = arith.divf %dt, %repsF : f64
  // ns per round.
  %c1e9 = arith.constant 1.0e9 : f64
  %chainNs = arith.mulf %perRound, %c1e9 : f64
  vector.print %chainNs : f64

  // Throughput-bound sweeps: 16 KiB (L1-resident) and 1 MiB (L2-resident)
  // buffers, ~21M rounds each.
  %l1_n = arith.constant 256 : index
  %l1_outer = arith.constant 20000 : index
  %l1_per = func.call @timed_sweep(%l1_n, %l1_outer, %x0, %y)
      : (index, index, vector<16xi32>, vector<16xi32>) -> f64
  %l1_ns = arith.mulf %l1_per, %c1e9 : f64
  vector.print %l1_ns : f64

  %l2_n = arith.constant 16384 : index
  %l2_outer = arith.constant 320 : index
  %l2_per = func.call @timed_sweep(%l2_n, %l2_outer, %x0, %y)
      : (index, index, vector<16xi32>, vector<16xi32>) -> f64
  %l2_ns = arith.mulf %l2_per, %c1e9 : f64
  vector.print %l2_ns : f64
  return
}
