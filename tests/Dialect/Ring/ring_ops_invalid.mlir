// RUN: prime-ir-opt --split-input-file --verify-diagnostics %s

// base_convert changes the RNS basis but must preserve the ring degree N.
func.func @base_convert_bad_degree(
    %x: !ring.rq<[12289], 8 : i32>) -> !ring.rq<[12289], 16 : i32> {
  // expected-error @+1 {{input and output rings must share the degree N}}
  %y = ring.base_convert %x : !ring.rq<[12289], 8 : i32> to !ring.rq<[12289], 16 : i32>
  return %y : !ring.rq<[12289], 16 : i32>
}
