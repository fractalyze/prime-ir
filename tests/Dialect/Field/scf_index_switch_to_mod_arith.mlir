// RUN: prime-ir-opt -field-to-mod-arith -split-input-file %s | FileCheck %s -enable-var-scope

// Upstream's SCF structural type conversion (populateSCFStructuralTypeConversions)
// covers scf.for/if/while but NOT scf.index_switch. Without the supplementary
// pattern in addStructuralConversionPatterns, a field-typed index_switch is left
// unconverted while its body's field ops lower to mod_arith, and the pass fails
// to legalize an unresolved materialization at the yields. This pins that an
// index_switch carrying a prime-field result lowers to mod_arith with the op.

!PF = !field.pf<97:i32>

// CHECK-LABEL: @lower_index_switch
// CHECK-SAME:  (%[[SEL:.*]]: index, %[[A:.*]]: [[T:.*]], %[[B:.*]]: [[T]]) -> [[T]]
func.func @lower_index_switch(%sel: index, %a: !PF, %b: !PF) -> !PF {
  // CHECK:      %[[R:.*]] = scf.index_switch %[[SEL]] -> [[T]]
  // CHECK:        %[[ADD:.*]] = mod_arith.add %[[A]], %[[B]] : [[T]]
  // CHECK:        scf.yield %[[ADD]] : [[T]]
  // CHECK:      default
  // CHECK:        scf.yield %[[A]] : [[T]]
  // CHECK:      return %[[R]] : [[T]]
  %res = scf.index_switch %sel -> !PF
  case 0 {
    %s = field.add %a, %b : !PF
    scf.yield %s : !PF
  }
  default {
    scf.yield %a : !PF
  }
  return %res : !PF
}
