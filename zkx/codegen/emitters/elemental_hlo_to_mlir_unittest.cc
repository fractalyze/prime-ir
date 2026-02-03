/* Copyright 2024 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "zkx/codegen/emitters/elemental_hlo_to_mlir.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.h"
#include "zkx/codegen/emitters/ir/zkx_ops.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/testlib/file_check.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/status_macros.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx::emitters {
namespace {

using ::testing::HasSubstr;

class ElementalHloToMlirTest : public HloTestBase {
 public:
  ElementalHloToMlirTest() {
    context_.loadDialect<mlir::tensor::TensorDialect, mlir::func::FuncDialect,
                         mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::math::MathDialect, mlir::scf::SCFDialect,
                         mlir::mhlo::MhloDialect, mlir::LLVM::LLVMDialect,
                         mlir::DLTIDialect, ZkxDialect, gpu::ZkxGpuDialect>();
  }

  // Converts the root subgraph of the entry function of the given hlo module to
  // MLIR.
  absl::Status Run(const std::string_view hlo,
                   const std::string_view filecheck_str,
                   std::function<EpilogueSpecification(HloComputation* entry)>
                       epilogue_spec_fn = nullptr,
                   bool set_zkx_entry = false,
                   std::optional<BackendKind> zkx_backend = std::nullopt) {
    auto hlo_module = ParseAndReturnVerifiedModule(hlo).value();

    mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&context_),
                                       &context_);
    auto module = llvm_ir::CreateMlirModuleOp(builder.getLoc());
    (*module)->setAttr(
        mlir::DLTIDialect::kDataLayoutAttrName,
        mlir::parseAttribute("#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>",
                             builder.getContext()));
    builder.setInsertionPointToStart(module->getBody());
    auto* entry_computation = hlo_module->entry_computation();
    std::vector<EpilogueSpecification> epilogue_spec;
    if (epilogue_spec_fn) {
      epilogue_spec.push_back(epilogue_spec_fn(entry_computation));
    }
    PartitionedComputations partitioned_computations(entry_computation,
                                                     &context_, epilogue_spec);
    auto fns = partitioned_computations.DeclareFunctions(module.get());
    auto entry_func = fns[&partitioned_computations
                               .FindPartitionedComputation(entry_computation)
                               .GetRootSubgraph()];
    if (set_zkx_entry) {
      entry_func->setAttr("zkx.entry", mlir::UnitAttr::get(&context_));
    }
    if (zkx_backend) {
      SetBackendKind(&context_, entry_func, *zkx_backend);
    }
    auto& entry_pc =
        partitioned_computations.FindPartitionedComputation(entry_computation);
    auto call_targets = partitioned_computations.CreateCallTargetProvider(fns);
    TF_RETURN_IF_ERROR(SubgraphToMlirFunction(
        entry_pc, entry_pc.GetRootSubgraph(), entry_func, call_targets));
    if (!partitioned_computations.epilogues().empty()) {
      const auto& epilogue = partitioned_computations.epilogues().front();
      TF_RETURN_IF_ERROR(SubgraphToMlirFunction(entry_pc, epilogue,
                                                fns[&epilogue], call_targets));
    }

    // Canonicalize and CSE for better readability of check tests.
    mlir::PassManager pm(&context_);
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    TF_RET_CHECK(pm.run(module.get()).succeeded());

    std::string out;
    llvm::raw_string_ostream stream(out);
    stream << module.get();

    TF_ASSIGN_OR_RETURN(auto filecheck_result,
                        RunFileCheck(out, filecheck_str));
    TF_RET_CHECK(filecheck_result);
    return absl::OkStatus();
  }

  mlir::MLIRContext context_;
};

TEST_F(ElementalHloToMlirTest, Reduce) {
  TF_EXPECT_OK(Run(R"(
    add {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT sum = s32[] add(p0, p1)
    }

    ENTRY main {
      p0 = s32[10,20,30,40] parameter(0)
      p1 = s32[] parameter(1)
      ROOT r = s32[10,30] reduce(p0, p1), dimensions={1,3},
                                          to_apply=add
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:   %[[ARG0:.*]]: tensor<10x20x30x40xi32>
    // CHECK-SAME:   %[[ARG1:.*]]: tensor<i32>
    // CHECK-SAME:   %[[X:.*]]: index {{.*}}, %[[Y:.*]]: index {{.*}} -> i32
    // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
    // CHECK-DAG:  %[[C20:.*]] = arith.constant 20
    // CHECK-DAG:  %[[C40:.*]] = arith.constant 40
    // CHECK:      %[[INIT:.*]] = tensor.extract %[[ARG1]][]
    // CHECK:      %[[RET:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C20]]
    // CHECK-SAME:   step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]])
    // CHECK:        %[[RET_INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C40]]
    // CHECK-SAME:     iter_args(%[[ACC_INNER:.*]] = %[[ACC]])
    // CHECK:          %[[VAL:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:        [%[[X]], %[[I]], %[[Y]], %[[J]]]
    // CHECK:          %[[UPD:.*]] = func.call @add_sum(%[[ACC_INNER]],
    // CHECK-SAME:                                      %[[VAL]])
    // CHECK:          scf.yield %[[UPD]]
    // CHECK:        }
    // CHECK:        scf.yield %[[RET_INNER]]
    // CHECK:      }
    // CHECK:      return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, ReduceUnsigned) {
  TF_EXPECT_OK(Run(R"(
    add {
      p0 = u32[] parameter(0)
      p1 = u32[] parameter(1)
      ROOT sum = u32[] add(p0, p1)
    }

    ENTRY main {
      p0 = u32[10,20,30,40] parameter(0)
      p1 = u32[] parameter(1)
      ROOT r = u32[10,30] reduce(p0, p1), dimensions={1,3},
                                          to_apply=add
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:   %[[ARG0:.*]]: tensor<10x20x30x40xi32>
    // CHECK-SAME:   %[[ARG1:.*]]: tensor<i32>
    // CHECK-SAME:   %[[X:.*]]: index {{.*}}, %[[Y:.*]]: index {{.*}} -> i32
    // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
    // CHECK-DAG:  %[[C20:.*]] = arith.constant 20
    // CHECK-DAG:  %[[C40:.*]] = arith.constant 40
    // CHECK:      %[[INIT:.*]] = tensor.extract %[[ARG1]][]
    // CHECK:      %[[RET:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C20]]
    // CHECK-SAME:   step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]])
    // CHECK:        %[[RET_INNER:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C40]]
    // CHECK-SAME:     iter_args(%[[ACC_INNER:.*]] = %[[ACC]])
    // CHECK:          %[[VAL:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:        [%[[X]], %[[I]], %[[Y]], %[[J]]]
    // CHECK:          %[[UPD:.*]] = func.call @add_sum(%[[ACC_INNER]],
    // CHECK-SAME:                                      %[[VAL]])
    // CHECK:          scf.yield %[[UPD]]
    // CHECK:        }
    // CHECK:        scf.yield %[[RET_INNER]]
    // CHECK:      }
    // CHECK:      return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, ReduceWindow) {
  TF_EXPECT_OK(Run(R"(
    add {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT sum = s32[] add(p0, p1)
    }

    ENTRY main {
      p0 = s32[42,12,8] parameter(0)
      p1 = s32[] parameter(1)
      ROOT r = s32[42,3,8] reduce-window(p0, p1), window={
                                                size=1x1x7
                                                stride=1x4x1
                                                pad=0_0x0_0x3_3
                                               },
                                               to_apply=add
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:   %[[ARG0:.*]]: tensor<42x12x8xi32>
    // CHECK-SAME:   %[[ARG1:.*]]: tensor<i32>
    // CHECK-SAME:   %[[X:arg[0-9]*]]: index {{[^}]*}}},
    // CHECK-SAME:   %[[Y:arg[0-9]*]]: index {{[^}]*}}},
    // CHECK-SAME:   %[[Z:arg[0-9]*]]: index {{[^}]*}}}) -> i32
    // CHECK:      %[[INIT:.*]] = tensor.extract %[[ARG1]][]
    // NOTE: zkx uses `zkx.loop` (EmitZkxLoopOp) instead of XLA's
    // `scf.for` (EmitLoopNest) to preserve indexing map semantics.
    // CHECK:      %[[RET:.*]] = zkx.loop
    // CHECK-SAME:   in #zkx.indexing_map<"(d0, d1, d2)[s0] -> (d0, d1 * 4, d2 + s0 - 3),
    // CHECK-SAME:     domain: d0 in [0, 41], d1 in [0, 2], d2 in [0, 7],
    // CHECK-SAME:     s0 in [0, 6], d2 + s0 in [3, 10]">
    // CHECK-SAME:   iter_args({{.*}} = %[[INIT]])
    // CHECK:      tensor.extract %[[ARG0]]
    // CHECK:      func.call @add_sum
    // CHECK:      zkx.yield
    // CHECK:      return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, ReduceWindowWithRescaling) {
  TF_EXPECT_OK(Run(R"(
    add {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT sum = s32[] add(p0, p1)
    }

    ENTRY main {
      p0 = s32[42,12,8] parameter(0)
      p1 = s32[] parameter(1)
      ROOT r = s32[19,12,8] reduce-window(p0, p1), window={
                                                size=8x1x1
                                                stride=4x1x1
                                                pad=0_0x0_0x0_0
                                                lhs_dilate=2x1x1
                                               },
                                               to_apply=add
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:   %[[ARG0:.*]]: tensor<42x12x8xi32>
    // CHECK-SAME:   %[[ARG1:.*]]: tensor<i32>
    // CHECK-SAME:   %[[X:arg[0-9]*]]: index {{[^}]*}}},
    // CHECK-SAME:   %[[Y:arg[0-9]*]]: index {{[^}]*}}},
    // CHECK-SAME:   %[[Z:arg[0-9]*]]: index {{[^}]*}}}) -> i32
    // CHECK:      %[[INIT:.*]] = tensor.extract %[[ARG1]][]
    // CHECK:      %[[RET:.*]] = zkx.loop
    // If symbol rescaling wasn't working we would have a
    // `floordiv` in the map:
    // CHECK-SAME:   #zkx.indexing_map<"(d0, d1, d2)[s0] -> (d0 * 2 + s0, d1, d2),
    // CHECK-SAME:     domain: d0 in [0, 18], d1 in [0, 11], d2 in [0, 7], s0 in [0, 3]">
    // CHECK-SAME:   iter_args({{.*}} = %[[INIT]])
    // CHECK:      tensor.extract %[[ARG0]]
    // CHECK:      return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Concatenate) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[10,20,30] parameter(0)
      p1 = s32[10,15,30] parameter(1)
      p2 = s32[10,3,30] parameter(2)
      ROOT r = s32[10,38,30] concatenate(p0, p1, p2), dimensions={1}
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<10x20x30xi32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<10x15x30xi32>,
    // CHECK-SAME:     %[[ARG2:.*]]: tensor<10x3x30xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK-DAG:    %[[C35:.*]] = arith.constant 35
    // CHECK-DAG:    %[[C20:.*]] = arith.constant 20
    // CHECK:        %[[IN_BOUNDS:.*]] = arith.cmpi ult, %[[Y]], %[[C20]]
    // CHECK:        %[[CONCAT:.*]] = scf.if %[[IN_BOUNDS]]
    // CHECK:          %[[P0_VAL:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:         [%[[X]], %[[Y]], %[[Z]]]
    // CHECK:          scf.yield %[[P0_VAL]]
    // CHECK:        } else {
    // CHECK:          %[[IN_BOUNDS:.*]] = arith.cmpi ult, %[[Y]], %[[C35]]
    // CHECK:          %[[CONCAT2:.*]] = scf.if %[[IN_BOUNDS]]
    // CHECK:            %[[OFFSET:.*]] = arith.subi %[[Y]], %[[C20]]
    // CHECK:            %[[P1_VAL:.*]] = tensor.extract %[[ARG1]]
    // CHECK-SAME:           [%[[X]], %[[OFFSET]], %[[Z]]]
    // CHECK:            scf.yield %[[P1_VAL]]
    // CHECK:          } else {
    // CHECK:            %[[OFFSET:.*]] = arith.subi %[[Y]], %[[C35]]
    // CHECK:            %[[P2_VAL:.*]] = tensor.extract %[[ARG2]]
    // CHECK-SAME:           [%[[X]], %[[OFFSET]], %[[Z]]]
    // CHECK:            scf.yield %[[P2_VAL]]
    // CHECK:          }
    // CHECK:          scf.yield %[[CONCAT2]]
    // CHECK:        }
    // CHECK:        return %[[CONCAT]]
  )"));
}

TEST_F(ElementalHloToMlirTest, ConcatenateMany) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[10,1,30] parameter(0)
      p1 = s32[10,2,30] parameter(1)
      p2 = s32[10,3,30] parameter(2)
      p3 = s32[10,4,30] parameter(3)
      p4 = s32[10,5,30] parameter(4)
      p5 = s32[10,6,30] parameter(5)
      p6 = s32[10,7,30] parameter(6)
      ROOT r = s32[10,28,30] concatenate(p0, p1, p2, p3, p4, p5, p6),
          dimensions={1}
    })",
                   R"(
      // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
      // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
      // CHECK-DAG:  %[[C6:.*]] = arith.constant 6 : index
      // CHECK-DAG:  %[[C10:.*]] = arith.constant 10 : index
      // CHECK-DAG:  %[[C15:.*]] = arith.constant 15 : index
      // CHECK-DAG:  %[[C21:.*]] = arith.constant 21 : index
      // CHECK:      %[[P0TO2:.*]] = arith.cmpi ult, %[[I:.*]], %[[C6]]
      // CHECK:      %[[CONCAT:.*]] = scf.if %[[P0TO2]] -> (i32)
      // CHECK-NEXT:   %[[P0:.*]] = arith.cmpi ult, %[[I]], %[[C1]]
      // CHECK-NEXT:   scf.if %[[P0]]
      // CHECK-NEXT:     tensor.extract {{.*}}[{{.*}}, %[[I]], {{.*}}] : tensor<10x1x30xi32>
      // CHECK-NEXT:     yield
      // CHECK-NEXT:   } else {
      // CHECK-NEXT:     %[[P1:.*]] = arith.cmpi ult, %[[I]], %[[C3]]
      // CHECK-NEXT:     scf.if %[[P1]]
      // CHECK-NEXT:       %[[O:.*]] = arith.subi %[[I]], %[[C1]]
      // CHECK-NEXT:       tensor.extract {{.*}}[{{.*}}, %[[O]], {{.*}}] : tensor<10x2x30xi32>
      // CHECK-NEXT:       yield
      // CHECK-NEXT:     } else {
      // CHECK-NEXT:       %[[O:.*]] = arith.subi %[[I]], %[[C3]]
      // CHECK-NEXT:       tensor.extract {{.*}}[{{.*}}, %[[O]], {{.*}}] : tensor<10x3x30xi32>
      // CHECK-NEXT:       yield
      // CHECK-NEXT:     }
      // CHECK-NEXT:     yield
      // CHECK-NEXT:   }
      // CHECK-NEXT:   yield
      // CHECK-NEXT: } else {
      // CHECK-NEXT:   %[[P3TO4:.*]] = arith.cmpi ult, %[[I]], %[[C15]]
      // CHECK-NEXT:   scf.if %[[P3TO4]]
      // CHECK-NEXT:     %[[P3:.*]] = arith.cmpi ult, %[[I]], %[[C10]]
      // CHECK-NEXT:     scf.if %[[P3]]
      // CHECK-NEXT:       %[[O:.*]] = arith.subi %[[I]], %[[C6]]
      // CHECK-NEXT:       tensor.extract {{.*}}[{{.*}}, %[[O]], {{.*}}] : tensor<10x4x30xi32>
      // CHECK-NEXT:       yield
      // CHECK-NEXT:     } else {
      // CHECK-NEXT:       %[[O:.*]] = arith.subi %[[I]], %[[C10]]
      // CHECK-NEXT:       tensor.extract {{.*}}[{{.*}}, %[[O]], {{.*}}] : tensor<10x5x30xi32>
      // CHECK-NEXT:       yield
      // CHECK-NEXT:     }
      // CHECK-NEXT:     yield
      // CHECK-NEXT:   } else {
      // CHECK-NEXT:     %[[P5:.*]] = arith.cmpi ult, %[[I]], %[[C21]]
      // CHECK-NEXT:     scf.if %[[P5]]
      // CHECK-NEXT:       %[[O:.*]] = arith.subi %[[I]], %[[C15]]
      // CHECK-NEXT:       tensor.extract {{.*}}[{{.*}}, %[[O]], {{.*}}] : tensor<10x6x30xi32>
      // CHECK-NEXT:       yield
      // CHECK-NEXT:     } else {
      // CHECK-NEXT:       %[[O:.*]] = arith.subi %[[I]], %[[C21]]
      // CHECK-NEXT:       tensor.extract {{.*}}[{{.*}}, %[[O]], {{.*}}] : tensor<10x7x30xi32>
      // CHECK-NEXT:       yield
      // CHECK-NEXT:     }
      // CHECK-NEXT:     yield
      // CHECK-NEXT:   }
      // CHECK-NEXT:   yield
      // CHECK-NEXT: }
      // CHECK-NEXT: return %[[CONCAT]]
  )"));
}

TEST_F(ElementalHloToMlirTest, ConcatenateUnsigned) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = u32[10,20,30] parameter(0)
      p1 = u32[10,15,30] parameter(1)
      ROOT r = u32[10,35,30] concatenate(p0, p1), dimensions={1}
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<10x20x30xi32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<10x15x30xi32>
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK-DAG:    %[[C20:.*]] = arith.constant 20
    // CHECK:        %[[IN_BOUNDS:.*]] = arith.cmpi ult, %[[Y]], %[[C20]]
    // CHECK:        %[[CONCAT:.*]] = scf.if %[[IN_BOUNDS]]
    // CHECK:          %[[P0_VAL:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:         [%[[X]], %[[Y]], %[[Z]]]
    // CHECK:          scf.yield %[[P0_VAL]]
    // CHECK:        } else {
    // CHECK:          %[[OFFSET:.*]] = arith.subi %[[Y]], %[[C20]]
    // CHECK:          %[[P1_VAL:.*]] = tensor.extract %[[ARG1]]
    // CHECK-SAME:         [%[[X]], %[[OFFSET]], %[[Z]]]
    // CHECK:          scf.yield %[[P1_VAL]]
    // CHECK:        }
    // CHECK:        return %[[CONCAT]]
  )"));
}

// TODO(chokobole): Enable this test. Dependency: mhlo::GatherOp
TEST_F(ElementalHloToMlirTest, DISABLED_Gather) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      operand = s32[33,34] parameter(0)
      indices = s32[1806,1] parameter(1)
      ROOT r = s32[1806,7,8] gather(operand, indices), offset_dims={1,2},
                                 collapsed_slice_dims={}, start_index_map={0},
                                 index_vector_dim=1, slice_sizes={7,8}
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<33x34xi32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<1806x1xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:    %[[C26:.*]] = arith.constant 26
    // CHECK:        %[[IDX_I32:.*]] = tensor.extract %[[ARG1]][%[[X]], %[[C0]]]
    // CHECK:        %[[IDX:.*]] = arith.index_cast %[[IDX_I32]] : i32 to index
    // CHECK:        %[[CLAMP_HIGH:.*]] = arith.minsi %[[IDX]], %[[C26]]
    // CHECK:        %[[CLAMPED:.*]] = arith.maxsi %[[CLAMP_HIGH]], %[[C0]]
    // CHECK:        %[[X_IN:.*]] = arith.addi %[[CLAMPED]], %[[Y]]
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]][%[[X_IN]], %[[Z]]]
    // CHECK:        return %[[RET]]
  )"));
}

// TODO(chokobole): Enable this test. Dependency: mhlo::GatherOp
TEST_F(ElementalHloToMlirTest, DISABLED_GatherWithImplicitVectorDim) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      operand = s32[33,34] parameter(0)
      indices = s32[1806] parameter(1)
      ROOT r = s32[1806,7,8] gather(operand, indices), offset_dims={1,2},
                                 collapsed_slice_dims={}, start_index_map={0},
                                 index_vector_dim=1, slice_sizes={7,8}
    })",
                   R"(
    // CHECK:      @main_r(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<33x34xs32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<1806xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:    %[[C26:.*]] = arith.constant 26
    // CHECK:        %[[IDX_I32:.*]] = tensor.extract %[[ARG1]][%[[X]]]
    // CHECK:        %[[IDX:.*]] = arith.index_cast %[[IDX_I32]] : i32 to index
    // CHECK:        %[[CLAMP_HIGH:.*]] = arith.minsi %[[IDX]], %[[C26]]
    // CHECK:        %[[CLAMPED:.*]] = arith.maxsi %[[CLAMP_HIGH]], %[[C0]]
    // CHECK:        %[[X_IN:.*]] = arith.addi %[[CLAMPED]], %[[Y]]
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]][%[[X_IN]], %[[Z]]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Pad) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[4, 4] parameter(0)
      p1 = s32[] parameter(1)
      ROOT pad = s32[9, 16] pad(p0, p1), padding=1_4x4_8
    })",
                   R"(
    // CHECK:      @main_pad(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4x4xi32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<i32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}}
    // CHECK-DAG:        %[[X_L:.*]] = arith.cmpi sge, %[[X]], %{{.*}}
    // CHECK-DAG:        %[[X_H:.*]] = arith.cmpi sle, %[[X]], %{{.*}}
    // CHECK:        %[[X_BOUNDS:.*]] = arith.andi %[[X_L]], %[[X_H]]
    // CHECK-DAG:        %[[Y_L:.*]] = arith.cmpi sge, %[[Y]], %{{.*}}
    // CHECK-DAG:        %[[Y_H:.*]] = arith.cmpi sle, %[[Y]], %{{.*}}
    // CHECK:        %[[Y_BOUNDS:.*]] = arith.andi %[[Y_L]], %[[Y_H]]
    // CHECK:        %[[FROM_INPUT:.*]] = arith.andi %[[X_BOUNDS]], %[[Y_BOUNDS]]
    // CHECK:        %[[RET:.*]] = scf.if %[[FROM_INPUT]]
    // CHECK:          %[[IN0:.*]] = zkx.apply_indexing
    // CHECK:          %[[IN1:.*]] = zkx.apply_indexing
    // CHECK:          %[[VAL:.*]] = tensor.extract %[[ARG0]][%[[IN0]], %[[IN1]]]
    // CHECK:          scf.yield %[[VAL]]
    // CHECK:        } else {
    // CHECK:          %[[PAD_VAL:.*]] = tensor.extract %[[ARG1]][]
    // CHECK:          scf.yield %[[PAD_VAL]]
    // CHECK:        }
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, PadUnsigned) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = u32[4, 4] parameter(0)
      p1 = u32[] parameter(1)
      ROOT pad = u32[9, 16] pad(p0, p1), padding=1_4x4_8
    })",
                   R"(
    // CHECK:      @main_pad(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4x4xi32>,
    // CHECK-SAME:     %[[ARG1:.*]]: tensor<i32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}}
    // CHECK-DAG:        %[[X_L:.*]] = arith.cmpi sge, %[[X]], %{{.*}}
    // CHECK-DAG:        %[[X_H:.*]] = arith.cmpi sle, %[[X]], %{{.*}}
    // CHECK:        %[[X_BOUNDS:.*]] = arith.andi %[[X_L]], %[[X_H]]
    // CHECK-DAG:        %[[Y_L:.*]] = arith.cmpi sge, %[[Y]], %{{.*}}
    // CHECK-DAG:        %[[Y_H:.*]] = arith.cmpi sle, %[[Y]], %{{.*}}
    // CHECK:        %[[Y_BOUNDS:.*]] = arith.andi %[[Y_L]], %[[Y_H]]
    // CHECK:        %[[FROM_INPUT:.*]] = arith.andi %[[X_BOUNDS]], %[[Y_BOUNDS]]
    // CHECK:        %[[RET:.*]] = scf.if %[[FROM_INPUT]]
    // CHECK:          %[[IN0:.*]] = zkx.apply_indexing
    // CHECK:          %[[IN1:.*]] = zkx.apply_indexing
    // CHECK:          %[[VAL:.*]] = tensor.extract %[[ARG0]][%[[IN0]], %[[IN1]]]
    // CHECK:          scf.yield %[[VAL]]
    // CHECK:        } else {
    // CHECK:          %[[PAD_VAL:.*]] = tensor.extract %[[ARG1]][]
    // CHECK:          scf.yield %[[PAD_VAL]]
    // CHECK:        }
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, DotWithS32Type) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[3, 4] parameter(0)
      p1 = s32[4, 5] parameter(1)
      ROOT dot = s32[3, 5] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })",
                   R"(
    // CHECK:      @main_dot(
    // CHECK-SAME: %[[A:.*]]: tensor<3x4xi32>, %[[B:.*]]: tensor<4x5xi32>,
    // CHECK-SAME: %[[I:.*]]: index {zkx.range = [0 : index, 2 : index]},
    // CHECK-SAME: %[[J:.*]]: index {zkx.range = [0 : index, 4 : index]})
    // CHECK-SAME: -> i32
    // CHECK-SAME: {
    // CHECK-DAG:    %[[ACCUM_INIT:.*]] = arith.constant 0 : i32
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
    // CHECK:        %[[FOR0:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
    // CHECK-SAME:   iter_args(%[[ACCUM:.*]] = %[[ACCUM_INIT]]) -> (i32) {
    // CHECK-DAG:      %[[CMPI0:.*]] = arith.cmpi sge, %[[I]], %[[C0]] : index
    // CHECK-DAG:      %[[CMPI1:.*]] = arith.cmpi sle, %[[I]], %[[C2]] : index
    // CHECK-DAG:      %[[I_IN_RANGE:.*]] = arith.andi %[[CMPI0]], %[[CMPI1]] : i1
    // CHECK-DAG:      %[[CMPI2:.*]] = arith.cmpi sge, %[[J]], %[[C0]] : index
    // CHECK-DAG:      %[[CMPI3:.*]] = arith.cmpi sle, %[[J]], %[[C4]] : index
    // CHECK-DAG:      %[[J_IN_RANGE:.*]] = arith.andi %[[CMPI2]], %[[CMPI3]] : i1
    // CHECK-DAG:      %[[I_J_IN_RANGE:.*]] = arith.andi %[[I_IN_RANGE]], %[[J_IN_RANGE]] : i1
    // CHECK:          %[[IF0:.*]] = scf.if %[[I_J_IN_RANGE]] -> (i32) {
    // CHECK-DAG:        %[[A_I_K:.*]] = tensor.extract %[[A]][%[[I]], %[[K]]] : tensor<3x4xi32>
    // CHECK-DAG:        %[[B_K_J:.*]] = tensor.extract %[[B]][%[[K]], %[[J]]] : tensor<4x5xi32>
    // CHECK-DAG:        %[[MUL0:.*]] = arith.muli %[[A_I_K]], %[[B_K_J]] : i32
    // CHECK-DAG:        %[[ADD0:.*]] = arith.addi %[[ACCUM]], %[[MUL0]] : i32
    // CHECK-DAG:        scf.yield %[[ADD0]] : i32
    // CHECK:          } else {
    // CHECK:            scf.yield %[[ACCUM]] : i32
    // CHECK:          }
    // CHECK:          scf.yield %[[IF0]] : i32
    // CHECK:        }
    // CHECK:        return %[[FOR0]] : i32
    // CHECK:      }
  )"));
}

TEST_F(ElementalHloToMlirTest, DotWithU32Type) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = u32[3, 4] parameter(0)
      p1 = u32[4, 5] parameter(1)
      ROOT dot = u32[3, 5] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })",
                   R"(
    // CHECK:      @main_dot(
    // CHECK-SAME: %[[A:.*]]: tensor<3x4xi32>, %[[B:.*]]: tensor<4x5xi32>,
    // CHECK-SAME: %[[I:.*]]: index {zkx.range = [0 : index, 2 : index]},
    // CHECK-SAME: %[[J:.*]]: index {zkx.range = [0 : index, 4 : index]})
    // CHECK-SAME: -> i32
    // CHECK-SAME: {
    // CHECK-DAG:    %[[ACCUM_INIT:.*]] = arith.constant 0 : i32
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
    // CHECK:        %[[FOR0:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
    // CHECK-SAME:   iter_args(%[[ACCUM:.*]] = %[[ACCUM_INIT]]) -> (i32) {
    // CHECK-DAG:      %[[CMPI0:.*]] = arith.cmpi sge, %[[I]], %[[C0]] : index
    // CHECK-DAG:      %[[CMPI1:.*]] = arith.cmpi sle, %[[I]], %[[C2]] : index
    // CHECK-DAG:      %[[I_IN_RANGE:.*]] = arith.andi %[[CMPI0]], %[[CMPI1]] : i1
    // CHECK-DAG:      %[[CMPI2:.*]] = arith.cmpi sge, %[[J]], %[[C0]] : index
    // CHECK-DAG:      %[[CMPI3:.*]] = arith.cmpi sle, %[[J]], %[[C4]] : index
    // CHECK-DAG:      %[[J_IN_RANGE:.*]] = arith.andi %[[CMPI2]], %[[CMPI3]] : i1
    // CHECK-DAG:      %[[I_J_IN_RANGE:.*]] = arith.andi %[[I_IN_RANGE]], %[[J_IN_RANGE]] : i1
    // CHECK:          %[[IF0:.*]] = scf.if %[[I_J_IN_RANGE]] -> (i32) {
    // CHECK-DAG:        %[[A_I_K:.*]] = tensor.extract %[[A]][%[[I]], %[[K]]] : tensor<3x4xi32>
    // CHECK-DAG:        %[[B_K_J:.*]] = tensor.extract %[[B]][%[[K]], %[[J]]] : tensor<4x5xi32>
    // CHECK-DAG:        %[[MUL0:.*]] = arith.muli %[[A_I_K]], %[[B_K_J]] : i32
    // CHECK-DAG:        %[[ADD0:.*]] = arith.addi %[[ACCUM]], %[[MUL0]] : i32
    // CHECK-DAG:        scf.yield %[[ADD0]] : i32
    // CHECK:          } else {
    // CHECK:            scf.yield %[[ACCUM]] : i32
    // CHECK:          }
    // CHECK:          scf.yield %[[IF0]] : i32
    // CHECK:        }
    // CHECK:        return %[[FOR0]] : i32
    // CHECK:      }
  )"));
}

TEST_F(ElementalHloToMlirTest, DotWithPredType) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = pred[3, 4] parameter(0)
      p1 = pred[4, 5] parameter(1)
      ROOT dot = pred[3, 5] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })",
                   R"(
    // CHECK:      @main_dot(
    // CHECK-SAME: %[[A:.*]]: tensor<3x4xi1>, %[[B:.*]]: tensor<4x5xi1>,
    // CHECK-SAME: %[[I:.*]]: index {zkx.range = [0 : index, 2 : index]},
    // CHECK-SAME: %[[J:.*]]: index {zkx.range = [0 : index, 4 : index]})
    // CHECK-SAME: -> i1
    // CHECK-SAME: {
    // CHECK-DAG:    %[[ACCUM_INIT:.*]] = arith.constant false
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
    // CHECK:        %[[FOR0:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
    // CHECK-SAME:   iter_args(%[[ACCUM:.*]] = %[[ACCUM_INIT]]) -> (i1) {
    // CHECK-DAG:      %[[CMPI0:.*]] = arith.cmpi sge, %[[I]], %[[C0]] : index
    // CHECK-DAG:      %[[CMPI1:.*]] = arith.cmpi sle, %[[I]], %[[C2]] : index
    // CHECK-DAG:      %[[I_IN_RANGE:.*]] = arith.andi %[[CMPI0]], %[[CMPI1]] : i1
    // CHECK-DAG:      %[[CMPI2:.*]] = arith.cmpi sge, %[[J]], %[[C0]] : index
    // CHECK-DAG:      %[[CMPI3:.*]] = arith.cmpi sle, %[[J]], %[[C4]] : index
    // CHECK-DAG:      %[[J_IN_RANGE:.*]] = arith.andi %[[CMPI2]], %[[CMPI3]] : i1
    // CHECK-DAG:      %[[I_J_IN_RANGE:.*]] = arith.andi %[[I_IN_RANGE]], %[[J_IN_RANGE]] : i1
    // CHECK:          %[[IF0:.*]] = scf.if %[[I_J_IN_RANGE]] -> (i1) {
    // CHECK-DAG:        %[[A_I_K:.*]] = tensor.extract %[[A]][%[[I]], %[[K]]] : tensor<3x4xi1>
    // CHECK-DAG:        %[[B_K_J:.*]] = tensor.extract %[[B]][%[[K]], %[[J]]] : tensor<4x5xi1>
    // CHECK-DAG:        %[[AND0:.*]] = arith.andi %[[A_I_K]], %[[B_K_J]] : i1
    // CHECK-DAG:        %[[OR0:.*]] = arith.ori %[[ACCUM]], %[[AND0]] : i1
    // CHECK-DAG:        scf.yield %[[OR0]] : i1
    // CHECK:          } else {
    // CHECK:            scf.yield %[[ACCUM]] : i1
    // CHECK:          }
    // CHECK:          scf.yield %[[IF0]] : i1
    // CHECK:        }
    // CHECK:        return %[[FOR0]] : i1
    // CHECK:      }
  )"));
}

TEST_F(ElementalHloToMlirTest, DotWithBatchAnd2ContractingDims) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[7, 3, 4, 5] parameter(0)
      p1 = s32[5, 6, 4, 7] parameter(1)
      ROOT dot = s32[7, 3, 6] dot(p0, p1),
                 lhs_contracting_dims={2, 3}, rhs_contracting_dims={2, 0},
                 lhs_batch_dims={0}, rhs_batch_dims={3}
    })",
                   R"(
    // CHECK:      @main_dot(
    // CHECK-SAME: %[[A:.*]]: tensor<7x3x4x5xi32>, %[[B:.*]]: tensor<5x6x4x7xi32>,
    // CHECK-SAME: %[[N:.*]]: index {zkx.range = [0 : index, 6 : index]},
    // CHECK-SAME: %[[I:.*]]: index {zkx.range = [0 : index, 2 : index]},
    // CHECK-SAME: %[[J:.*]]: index {zkx.range = [0 : index, 5 : index]})
    // CHECK-SAME: -> i32
    // CHECK-SAME: {
    // CHECK-DAG:    %[[C0F:.*]] = arith.constant 0 : i32
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
    // CHECK-DAG:    %[[C5:.*]] = arith.constant 5 : index
    // CHECK-DAG:    %[[C6:.*]] = arith.constant 6 : index
    // CHECK:        %[[FOR0:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
    // CHECK-SAME:   iter_args(%[[ACCUM0:.*]] = %[[C0F]]) -> (i32) {
    // CHECK:          %[[FOR1:.*]] = scf.for %[[L:.*]] = %[[C0]] to %[[C5]] step %[[C1]]
    // CHECK-SAME:     iter_args(%[[ACCUM1:.*]] = %[[ACCUM0]]) -> (i32) {
    // CHECK-DAG:        %[[CMPI0:.*]] = arith.cmpi sge, %[[N]], %[[C0]] : index
    // CHECK-DAG:        %[[CMPI1:.*]] = arith.cmpi sle, %[[N]], %[[C6]] : index
    // CHECK-DAG:        %[[N_IN_RANGE:.*]] = arith.andi %[[CMPI0]], %[[CMPI1]] : i1
    // CHECK-DAG:        %[[CMPI2:.*]] = arith.cmpi sge, %[[I]], %[[C0]] : index
    // CHECK-DAG:        %[[CMPI3:.*]] = arith.cmpi sle, %[[I]], %[[C2]] : index
    // CHECK-DAG:        %[[I_IN_RANGE:.*]] = arith.andi %[[CMPI2]], %[[CMPI3]] : i1
    // CHECK-DAG:        %[[N_I_IN_RANGE:.*]] = arith.andi %[[N_IN_RANGE]], %[[I_IN_RANGE]] : i1
    // CHECK-DAG:        %[[CMPI4:.*]] = arith.cmpi sge, %[[J]], %[[C0]] : index
    // CHECK-DAG:        %[[CMPI5:.*]] = arith.cmpi sle, %[[J]], %[[C5]] : index
    // CHECK-DAG:        %[[J_IN_RANGE:.*]] = arith.andi %[[CMPI4]], %[[CMPI5]] : i1
    // CHECK-DAG:        %[[N_I_J_IN_RANGE:.*]] = arith.andi %[[N_I_IN_RANGE]], %[[J_IN_RANGE]] : i1
    // CHECK:            %[[IF0:.*]] = scf.if %[[N_I_J_IN_RANGE]] -> (i32) {
    // CHECK-DAG:          %[[A_N_I_K_L:.*]] = tensor.extract %[[A]][%[[N]], %[[I]], %[[K]], %[[L]]] : tensor<7x3x4x5xi32>
    // CHECK-DAG:          %[[B_L_J_K_N:.*]] = tensor.extract %[[B]][%[[L]], %[[J]], %[[K]], %[[N]]] : tensor<5x6x4x7xi32>
    // CHECK-DAG:          %[[MULF0:.*]] = arith.muli %[[A_N_I_K_L]], %[[B_L_J_K_N]] : i32
    // CHECK-DAG:          %[[ADDF0:.*]] = arith.addi %[[ACCUM1]], %[[MULF0]] : i32
    // CHECK-DAG:          scf.yield %[[ADDF0]] : i32
    // CHECK:            } else {
    // CHECK:              scf.yield %[[ACCUM1]] : i32
    // CHECK:            }
    // CHECK:            scf.yield %[[IF0]] : i32
    // CHECK:          }
    // CHECK:          scf.yield %[[FOR1]] : i32
    // CHECK:        }
    // CHECK:        return %[[FOR0]] : i32
    // CHECK:      }
  )"));
}

TEST_F(ElementalHloToMlirTest, Transpose) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[4,5,6] parameter(0)
      ROOT transpose = s32[6,5,4] transpose(p0), dimensions={2,1,0}
    })",
                   R"(
    // CHECK:      @main_transpose(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4x5x6xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:     [%[[Z]], %[[Y]], %[[X]]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Broadcast) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[4,5] parameter(0)
      ROOT broadcast = s32[6,4,5] broadcast(p0), dimensions={1,2}
    })",
                   R"(
    // CHECK:      @main_broadcast(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4x5xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}},
    // CHECK-SAME:     %[[Z:.*]]: index {{{.*}}}
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]]
    // CHECK-SAME:     [%[[Y]], %[[Z]]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Add) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[4] parameter(0)
      p1 = s32[4] parameter(1)
      ROOT add = s32[4] add(p0, p1)
    })",
                   R"(
    // CHECK:      @main_add(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4xi32>, %[[ARG1:.*]]: tensor<4xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{.*}}
    // CHECK:        %[[A:.*]] = tensor.extract %[[ARG0]][%[[X]]]
    // CHECK:        %[[B:.*]] = tensor.extract %[[ARG1]][%[[X]]]
    // CHECK:        %[[RET:.*]] = arith.addi %[[A]], %[[B]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, UnsignedDiv) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = u32[4] parameter(0)
      p1 = u32[4] parameter(1)
      ROOT div = u32[4] divide(p0, p1)
    })",
                   R"(
    // CHECK:      @main_div(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<4xi32>, %[[ARG1:.*]]: tensor<4xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{.*}}
    // CHECK:        %[[DIV:.*]] = arith.divui %{{.*}}, %{{.*}} : i32
  )"));
}

// TODO(chokobole): Enable this test.
// Error message: error: type of return operand 0 ('i8') doesn't match function
// result type ('i1') in function @main_convert
TEST_F(ElementalHloToMlirTest, DISABLED_ConvertS8ToPred) {
  // Both s8 and pred are represented as i8, but a conversion is still needed.
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s8[4] parameter(0)
      ROOT convert = pred[4] convert(p0)
    })",
                   R"(
    // CHECK:      @main_convert(
    // CHECK:       %[[C0:.*]] = arith.constant 0 : i8
    // CHECK:       %[[CMP:.*]] = arith.cmpi ne, %{{.*}}, %[[C0]] : i8
    // CHECK:       %[[RET:.*]] = arith.extui %[[CMP]] : i1 to i8
    // CHECK:       return %[[RET]] : i8
  )"));
}

// TODO(batzor): Hangs during MLIR emission — needs investigation.
TEST_F(ElementalHloToMlirTest, DISABLED_PopulationCountUnsigned) {
  TF_EXPECT_OK(Run(R"(
     ENTRY main{
       p0 = u32[10,1,4]{2,1,0} parameter(0)
       ROOT popcnt = u32[10,1,4]{2,1,0} popcnt(p0)
     })",
                   R"(
    // CHECK:      @main_popcnt(
    // CHECK:        math.ctpop %{{.*}} : i32
  )"));
}

class ElementalHloToMlirEpilogueTest : public ElementalHloToMlirTest {
 protected:
  std::function<EpilogueSpecification(HloComputation* entry)> EpilogueSpec() {
    return [this](HloComputation* entry) {
      EpilogueSpecification epilogue;
      epilogue.heroes.push_back(entry->GetInstructionWithName("transpose"));
      epilogue.roots.push_back(entry->GetInstructionWithName("add"));
      epilogue.index_ranges = {2, 16, 17};
      epilogue.root_indexing.push_back(
          IndexingMap{mlir::AffineMap::getMultiDimIdentityMap(3, &context_)
                          .getSubMap({0, 2, 1}),
                      DimVarsFromTensorSizes({2, 17, 17}),
                      {},
                      {}});
      return epilogue;
    };
  }
  static constexpr std::string_view kHlo =
      R"(
      ENTRY main {
        // Note: %p0 is only used in some of the tests.
        %p0 = s32[7] parameter(0)
        %p1 = s32[2,16,17] parameter(1)
        %neg = s32[2,16,17] negate(%p1)
        %transpose = s32[2,17,16] transpose(%neg), dimensions={0,2,1}
        %p2 = s32[] parameter(2)
        %bc = s32[2,17,16] broadcast(%p2), dimensions={}
        ROOT %add = s32[2,17,16] add(%transpose, %bc)
      })";
  static constexpr std::string_view kCheck =
      R"(
      // CHECK:      @main_add(
      // CHECK-SAME:     %[[A0:.*]]: tensor<7xi32>
      // CHECK:        %[[PURE:.*]] = zkx.pure_call @main_transpose(%[[A0]],
      // CHECK:      @main_transpose(tensor<7xi32>,
      // CHECK:      @main__epilogue__add(
      // CHECK-SAME:     %[[ARG0:.*]]: tensor<7xi32>
      // CHECK-SAME:     %[[ARG1:.*]]: tensor<2x16x17xi32>
      // CHECK-SAME:     %[[ARG2:.*]]: tensor<i32>
      // CHECK-SAME:     %[[X:.*]]: index {zkx.range = [0 : index, 1 :
      // CHECK-SAME:     %[[Y:.*]]: index {zkx.range = [0 : index, 15 :
      // CHECK-SAME:     %[[Z:.*]]: index {zkx.range = [0 : index, 16 :
      // CHECK-SAME:     %[[TRANSPOSE:.*]]: i32) -> i32
      // CHECK:        %[[B:.*]] = tensor.extract %[[ARG2]][]
      // CHECK:        %[[RET:.*]] = arith.addi %[[TRANSPOSE]], %[[B]]
      // CHECK:        return %[[RET]]
      )";
};

TEST_F(ElementalHloToMlirEpilogueTest, Epilogue) {
  TF_EXPECT_OK(Run(kHlo, kCheck, EpilogueSpec()));
}

TEST_F(ElementalHloToMlirEpilogueTest, ZkxEntry) {
  TF_EXPECT_OK(Run(kHlo, kCheck, EpilogueSpec(), /*set_zkx_entry=*/true));
}

TEST_F(ElementalHloToMlirEpilogueTest, ZkxGpuEntry) {
  TF_EXPECT_OK(Run(kHlo, kCheck, EpilogueSpec(), /*set_zkx_entry=*/true,
                   /*zkx_backend=*/BackendKind::kGpu));
}

TEST_F(ElementalHloToMlirEpilogueTest, ZkxCpuEntry) {
  TF_EXPECT_OK(Run(kHlo,
                   R"(
      // CHECK:      @main_add(
      // CHECK-SAME:     %[[ARG0:.*]]: tensor<7xi32>
      // main_transpose must still have arg0, but the pure_call must not.
      // CHECK:          %[[PURE:.*]] = zkx.pure_call @main_transpose(%arg1,
      // CHECK:      @main_transpose(tensor<7xi32)",
                   EpilogueSpec(), /*set_zkx_entry=*/true,
                   /*zkx_backend=*/BackendKind::kCpu));
}

TEST_F(ElementalHloToMlirTest, ScalarConstant) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[1,1] parameter(0)
      c1 = s32[1,1] constant({{1}})
      ROOT add = s32[1,1] add(p0, c1)
    })",
                   R"(
    // CHECK:      @main_add(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<1x1xi32>
    // CHECK-SAME:     %[[X:.*]]: index {{.*}}, %[[Y:.*]]: index {{.*}}
    // CHECK:        %[[C_1:.*]] = arith.constant 1
    // CHECK:        %[[A:.*]] = tensor.extract %[[ARG0]][%[[X]], %[[Y]]]
    // CHECK:        %[[RET:.*]] = arith.addi %[[A]], %[[C_1]]
    // CHECK:        return %[[RET]]
  })"));
}

TEST_F(ElementalHloToMlirTest, ScalarUnsignedConstant) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = u32[1,1] parameter(0)
      c1 = u32[1,1] constant({{1}})
      ROOT add = u32[1,1] add(p0, c1)
    })",
                   R"(
    // CHECK:      @main_add(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<1x1xi32>
    // CHECK-SAME:     %[[X:.*]]: index {{.*}}, %[[Y:.*]]: index {{.*}}
    // CHECK:        %[[C_1:.*]] = arith.constant 1
    // CHECK:        %[[A:.*]] = tensor.extract %[[ARG0]][%[[X]], %[[Y]]]
    // CHECK:        %[[RET:.*]] = arith.addi %[[A]], %[[C_1]]
    // CHECK:        return %[[RET]]
  })"));
}

TEST_F(ElementalHloToMlirTest, TensorConstant) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = s32[2,1] parameter(0)
      c1 = s32[2,1] constant({{1}, {2}})
      ROOT add = s32[2,1] add(p0, c1)
    })",
                   R"(
    // CHECK:      @main_add(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<2x1xi32>
    // CHECK-SAME:     %[[X:.*]]: index {{.*}}, %[[Y:.*]]: index {{.*}}
    // CHECK:        %[[C_1:.*]] = arith.constant dense<[
    // CHECK-SAME:       [1], [2]]>
    // CHECK:        %[[A:.*]] = tensor.extract %[[ARG0]][%[[X]], %[[Y]]]
    // CHECK:        %[[B:.*]] = tensor.extract %[[C_1]][%[[X]], %[[Y]]]
    // CHECK:        %[[RET:.*]] = arith.addi %[[A]], %[[B]]
    // CHECK:        return %[[RET]]
  })"));
}

// TODO(chokobole): Enable this test.
// Error message: emitters_unittests:
// external/llvm-project/mlir/lib/IR/BuiltinAttributes.cpp:1484: auto
// mappingHelper(llvm::function_ref<llvm::APInt (const llvm::APInt &)>, const
// mlir::DenseIntElementsAttr &, ShapedType, Type, llvm::SmallVectorImpl<char>
// &)::(anonymous class)::operator()(decltype(* attr.begin()), size_t) const:
// Assertion `newInt.getBitWidth() == bitWidth' failed.
TEST_F(ElementalHloToMlirTest, DISABLED_TensorConstantPred) {
  TF_EXPECT_OK(Run(
      R"(
    ENTRY main {
      ROOT c1 = pred[2] constant({1, 0})
    })",
      "// CHECK: arith.constant dense<[1, 0]> : tensor<2xi8>"));
}

TEST_F(ElementalHloToMlirTest, DynamicSlice) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      in = s32[20,30] parameter(0)
      i0 = s32[] parameter(1)
      i1 = s32[] parameter(2)
      ROOT slice = s32[4,5] dynamic-slice(in, i0, i1), dynamic_slice_sizes={4,5}
    })",
                   R"(
    // CHECK:      @main_slice(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<20x30xi32>,
    // CHECK-SAME:     %[[I0_T:.*]]: tensor<i32>, %[[I1_T:.*]]: tensor<i32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:    %[[C16:.*]] = arith.constant 16
    // CHECK-DAG:    %[[C25:.*]] = arith.constant 25
    // CHECK:        %[[I0:.*]] = tensor.extract %[[I0_T]]
    // CHECK:        %[[I0_1:.*]] = arith.index_cast %[[I0]]
    // CHECK:        %[[I0_2:.*]] = arith.minsi %[[I0_1]], %[[C16]]
    // CHECK:        %[[I0_3:.*]] = arith.maxsi %[[I0_2]], %[[C0]]
    // CHECK:        %[[X_IN:.*]] = arith.addi %[[X]], %[[I0_3]]
    // CHECK:        %[[I1:.*]] = tensor.extract %[[I1_T]]
    // CHECK:        %[[I1_1:.*]] = arith.index_cast %[[I1]]
    // CHECK:        %[[I1_2:.*]] = arith.minsi %[[I1_1]], %[[C25]]
    // CHECK:        %[[I1_3:.*]] = arith.maxsi %[[I1_2]], %[[C0]]
    // CHECK:        %[[Y_IN:.*]] = arith.addi %[[Y]], %[[I1_3]]
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]][%[[X_IN]], %[[Y_IN]]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, DynamicSliceUnsignedIndices) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      in = s32[20,30] parameter(0)
      i0 = u32[] parameter(1)
      i1 = u32[] parameter(2)
      ROOT slice = s32[4,5] dynamic-slice(in, i0, i1), dynamic_slice_sizes={4,5}
    })",
                   R"(
    // CHECK:      @main_slice(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<20x30xi32>,
    // CHECK-SAME:     %[[I0_T:.*]]: tensor<i32>, %[[I1_T:.*]]: tensor<i32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {
    // CHECK-DAG:    %[[C16:.*]] = arith.constant 16
    // CHECK-DAG:    %[[C25:.*]] = arith.constant 25
    // CHECK:        %[[I0:.*]] = tensor.extract %[[I0_T]]
    // CHECK:        %[[I0_1:.*]] = arith.index_castui %[[I0]]
    // CHECK:        %[[I0_2:.*]] = arith.minui %[[I0_1]], %[[C16]]
    // CHECK:        %[[X_IN:.*]] = arith.addi %[[X]], %[[I0_2]]
    // CHECK:        %[[I1:.*]] = tensor.extract %[[I1_T]]
    // CHECK:        %[[I1_1:.*]] = arith.index_castui %[[I1]]
    // CHECK:        %[[I1_2:.*]] = arith.minui %[[I1_1]], %[[C25]]
    // CHECK:        %[[Y_IN:.*]] = arith.addi %[[Y]], %[[I1_2]]
    // CHECK:        %[[RET:.*]] = tensor.extract %[[ARG0]][%[[X_IN]], %[[Y_IN]]]
    // CHECK:        return %[[RET]]
  )"));
}

TEST_F(ElementalHloToMlirTest, DynamicSliceIndexIsNotCanonical_NotSupported) {
  auto status = Run(R"(
    ENTRY main {
      in = s32[20,30] parameter(0)
      idx = s32[2] parameter(1)
      ROOT slice = s32[4,5] dynamic-slice(in, idx), dynamic_slice_sizes={4,5}
    })",
                    "");

  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              HasSubstr("Dynamic indexing instruction with non-scalar index is "
                        "not supported."));
}

TEST_F(ElementalHloToMlirTest, DynamicUpdateSlice) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      in = s32[20,30] parameter(0)
      updates = s32[5,6] parameter(1)
      i0 = s32[] parameter(2)
      i1 = s32[] parameter(3)
      ROOT updated = s32[20,30] dynamic-update-slice(in, updates, i0, i1)
    })",
                   R"(
    // CHECK:      @main_updated(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<20x30xi32>, %[[ARG1:.*]]: tensor<5x6xi32>
    // CHECK-SAME:     %[[I0_T:.*]]: tensor<i32>, %[[I1_T:.*]]: tensor<i32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:    %[[C5:.*]] = arith.constant 5
    // CHECK-DAG:    %[[C6:.*]] = arith.constant 6
    // CHECK-DAG:    %[[C15:.*]] = arith.constant 15
    // CHECK-DAG:    %[[C24:.*]] = arith.constant 24
    // CHECK:        %[[I0:.*]] = tensor.extract %[[I0_T]]
    // CHECK:        %[[I0_1:.*]] = arith.index_cast %[[I0]]
    // CHECK:        %[[I0_2:.*]] = arith.minsi %[[I0_1]], %[[C15]]
    // CHECK:        %[[START_X:.*]] = arith.maxsi %[[I0_2]], %[[C0]]
    // CHECK:        %[[END_X:.*]] = arith.addi %[[START_X]], %[[C5]]
    // CHECK:        %[[LOW_X:.*]] = arith.cmpi sge, %[[X]], %[[START_X]]
    // CHECK:        %[[HIGH_X:.*]] = arith.cmpi slt, %[[X]], %[[END_X]]
    // CHECK:        %[[BOUNDS_X:.*]] = arith.andi %[[LOW_X]], %[[HIGH_X]]
    // CHECK:        %[[UPDATES_X:.*]] = arith.subi %[[X]], %[[START_X]]
    // CHECK:        arith.andi
    // CHECK:        %[[BOUNDS:.*]] = arith.andi
    // CHECK:        scf.if %[[BOUNDS]]
    // CHECK:          tensor.extract %[[ARG1]][%[[UPDATES_X]]
    // CHECK:        } else {
    // CHECK:          tensor.extract %[[ARG0]][%[[X]]
  )"));
}

TEST_F(ElementalHloToMlirTest, DynamicUpdateSliceUnsigned) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      in = u32[20,30] parameter(0)
      updates = u32[5,6] parameter(1)
      i0 = s32[] parameter(2)
      i1 = s32[] parameter(3)
      ROOT updated = u32[20,30] dynamic-update-slice(in, updates, i0, i1)
    })",
                   R"(
    // CHECK:      @main_updated(
    // CHECK-SAME:     %[[ARG0:.*]]: tensor<20x30xi32>, %[[ARG1:.*]]: tensor<5x6xi32>
    // CHECK-SAME:     %[[I0_T:.*]]: tensor<i32>, %[[I1_T:.*]]: tensor<i32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {
    // CHECK-DAG:    %[[C0:.*]] = arith.constant 0
    // CHECK-DAG:    %[[C5:.*]] = arith.constant 5
    // CHECK-DAG:    %[[C6:.*]] = arith.constant 6
    // CHECK-DAG:    %[[C15:.*]] = arith.constant 15
    // CHECK-DAG:    %[[C24:.*]] = arith.constant 24
    // CHECK:        %[[I0:.*]] = tensor.extract %[[I0_T]]
    // CHECK:        %[[I0_1:.*]] = arith.index_cast %[[I0]]
    // CHECK:        %[[I0_2:.*]] = arith.minsi %[[I0_1]], %[[C15]]
    // CHECK:        %[[START_X:.*]] = arith.maxsi %[[I0_2]], %[[C0]]
    // CHECK:        %[[END_X:.*]] = arith.addi %[[START_X]], %[[C5]]
    // CHECK:        %[[LOW_X:.*]] = arith.cmpi sge, %[[X]], %[[START_X]]
    // CHECK:        %[[HIGH_X:.*]] = arith.cmpi slt, %[[X]], %[[END_X]]
    // CHECK:        %[[BOUNDS_X:.*]] = arith.andi %[[LOW_X]], %[[HIGH_X]]
    // CHECK:        %[[UPDATES_X:.*]] = arith.subi %[[X]], %[[START_X]]
    // CHECK:        arith.andi
    // CHECK:        %[[BOUNDS:.*]] = arith.andi
    // CHECK:        scf.if %[[BOUNDS]]
    // CHECK:          %[[VAL0:.*]] = tensor.extract %[[ARG1]][%[[UPDATES_X]]
    // CHECK:        } else {
    // CHECK:          %[[VAL1:.*]] = tensor.extract %[[ARG0]][%[[X]]
  )"));
}

TEST_F(ElementalHloToMlirTest,
       DynamicUpdateSliceIndexIsNotCanonical_NotSupported) {
  auto status = Run(R"(
    ENTRY main {
      in = s32[20,30] parameter(0)
      updates = s32[5,6] parameter(1)
      idx = s32[2] parameter(2)
      ROOT updated = s32[20,30] dynamic-update-slice(in, updates, idx)
    })",
                    "");

  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              HasSubstr("Dynamic indexing instruction with non-scalar index is "
                        "not supported."));
}

TEST_F(ElementalHloToMlirTest, IotaUnsigned) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      ROOT iota = u32[10,20] iota(), iota_dimension=0
    })",
                   R"(
    // CHECK:      @main_iota(
    // CHECK-SAME:     %[[I0:.*]]: index {{.*}}, %[[I1:.*]]: index {{.*}} {
    // CHECK:        %[[VAL:.*]] = arith.index_castui %[[I0]] : index to i32
  )"));
}

TEST_F(ElementalHloToMlirTest, MixedIndexingTuple) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      %p0 = s32[10,10] parameter(0)
      %p1 = s32[100] parameter(1)
      ROOT tuple = (s32[10,10], s32[100]) tuple(%p0, %p1)
    })",
                   R"(
    // CHECK:      @main_tuple(
    // CHECK-SAME:     %[[P0:.*]]: tensor<10x10xi32>,
    // CHECK-SAME:     %[[P1:.*]]: tensor<100xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}}
    // CHECK:        %[[A:.*]] = tensor.extract %[[P0]][%[[X]], %[[Y]]]
    // CHECK:        %[[IDX:.*]] = zkx.apply_indexing
    // CHECK-SAME:       #zkx.indexing_map<"(d0, d1) -> (d0 * 10 + d1),
    // CHECK-SAME:       d0 in [0, 9], d1 in [0, 9]">(%[[X]], %[[Y]])
    // CHECK:        %[[B:.*]] = tensor.extract %[[P1]][%[[IDX]]]
    // CHECK:        return %[[A]], %[[B]]
  )"));
}

TEST_F(ElementalHloToMlirTest, NestedTuple) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      %p0 = s32[10,10] parameter(0)
      %p1 = s32[100] parameter(1)
      %t0 = (s32[10,10], s32[100]) tuple(%p0, %p1)
      %t1 = (s32[100], s32[10,10]) tuple(%p1, %p0)
      ROOT tuple = ((s32[10,10], s32[100]), s32[100], (s32[100], s32[10,10]))
        tuple(%t0, %p1, %t1)
    })",
                   R"(
    // CHECK:      @main_tuple(
    // CHECK-SAME:     %[[P0:.*]]: tensor<10x10xi32>,
    // CHECK-SAME:     %[[P1:.*]]: tensor<100xi32>,
    // CHECK-SAME:     %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}}
    // CHECK:          %[[P0_V:.*]] = zkx.pure_call @main_p0
    // CHECK:          %[[IDX:.*]] =
    // CHECK-SAME:       #zkx.indexing_map<"(d0, d1) -> (d0 * 10 + d1),
    // CHECK-SAME:       d0 in [0, 9], d1 in [0, 9]">(%[[X]], %[[Y]])
    // CHECK:          %[[P1_V:.*]] = zkx.pure_call @main_p1
    // CHECK-SAME:       (%[[P0]], %[[P1]], %[[IDX]])
    // CHECK:          return %[[P0_V]], %[[P1_V]], %[[P1_V]], %[[P1_V]], %[[P0_V]]
  )"));
}

TEST_F(ElementalHloToMlirTest, Map) {
  TF_EXPECT_OK(Run(R"(
    mapper {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }
    ENTRY main {
      %p0 = s32[5,7] parameter(0)
      %p1 = s32[5,7] parameter(1)
      ROOT r = s32[5,7] map(%p0, %p1), dimensions={}, to_apply=mapper
    })",
                   R"(
    // CHECK: @main
    // CHECK-NEXT: tensor.extract
    // CHECK-NEXT: tensor.extract
    // CHECK-NEXT: pure_call @mapper_add
    // CHECK-NEXT: return
  )"));
}

TEST_F(ElementalHloToMlirTest, BroadcastSelect) {
  TF_EXPECT_OK(Run(R"(
    ENTRY main {
      p0 = pred[] parameter(0)
      p1 = s32[5,7] parameter(1)
      p2 = s32[5,7] parameter(2)
      ROOT r = s32[5,7] select(p0, p1, p2)
    })",
                   R"(
    // CHECK: @main
    // CHECK-SAME: %[[P0:.*]]: tensor<i1>
    // CHECK-SAME: %[[P1:.*]]: tensor<5x7xi32>, %[[P2:.*]]: tensor<5x7xi32>
    // CHECK-SAME: %[[X:.*]]: index {{{.*}}}, %[[Y:.*]]: index {{{.*}}}
    // CHECK-DAG: tensor.extract %[[P0]][]
    // CHECK-DAG: tensor.extract %[[P1]][%[[X]], %[[Y]]]
    // CHECK-DAG: tensor.extract %[[P2]][%[[X]], %[[Y]]]
  )"));
}

}  // namespace
}  // namespace zkx::emitters
