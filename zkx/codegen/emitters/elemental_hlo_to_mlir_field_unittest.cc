/* Copyright 2026 The ZKX Authors.

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

#include <string>

#include "absl/strings/substitute.h"
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
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/gpu/codegen/emitters/ir/zkx_gpu_ops.h"
#include "zkx/codegen/emitters/elemental_hlo_to_mlir.h"
#include "zkx/codegen/emitters/ir/zkx_ops.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/testlib/file_check.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "zkx/primitive_util.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/status_macros.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx::emitters {
namespace {

class FieldElementalHloToMlirTestBase : public HloTestBase {
 public:
  FieldElementalHloToMlirTestBase() {
    context_.loadDialect<
        // clang-format off
        gpu::ZkxGpuDialect,
        mlir::affine::AffineDialect,
        mlir::arith::ArithDialect,
        mlir::DLTIDialect,
        mlir::func::FuncDialect,
        mlir::LLVM::LLVMDialect,
        mlir::math::MathDialect,
        mlir::mhlo::MhloDialect,
        mlir::prime_ir::field::FieldDialect,
        mlir::scf::SCFDialect,
        mlir::tensor::TensorDialect,
        ZkxDialect
        // clang-format on
        >();
  }

  absl::Status Run(std::string_view hlo, std::string_view filecheck_str) {
    TF_ASSIGN_OR_RETURN(auto hlo_module, ParseAndReturnUnverifiedModule(hlo));

    mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&context_),
                                       &context_);
    auto module = llvm_ir::CreateMlirModuleOp(builder.getLoc());
    (*module)->setAttr(
        mlir::DLTIDialect::kDataLayoutAttrName,
        mlir::parseAttribute("#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>",
                             builder.getContext()));
    builder.setInsertionPointToStart(module->getBody());
    auto* entry_computation = hlo_module->entry_computation();
    PartitionedComputations partitioned_computations(entry_computation,
                                                     &context_);
    auto fns = partitioned_computations.DeclareFunctions(module.get());
    auto entry_func = fns[&partitioned_computations
                               .FindPartitionedComputation(entry_computation)
                               .GetRootSubgraph()];
    auto& entry_pc =
        partitioned_computations.FindPartitionedComputation(entry_computation);
    auto call_targets = partitioned_computations.CreateCallTargetProvider(fns);
    TF_RETURN_IF_ERROR(SubgraphToMlirFunction(
        entry_pc, entry_pc.GetRootSubgraph(), entry_func, call_targets));

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

template <typename F>
class FieldElementalHloToMlirTest : public FieldElementalHloToMlirTestBase {
 public:
  void SetUp() override {
    FieldElementalHloToMlirTestBase::SetUp();
    field_name_ = std::string(primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>()));
    using StdType = typename F::Config::StdConfig;
    using StdField = zk_dtypes::PrimeField<StdType>;
    std_field_name_ = std::string(primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<StdField>()));
  }

 protected:
  absl::Status RunField(std::string_view hlo_template,
                        std::string_view filecheck_str) {
    std::string hlo =
        absl::Substitute(hlo_template, field_name_, std_field_name_);
    return Run(hlo, filecheck_str);
  }

  std::string field_name_;
  std::string std_field_name_;
};

using FieldTypes = testing::Types<
    // clang-format off
    zk_dtypes::BabybearMont,
    zk_dtypes::GoldilocksMont,
    zk_dtypes::bn254::FrMont
    // clang-format on
    >;
TYPED_TEST_SUITE(FieldElementalHloToMlirTest, FieldTypes);

TYPED_TEST(FieldElementalHloToMlirTest, ConvertIntToField) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = u32[4] parameter(0)
      ROOT convert = $0[4] convert(p0)
    })",
                              R"(
    // CHECK: @main_convert
    // CHECK: field.bitcast
    // CHECK: field.to_mont
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, ConvertFieldToStd) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      ROOT convert = $1[4] convert(p0)
    })",
                              R"(
    // CHECK: @main_convert
    // CHECK: field.from_mont
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, FieldAdd) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      p1 = $0[4] parameter(1)
      ROOT add = $0[4] add(p0, p1)
    })",
                              R"(
    // CHECK: @main_add
    // CHECK: field.add
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, FieldMultiply) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      p1 = $0[4] parameter(1)
      ROOT mul = $0[4] multiply(p0, p1)
    })",
                              R"(
    // CHECK: @main_mul
    // CHECK: field.mul
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, FieldNegate) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      ROOT neg = $0[4] negate(p0)
    })",
                              R"(
    // CHECK: @main_neg
    // CHECK: field.negate
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, FieldInverse) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      ROOT inv = $0[4] inverse(p0)
    })",
                              R"(
    // CHECK: @main_inv
    // CHECK: field.inverse
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, FieldCompare) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      p1 = $0[4] parameter(1)
      ROOT cmp = pred[4] compare(p0, p1), direction=EQ
    })",
                              R"(
    // CHECK: @main_cmp
    // CHECK: field.cmp
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, FieldDivide) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      p1 = $0[4] parameter(1)
      ROOT div = $0[4] divide(p0, p1)
    })",
                              R"(
    // CHECK: @main_div
    // CHECK: field.inverse
    // CHECK: field.mul
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, FieldSubtract) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      p1 = $0[4] parameter(1)
      ROOT sub = $0[4] subtract(p0, p1)
    })",
                              R"(
    // CHECK: @main_sub
    // CHECK: field.sub
  )"));
}

TYPED_TEST(FieldElementalHloToMlirTest, FieldPower) {
  TF_EXPECT_OK(this->RunField(R"(
    ENTRY main {
      p0 = $0[4] parameter(0)
      p1 = u32[4] parameter(1)
      ROOT pow = $0[4] power(p0, p1)
    })",
                              R"(
    // CHECK: @main_pow
    // CHECK: field.powui
  )"));
}

}  // namespace
}  // namespace zkx::emitters
