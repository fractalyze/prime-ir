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

#include "zkx/service/gpu/transforms/dynamic_slice_fusion_rewriter.h"

#include <optional>

#include "gtest/gtest.h"

#include "zkx/ffi/ffi.h"
#include "zkx/ffi/ffi_api.h"
#include "zkx/hlo/builder/zkx_builder.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/utils/hlo_matchers.h"
#include "zkx/service/custom_call_target_registry.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx::gpu {

class DynamicSliceFusionRewriterTest : public HloTestBase {};

namespace {
absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                    ffi::AnyBuffer dst) {
  se::DeviceMemoryBase dst_mem = dst.device_memory();
  se::DeviceMemoryBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}
}  // namespace

ZKX_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src
                           .Arg<ffi::AnyBuffer>()  // dst
);
ZKX_FFI_REGISTER_HANDLER(ffi::GetZkxFfiApi(), "__zkx_test$$memcpy", "gpu",
                         kMemcpy);

TEST_F(DynamicSliceFusionRewriterTest, SimpleCustomCall) {
  ZkxBuilder b(TestName());
  CustomCall(
      &b, "__zkx_test$$memcpy",
      /*operands=*/
      {Slice(Broadcast(ConstantR0<int32_t>(&b, 42), {256}), {0}, {128}, {1})},
      ShapeUtil::MakeShape(S32, {128}), /*opaque=*/"",
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  zkx::HloModuleConfig hlo_config(
      zkx::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_zkx_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, zkx::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = s32[256]{0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = s32[128]{0} slice([[P0]]), slice={[0:128]}
    ; CHECK:       ROOT [[CC:%[^ ]+]] = s32[128]{0} custom-call([[S0]]),
    ; CHECK:              custom_call_target="__zkx_test$$memcpy",
    ; CHECK:              api_version=API_VERSION_TYPED_FFI
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       [[C0:%[^ ]+]] = s32[] constant(42)
    ; CHECK:       [[BC:%[^ ]+]] = s32[256]{0} broadcast([[C0]])
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = s32[128]{0} fusion([[BC]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            expected);
}

void Callback_Void(void* stream, void** buffers, const char* /*opaque*/,
                   size_t /*opaque_len*/) {}

ZKX_REGISTER_CUSTOM_CALL_TARGET(Callback_Void, "gpu");

TEST_F(DynamicSliceFusionRewriterTest, SimpleCustomCallLegacy) {
  ZkxBuilder b(TestName());
  CustomCall(
      &b, "Callback_Void",
      /*operands=*/
      {Slice(Broadcast(ConstantR0<int32_t>(&b, 42), {256}), {0}, {128}, {1})},
      ShapeUtil::MakeShape(S32, {128}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  zkx::HloModuleConfig hlo_config(
      zkx::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_zkx_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, zkx::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = s32[256]{0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = s32[128]{0} slice([[P0]]), slice={[0:128]}
    ; CHECK:       ROOT [[CC:%[^ ]+]] = s32[128]{0} custom-call([[S0]]),
    ; CHECK:              custom_call_target="Callback_Void"
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       [[C0:%[^ ]+]] = s32[] constant(42)
    ; CHECK:       [[BC:%[^ ]+]] = s32[256]{0} broadcast([[C0]])
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = s32[128]{0} fusion([[BC]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            expected);
}

TEST_F(DynamicSliceFusionRewriterTest, TupleSliceCustomCallLegacy) {
  ZkxBuilder b(TestName());
  CustomCall(&b, "Callback_Void",
             /*operands=*/
             {
                 Tuple(&b,
                       {
                           Slice(Broadcast(ConstantR0<int32_t>(&b, 5), {8, 8}),
                                 {0, 0}, {4, 8}, {1, 1}),
                           Broadcast(ConstantR0<int32_t>(&b, 2), {256}),
                       }),
                 Tuple(&b,
                       {
                           Broadcast(ConstantR0<int32_t>(&b, 3), {1024}),
                           Broadcast(ConstantR0<int32_t>(&b, 4), {8}),
                       }),
             },
             ShapeUtil::MakeShape(S32, {128}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  zkx::HloModuleConfig hlo_config(
      zkx::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_zkx_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, zkx::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = s32[8,8]{1,0} parameter(0)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = s32[4,8]{1,0} slice([[P0]]), slice={[0:4], [0:8]}
    ; CHECK-DAG:   [[P1:%[^ ]+]] = s32[256]{0} parameter(1)
    ; CHECK-DAG:   [[T0:%[^ ]+]] = (s32[4,8]{1,0}, s32[256]{0}) tuple([[S0]], [[P1]])
    ; CHECK-DAG:   [[P2:%[^ ]+]] = (s32[1024]{0}, s32[8]{0}) parameter(2)
    ; CHECK:       ROOT [[CC:%[^ ]+]] = s32[128]{0} custom-call([[T0]], [[P2]]),
    ; CHECK:              custom_call_target="Callback_Void"
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = s32[128]{0} fusion(
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            expected);
}

TEST_F(DynamicSliceFusionRewriterTest, TupledOutputCustomCallLegacy) {
  ZkxBuilder b(TestName());
  auto custom_call = CustomCall(
      &b, "Callback_Void",
      /*operands=*/
      {
          Tuple(&b,
                {
                    Slice(Broadcast(ConstantR0<int32_t>(&b, 5), {8, 8}), {0, 0},
                          {4, 8}, {1, 1}),
                    Broadcast(ConstantR0<int32_t>(&b, 2), {256}),
                }),
          Tuple(&b,
                {
                    Broadcast(ConstantR0<int32_t>(&b, 3), {1024}),
                    Broadcast(ConstantR0<int32_t>(&b, 4), {8}),
                }),
      },
      ShapeUtil::MakeTupleShape({
          ShapeUtil::MakeShape(S32, {8}),
          ShapeUtil::MakeTupleShape({
              ShapeUtil::MakeShape(S32, {128}),
              ShapeUtil::MakeShape(S32, {256}),
          }),
          ShapeUtil::MakeShape(S32, {1024}),
          ShapeUtil::MakeShape(S32, {4, 8}),
      }),
      /*opaque=*/"");
  Tuple(&b, {GetTupleElement(GetTupleElement(custom_call, 1), 0),
             GetTupleElement(custom_call, 2)});
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  zkx::HloModuleConfig hlo_config(
      zkx::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_zkx_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, zkx::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P2:%[^ ]+]] = (s32[1024]{0}, s32[8]{0}) parameter(2)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = s32[256]{0} parameter(1)
    ; CHECK-DAG:   [[P0:%[^ ]+]] = s32[8,8]{1,0} parameter(0)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = s32[4,8]{1,0} slice([[P0]]), slice={[0:4], [0:8]}
    ; CHECK-DAG:   [[T0:%[^ ]+]] = (s32[4,8]{1,0}, s32[256]{0}) tuple([[S0]], [[P1]])
    ; CHECK:       [[CC:%[^ ]+]] = (s32[8]{0}, (s32[128]{0}, s32[256]{0}), s32[1024]{0}, s32[4,8]{1,0}) custom-call([[T0]], [[P2]]),
    ; CHECK:              custom_call_target="Callback_Void"
    ; CHECK-DAG:   [[GTE0:%[^ ]+]] = s32[8]{0} get-tuple-element([[CC]]), index=0
    ; CHECK-DAG:   [[GTE1:%[^ ]+]] = (s32[128]{0}, s32[256]{0}) get-tuple-element([[CC]]), index=1
    ; CHECK-DAG:   [[GTE2:%[^ ]+]] = s32[128]{0} get-tuple-element([[GTE1]]), index=0
    ; CHECK-DAG:   [[GTE3:%[^ ]+]] = s32[256]{0} get-tuple-element([[GTE1]]), index=1
    ; CHECK-DAG:   [[T1:%[^ ]+]] = (s32[128]{0}, s32[256]{0}) tuple([[GTE2]], [[GTE3]])
    ; CHECK-DAG:   [[GTE4:%[^ ]+]] = s32[1024]{0} get-tuple-element([[CC]]), index=2
    ; CHECK-DAG:   [[GTE5:%[^ ]+]] = s32[4,8]{1,0} get-tuple-element([[CC]]), index=3
    ; CHECK:       ROOT {{.*}} = (s32[8]{0}, (s32[128]{0}, s32[256]{0}), s32[1024]{0}, s32[4,8]{1,0}) tuple([[GTE0]], [[T1]], [[GTE4]], [[GTE5]])
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = (s32[8]{0}, (s32[128]{0}, s32[256]{0}), s32[1024]{0}, s32[4,8]{1,0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK-DAG:   [[GTE6:%[^ ]+]] = s32[1024]{0} get-tuple-element([[FUSION]]), index=2
    ; CHECK-DAG:   [[GTE7:%[^ ]+]] = (s32[128]{0}, s32[256]{0}) get-tuple-element([[FUSION]]), index=1
    ; CHECK-DAG:   [[GTE8:%[^ ]+]] = s32[128]{0} get-tuple-element([[GTE7]]), index=0
    ; CHECK:       ROOT {{.*}} = (s32[128]{0}, s32[1024]{0}) tuple([[GTE8]], [[GTE6]])
    ; CHECK:     }
  )";

  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            expected);
}

TEST_F(DynamicSliceFusionRewriterTest, UnalignedSlice) {
  ZkxBuilder b(TestName());
  CustomCall(
      &b, "Callback_Void",
      /*operands=*/
      {Slice(Broadcast(ConstantR0<int32_t>(&b, 42), {17}), {1}, {17}, {1})},
      ShapeUtil::MakeShape(S32, {16}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  zkx::HloModuleConfig hlo_config(
      zkx::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_zkx_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, zkx::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterDUSConstantOffset) {
  const char* hlo = R"(
  HloModule test, replica_count=2

  add {
    param_0 = s16[] parameter(0)
    param_1 = s16[] parameter(1)
    ROOT add.1 = s16[] add(param_0, param_1)
  }

  ENTRY main.9 {
    param_0 = s16[128,128]{1,0} parameter(0)
    param_1 = s16[128,128]{1,0} parameter(1)
    constant_20 = u32[] constant(20)
    constant_0 = u32[] constant(0)
    reduce-scatter = s16[64,128]{1,0} reduce-scatter(param_0), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
    ROOT loop_dynamic_update_slice_fusion = s16[128,128]{1,0} dynamic-update-slice(param_1, reduce-scatter, constant_20, constant_0)
  }
  )";

  const char* expected = R"(
  // CHECK: %dynamic-slice-fusion{{.+}} {
  // CHECK:   %[[RS:.+]] = s16[64,128]{1,0} reduce-scatter({{.+}})
  // CHECK:   ROOT %{{.+}} = s16[128,128]{1,0} dynamic-update-slice(%{{.+}}, %[[RS]], %{{.+}}, %{{.+}})
  // CHECK: }
  // CHECK: ENTRY {{.+}} {
  // CHECK-NOT: reduce-scatter
  // CHECK:   ROOT %{{.+}} = {{.+}} fusion(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}), kind=kCustom, calls=%dynamic-slice-fusion, {{.+}}"name":"dynamic_address_computation"
  // CHECK: }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterDUSParameterOffset) {
  const char* hlo = R"(
  HloModule test, replica_count=2

  add.clone {
    x.1 = s16[] parameter(0)
    y.1 = s16[] parameter(1)
    ROOT add.462 = s16[] add(x.1, y.1)
  }

  ENTRY %main.9 {
    param_0 = s16[128,128]{1,0} parameter(0)
    param_1 = s16[128,128]{1,0} parameter(1)
    param_2 = u32[] parameter(2)
    constant_0 = u32[] constant(0)
    reduce-scatter = s16[64,128]{1,0} reduce-scatter(param_0), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add.clone
    ROOT dynamic-update-slice = s16[128,128]{1,0} dynamic-update-slice(param_1, reduce-scatter, param_2, constant_0)
  })";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"),
                            std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterDUSLoopIterationOffset) {
  const char* hlo = R"(
  HloModule jit_scan, replica_count=2

  add {
    param_0 = s32[] parameter(0)
    param_1 = s32[] parameter(1)
    ROOT add.6 = s32[] add(param_0, param_1)
  }

  Body {
    arg_tuple.1 = (s32[], s32[128,128]{1,0}, s32[128,128,128]{2,1,0}, s32[128,128]{1,0}) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(arg_tuple.1), index=0
    constant.1 = s32[] constant(1)
    add.7 = s32[] add(get-tuple-element.5, constant.1)
    get-tuple-element.6 = s32[128,128]{1,0} get-tuple-element(arg_tuple.1), index=3
    get-tuple-element.7 = s32[128,128,128]{2,1,0} get-tuple-element(arg_tuple.1), index=2
    reduce-scatter.0 = s32[64,128]{1,0} reduce-scatter(get-tuple-element.6), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
    bitcast.63 = s32[1,64,128]{2,1,0} bitcast(reduce-scatter.0)
    constant.2 = s32[] constant(0)
    compare.4 = pred[] compare(get-tuple-element.5, constant.2), direction=LT
    constant.3 = s32[] constant(128)
    add.8 = s32[] add(get-tuple-element.5, constant.3)
    select.2 = s32[] select(compare.4, add.8, get-tuple-element.5)
    dynamic-update-slice.2 = s32[128,128,128]{2,1,0} dynamic-update-slice(get-tuple-element.7, bitcast.63, select.2, constant.2, constant.2)
    ROOT tuple.1 = tuple(add.7, get-tuple-element.6, dynamic-update-slice.2, get-tuple-element.6)
  } // Body

  Cond {
    arg_tuple.0 = (s32[], s32[128,128]{1,0}, s32[128,128,128]{2,1,0}, s32[128,128]{1,0}) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(arg_tuple.0), index=0
    constant.0 = s32[] constant(128)
    ROOT compare.5 = pred[] compare(get-tuple-element.4, constant.0), direction=LT
  }

  ENTRY main.55 {
    Arg_2.3 = s32[128,128,128]{2,1,0} parameter(2)
    constant.4 = s32[] constant(0)
    Arg_1.2 = s32[128,128]{1,0} parameter(1)
    constant.5 = s32[] constant(0)
    broadcast.1 = s32[128,128,128]{2,1,0} broadcast(constant.5), dimensions={}
    Arg_0.1 = s32[128,128]{1,0} parameter(0)
    tuple = tuple(constant.4, Arg_1.2, broadcast.1, Arg_0.1)
    while = while(tuple), condition=Cond, body=Body, backend_config={"known_trip_count":{"n":"128"}}
    get-tuple-element.50 = s32[128,128]{1,0} get-tuple-element(while), index=1
    get-tuple-element.51 = s32[128,128,128]{2,1,0} get-tuple-element(while), index=2
    ROOT tuple.54 = (s32[128,128]{1,0}, s32[128,128,128]{2,1,0}) tuple(get-tuple-element.50, get-tuple-element.51)
  })";
  const char* expected = R"(
  // CHECK: %dynamic-slice-fusion{{.*}}{
  // CHECK:   {{.+}} = {{.*}}reduce-scatter({{.+}})
  // CHECK:   {{.+}} = {{.*}}dynamic-update-slice({{.+}})
  // CHECK: }
  // CHECK: Body{{.+}}{
  // CHECK-NOT: {{.+}} = {{.*}}reduce-scatter({{.+}})
  // CHECK:   {{.+}} = {{.+}}fusion({{.+}}), kind=kCustom, calls=%dynamic-slice-fusion{{.*}}"name":"dynamic_address_computation"
  // CHECK: }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

// Remove this when tuple support is added to dynamic slice fusion
TEST_F(DynamicSliceFusionRewriterTest, DUSReduceScatterTupleNoTransform) {
  const char* hlo = R"(
  HloModule test, replica_count=2

  add {
    param_0 = s16[] parameter(0)
    param_1 = s16[] parameter(1)
    ROOT add.1 = s16[] add(param_0, param_1)
  }

  ENTRY main.9 {
    param_0 = s16[128,128]{1,0} parameter(0)
    param_1 = s16[128,128]{1,0} parameter(1)
    param_2 = s16[128,128]{1,0} parameter(2)
    constant_20 = u32[] constant(20)
    constant_0 = u32[] constant(0)
    reduce-scatter = (s16[64,128]{1,0}, s16[64,128]{1,0}) reduce-scatter(param_0, param_2), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
    rs1 = get-tuple-element(reduce-scatter), index=0
    ROOT loop_dynamic_update_slice_fusion = s16[128,128]{1,0} dynamic-update-slice(param_1, rs1, constant_20, constant_0)
  })";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"),
                            std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterSlice) {
  const char* hlo = R"(
  HloModule jit_slice, replica_count=2

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = add(a,b)
  }

  ENTRY %main.9 {
    p0 = s32[2,8,32]{2,1,0} parameter(0)
    slice = s32[1,8,32]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:32]}
    bc = s32[8,32]{1,0} bitcast(%slice)
    ROOT rs = s32[4,32] reduce-scatter(bc), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
  })";
  const char* expected = R"(
  // CHECK: dynamic-slice-fusion{{.*}} {
  // CHECK:   %[[p0:.+]] = {{.+}} parameter(0)
  // CHECK:   %[[slice:.+]] = {{.+}} slice(%[[p0]]), slice={[1:2], [0:8], [0:32]}
  // CHECK:   %[[bc:.+]] = {{.+}} bitcast(%[[slice]])
  // CHECK:   ROOT {{.+}} = {{.+}} reduce-scatter(%[[bc]])
  // CHECK: }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterDynamicSlice) {
  const char* hlo = R"(
  HloModule jit_slice, replica_count=2

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = add(a,b)
  }

  ENTRY %main.9 {
    p0 = s32[2,8,32]{2,1,0} parameter(0)
    c0 = s32[] constant(0)
    c1 = s32[] constant(1)
    slice = s32[1,8,32]{2,1,0} dynamic-slice(p0, c1, c0, c0), dynamic_slice_sizes={1,8,32}
    bc = s32[8,32]{1,0} bitcast(%slice)
    ROOT rs = s32[4,32] reduce-scatter(bc), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
  })";
  const char* expected = R"(
  // CHECK: add
  // CHECK: dynamic-slice-fusion{{.*}} {
  // CHECK:   %[[p0:.+]] = {{.+}} parameter(0)
  // CHECK:   %[[slice:.+]] = {{.+}} dynamic-slice(%[[p0]], {{.+}}), dynamic_slice_sizes={1,8,32}
  // CHECK:   %[[bc:.+]] = {{.+}} bitcast(%[[slice]])
  // CHECK:   ROOT {{.+}} = {{.+}} reduce-scatter(%[[bc]])
  // CHECK: }
  // CHECK: ENTRY
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest,
       OffsetAsFunctionOfInductionVariableShouldFuse) {
  const char* hlo = R"(
    HloModule test, replica_count=2
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }
    body {
      param.1 = (s32[], s32[32,32], s32[32,32]) parameter(0)
      iter.1 = s32[] get-tuple-element(param.1), index=0
      src = s32[32,32] get-tuple-element(param.1), index=1
      dest = s32[32,32] get-tuple-element(param.1), index=2

      // offset as a function of only the loop induction variable.
      add.1 = s32[] add(iter.1, iter.1)
      c3 = s32[] constant(3)
      multiply.1 = s32[] multiply(add.1, c3)
      c16 = s32[] constant(16)
      offset.1 = s32[] subtract(multiply.1, c16)

      c0 = s32[] constant(0)
      rs = s32[16,32] reduce-scatter(src), dimensions={0}, replica_groups={{0,1}}, to_apply=add
      dus = s32[32,32] dynamic-update-slice(dest, rs, offset.1, c0)
      c1 = s32[] constant(1)
      add.2 = s32[] add(iter.1, c1)
      ROOT tuple = tuple(add.2, src, dus)
    }
    condition {
      param.2 = (s32[], s32[32,32], s32[32,32]) parameter(0)
      iter.2 = s32[] get-tuple-element(param.2), index=0
      c16 = s32[] constant(16)
      ROOT compare = pred[] compare(iter.2, c16), direction=LT
    }
    ENTRY main {
      src = s32[32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[32,32], s32[32,32]) tuple(c0, src, dest)
      ROOT while = (s32[], s32[32,32], s32[32,32]) while(tuple), body=body, condition=condition
    }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), R"(
    // CHECK: dynamic-slice-fusion
    // CHECK:    %[[rs:.+]] = {{.+}} reduce-scatter({{.+}})
    // CHECK:    ROOT %[[dus:.+]] = {{.+}} dynamic-update-slice({{.+}})
    // CHECK: body
    // CHECK:   %[[fusion:.+]] = {{.+}} fusion({{.+}}), kind=kCustom, calls=%dynamic-slice-fusion,
    // CHECK-SAME:  "fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"
  )");
}

}  // namespace zkx::gpu
