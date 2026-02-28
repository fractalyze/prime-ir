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

#include "zkx/pjrt/plugin/gpu/gpu_pjrt_client.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/pjrt/gpu/se_gpu_pjrt_client.h"

namespace zkx {

TEST(ZkxGpuPjrtClientTest, GetZkxPjrtGpuClient) {
  GpuClientOptions options;
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetZkxGpuPjrtClient(options));
  EXPECT_TRUE(client->platform_name() == "cuda" ||
              client->platform_name() == "rocm");
}

}  // namespace zkx
