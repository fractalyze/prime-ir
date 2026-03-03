/* Copyright 2023 The OpenXLA Authors.
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

#ifndef ZKX_PJRT_C_PJRT_C_API_TEST_BASE_H_
#define ZKX_PJRT_C_PJRT_C_API_TEST_BASE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "zkx/pjrt/c/pjrt_c_api.h"
#include "zkx/pjrt/c/pjrt_c_api_helpers.h"
#include "zkx/pjrt/pjrt_client.h"
#include "zkx/pjrt/pjrt_future.h"
#include "zkx/shape.h"

namespace pjrt {

template <typename T>
absl::Span<const char> GetRawView(const std::vector<T>& v) {
  return absl::Span<const char>(reinterpret_cast<const char*>(v.data()),
                                v.size() * sizeof(T));
}

class PjrtCApiTestBase : public ::testing::Test {
 public:
  explicit PjrtCApiTestBase(const PJRT_Api* api);
  ~PjrtCApiTestBase() override;

 protected:
  const PJRT_Api* api_;
  PJRT_Client* client_;  // not owned
  void destroy_client(PJRT_Client* client);

  int GetDeviceId(PJRT_DeviceDescription* device_desc) const;

  int GetDeviceId(PJRT_Device* device) const;

  bool IsValidDeviceId(PJRT_Device* device) const;

  int GetLocalHardwareId(PJRT_Device* device) const;

  absl::Span<PJRT_Device* const> GetClientDevices() const;

  int GetNumDevices() const;

  std::string BuildSingleDeviceCompileOptionStr();

  absl::Span<PJRT_Device* const> GetClientAddressableDevices() const;

  PJRT_Client_BufferFromHostBuffer_Args CreateBufferFromHostBufferArgs(
      const std::vector<int32_t>& data, const zkx::Shape& shape,
      zkx::PjRtClient::HostBufferSemantics host_buffer_semantics,
      PJRT_Device* device = nullptr);

  std::pair<std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter>,
            zkx::PjRtFuture<>>
  create_buffer_from_data(const std::vector<int32_t>& int32_data,
                          const zkx::Shape& shape,
                          PJRT_Device* device = nullptr);

  std::pair<std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter>,
            zkx::PjRtFuture<>>
  create_buffer(PJRT_Device* device = nullptr);

  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> ToUniquePtr(
      PJRT_Error* error);

  std::unique_ptr<PJRT_AsyncHostToDeviceTransferManager,
                  ::pjrt::PJRT_AsyncHostToDeviceTransferManagerDeleter>
  create_transfer_manager(const zkx::Shape& host_shape);

 private:
  PjrtCApiTestBase(const PjrtCApiTestBase&) = delete;
  void operator=(const PjrtCApiTestBase&) = delete;
};

}  // namespace pjrt

#endif  // ZKX_PJRT_C_PJRT_C_API_TEST_BASE_H_
