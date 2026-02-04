/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#ifndef XLA_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_
#define XLA_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"

#include "xla/tsl/profiler/lib/profiler_interface.h"
#include "xla/tsl/profiler/protobuf/xplane.pb.h"

namespace tsl::profiler {

// ProfilerCollection multiplexes ProfilerInterface calls into a collection of
// profilers.
class ProfilerCollection : public ProfilerInterface {
 public:
  explicit ProfilerCollection(
      std::vector<std::unique_ptr<ProfilerInterface>> profilers);

  absl::Status Start() override;

  absl::Status Stop() override;

  absl::Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  std::vector<std::unique_ptr<ProfilerInterface>> profilers_;
};

}  // namespace tsl::profiler

#endif  // XLA_TSL_PROFILER_LIB_PROFILER_COLLECTION_H_
