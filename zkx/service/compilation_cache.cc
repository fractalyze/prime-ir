/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/service/compilation_cache.h"

#include <utility>

#include "absl/base/const_init.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "zkx/zkx_data.pb.h"

namespace zkx {

namespace {

int64_t GetUniqueId() {
  static absl::Mutex mu(absl::kConstInit);
  static int64_t counter = 0;
  absl::MutexLock loc(mu);
  const int64_t id = counter++;
  return id;
}

}  // namespace

ExecutionHandle CompilationCache::Insert(
    std::unique_ptr<Executable> executable) {
  absl::MutexLock lock(mutex_);

  CacheKey key = GetUniqueId();
  VLOG(2) << "inserting cache key: " << key;
  CHECK_EQ(cache_.count(key), 0);
  cache_.emplace(key, std::move(executable));

  ExecutionHandle handle;
  handle.set_handle(key);
  return handle;
}

absl::StatusOr<std::shared_ptr<Executable>> CompilationCache::LookUp(
    const ExecutionHandle& handle) const {
  absl::MutexLock lock(mutex_);

  CacheKey key = handle.handle();
  VLOG(2) << "looking up cache key: " << key;
  if (cache_.count(key) == 0) {
    VLOG(2) << "cache key not found: " << key;
    return absl::InvalidArgumentError(
        absl::StrCat("can not find executable with handle ", key));
  }
  auto& result = cache_.at(key);
  VLOG(2) << "hit executable: " << result->module().name();
  return result;
}

}  // namespace zkx
