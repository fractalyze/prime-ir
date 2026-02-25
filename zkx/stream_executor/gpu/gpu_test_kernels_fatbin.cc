/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/stream_executor/gpu/gpu_test_kernels_fatbin.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/strings/match.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/test_util.h"
#include "zkx/stream_executor/gpu/gpu_init.h"

namespace stream_executor::gpu {

namespace {

// Reads an archive file, searches for a section that starts with
// 'fatbin_section_prefix' and returns the contents of that section as a vector
// of bytes.
absl::StatusOr<std::vector<uint8_t>> GetFatbinFromArchive(
    llvm::StringRef archive_path, llvm::StringRef fatbin_section_prefix) {
  tsl::Env* env = tsl::Env::Default();

  std::string file_contents;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      env, std::string_view(archive_path.data(), archive_path.size()),
      &file_contents));

  const auto buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(file_contents),
      /*BufferName=*/"", /*RequiresNullTerminator=*/false);

  auto archive_ptr = llvm::object::Archive::create(buffer->getMemBufferRef());

  if (!archive_ptr) {
    return absl::InternalError(llvm::toString(archive_ptr.takeError()));
  }

  const llvm::object::Archive* archive = archive_ptr.get().get();

  llvm::Error archive_error = llvm::Error::success();
  for (const auto& child : archive->children(archive_error)) {
    if (archive_error) {
      return absl::InternalError(llvm::toString(std::move(archive_error)));
    }

    auto binary = child.getAsBinary();
    if (!binary) {
      continue;
    }

    auto executable_elf_object_file_ptr =
        llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(binary.get());
    if (!executable_elf_object_file_ptr) {
      continue;
    }

    const auto executable_elf_object_file =
        executable_elf_object_file_ptr.get();

    for (const auto& section : executable_elf_object_file->sections()) {
      if (absl::StartsWith(section.getName().get().str(),
                           fatbin_section_prefix)) {
        const std::string fatbin_contents = section.getContents().get().str();
        return std::vector<uint8_t>(fatbin_contents.begin(),
                                    fatbin_contents.end());
      }
    }
  }

  return absl::InternalError("Fatbin section not found in generated archive.");
}

}  // namespace

absl::StatusOr<std::vector<uint8_t>> GetGpuTestKernelsFatbin() {
  const std::string platform_name = GpuPlatformName();
  std::string archive_filename;
  std::string fatbin_prefix;

  if (platform_name == "CUDA") {
    archive_filename = "libgpu_test_kernels_cuda.a";
    fatbin_prefix = ".nv_fatbin";
  } else if (platform_name == "ROCM") {
    archive_filename = "libgpu_test_kernels_rocm.a";
    fatbin_prefix = ".hip_fatbin";
  } else {
    return absl::InternalError("Unsupported GPU platform: " + platform_name);
  }

  std::string file_path = tsl::io::JoinPath(
      tsl::testing::ZkxSrcRoot(), "stream_executor", "gpu", archive_filename);

  return GetFatbinFromArchive(file_path, fatbin_prefix);
}
}  // namespace stream_executor::gpu
