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

#include "zkx/literal_pool.h"

#include <algorithm>

#include "absl/debugging/leak_check.h"
#include "absl/log/log.h"

namespace zkx {

// static
LiteralPool* LiteralPool::Default() {
  static auto* pool = absl::IgnoreLeak(new LiteralPool());
  return pool;
}

// Erases expired weak pointers from the vector and returns the number of
// elements that were erased.
namespace {

size_t EraseExpiredLiterals(std::vector<std::weak_ptr<Literal>>& literals) {
  auto it = std::remove_if(literals.begin(), literals.end(),
                           [](auto& ptr) { return ptr.expired(); });
  size_t num_erased = std::distance(it, literals.end());

  literals.erase(it, literals.end());
  return num_erased;
}

}  // namespace

size_t LiteralPool::GarbageCollect() {
  absl::MutexLock lock(&mu_);
  size_t num_erased = 0;

  for (auto& [shape, literals] : literals_) {
    num_erased += EraseExpiredLiterals(literals);
  }

  VLOG(3) << "Garbage collected " << num_erased << " literals";
  return num_erased;
}

size_t LiteralPool::GarbageCollect(Shape shape) {
  absl::MutexLock lock(&mu_);
  size_t num_erased = 0;

  if (auto it = literals_.find(shape); it != literals_.end()) {
    num_erased = EraseExpiredLiterals(it->second);
  }

  VLOG(3) << "Garbage collected " << num_erased << " literals for shape "
          << shape.ToString();
  return num_erased;
}

// Tried to find a canonical literal in the pool. Return nullptr if not found.
namespace {

std::shared_ptr<Literal> FindCanonicalLiteral(
    std::vector<std::weak_ptr<Literal>>& literals, const Literal& literal) {
  for (std::weak_ptr<Literal>& ptr : literals) {
    if (auto locked_ptr = ptr.lock()) {
      if (locked_ptr->Equal(literal, /*layout_sensitive=*/true)) {
        return locked_ptr;
      }
    }
  }

  return nullptr;
}

}  // namespace

std::shared_ptr<Literal> LiteralPool::GetCanonicalLiteral(
    const Literal& literal) {
  absl::MutexLock lock(&mu_);

  auto& literals = literals_[literal.shape()];
  if (auto ptr = FindCanonicalLiteral(literals, literal)) {
    return ptr;
  }

  std::shared_ptr<Literal> new_literal = literal.CloneToUnique();
  literals.push_back(new_literal);
  return new_literal;
}

std::shared_ptr<Literal> LiteralPool::GetCanonicalLiteral(
    std::shared_ptr<Literal> literal) {
  absl::MutexLock lock(&mu_);

  auto& literals = literals_[literal->shape()];
  if (auto ptr = FindCanonicalLiteral(literals, *literal)) {
    return ptr;
  }

  literals.push_back(literal);
  return literal;
}

}  // namespace zkx
