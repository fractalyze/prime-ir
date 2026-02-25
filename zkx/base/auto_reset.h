// Copyright 2011 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE.chromium file.

/* Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_BASE_AUTO_RESET_H_
#define ZKX_BASE_AUTO_RESET_H_

#include <utility>

#include "absl/log/check.h"
// #include "base/memory/raw_ptr_exclusion.h"

// base::AutoReset<> is useful for setting a variable to a new value only within
// a particular scope. An base::AutoReset<> object resets a variable to its
// original value upon destruction, making it an alternative to writing
// "var = false;" or "var = old_val;" at all of a block's exit points.
//
// This should be obvious, but note that an base::AutoReset<> instance should
// have a shorter lifetime than its scoped_variable, to prevent invalid memory
// writes when the base::AutoReset<> object is destroyed.

namespace zkx::base {

template <typename T>
class [[maybe_unused, nodiscard]] AutoReset {
 public:
  template <typename U>
  AutoReset(T* scoped_variable, U&& new_value)
      : scoped_variable_(scoped_variable),
        original_value_(
            std::exchange(*scoped_variable_, std::forward<U>(new_value))) {}

  // A constructor that's useful for asserting the old value of
  // `scoped_variable`, especially when it's inconvenient to check this before
  // constructing the AutoReset object (e.g. in a class member initializer
  // list).
  template <typename U>
  AutoReset(T* scoped_variable, U&& new_value, const T& expected_old_value)
      : AutoReset(scoped_variable, new_value) {
    DCHECK_EQ(original_value_, expected_old_value);
  }

  AutoReset(AutoReset&& other)
      : scoped_variable_(std::exchange(other.scoped_variable_, nullptr)),
        original_value_(std::move(other.original_value_)) {}

  AutoReset& operator=(AutoReset&& rhs) {
    scoped_variable_ = std::exchange(rhs.scoped_variable_, nullptr);
    original_value_ = std::move(rhs.original_value_);
    return *this;
  }

  ~AutoReset() {
    if (scoped_variable_) *scoped_variable_ = std::move(original_value_);
  }

 private:
  // `scoped_variable_` is not a raw_ptr<T> for performance reasons: Large
  // number of non-PartitionAlloc pointees + AutoReset is typically short-lived
  // (e.g. allocated on the stack).
  // RAW_PTR_EXCLUSION T* scoped_variable_;
  T* scoped_variable_;  // not owned

  T original_value_;
};

}  // namespace zkx::base

#endif  // ZKX_BASE_AUTO_RESET_H_
