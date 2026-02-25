// Copyright 2026 The ZKX Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "zkx/maybe_owning.h"

#include "gtest/gtest.h"

namespace zkx {

TEST(MaybeOwningTest, Null) {
  MaybeOwning<char> m(nullptr);
  EXPECT_EQ(m.get(), nullptr);
  EXPECT_EQ(m.get_mutable(), nullptr);
}

TEST(MaybeOwningTest, Owning) {
  MaybeOwning<char> m(std::make_unique<char>());
  *m.get_mutable() = 'a';
  EXPECT_EQ(*m, 'a');
}

TEST(MaybeOwningTest, Shared) {
  auto owner = std::make_unique<char>();
  *owner = 'x';
  MaybeOwning<char> c1(owner.get());
  MaybeOwning<char> c2(owner.get());

  EXPECT_EQ(*c1, 'x');
  EXPECT_EQ(*c2, 'x');
  EXPECT_EQ(c1.get(), c2.get());
}

}  // namespace zkx
