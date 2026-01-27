/* Copyright 2019 The TensorFlow Authors All Rights Reserved.
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

#include "xla/tsl/profiler/lib/scoped_annotation.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

namespace tsl::profiler {
namespace {

TEST(ScopedAnnotation, Simple) {
  {
    ScopedAnnotation trace("blah");
    EXPECT_EQ(AnnotationStack::Get(), "");  // not enabled
  }

  {
    AnnotationStack::Enable(true);
    ScopedAnnotation trace("blah");
    EXPECT_EQ(AnnotationStack::Get(), "blah");  // enabled
    AnnotationStack::Enable(false);
  }

  {
    AnnotationStack::Enable(true);
    ScopedAnnotation outer("foo");
    ScopedAnnotation inner("bar");
    EXPECT_EQ(AnnotationStack::Get(), "foo::bar");  // enabled
    AnnotationStack::Enable(false);
  }

  {
    AnnotationStack::Enable(true);
    PushAnnotation("foo");
    PushAnnotation("bar");
    EXPECT_EQ(AnnotationStack::Get(), "foo::bar");  // enabled
    PopAnnotation();
    PopAnnotation();
    AnnotationStack::Enable(false);
  }

  EXPECT_EQ(AnnotationStack::Get(), "");  // not enabled
}

}  // namespace
}  // namespace tsl::profiler
