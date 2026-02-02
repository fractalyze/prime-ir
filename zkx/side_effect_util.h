/* Copyright 2018 The OpenXLA Authors.
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

#ifndef ZKX_SIDE_EFFECT_UTIL_H_
#define ZKX_SIDE_EFFECT_UTIL_H_

namespace zkx {

// ZKX frontend attribute for stream annotation.
extern const char kZkxStreamAnnotationAttr[];

// ZKX frontend attribute for specifying the scheduling group id annotations.
extern const char kZkxSchedulingGroupIdAttr[];

// ZKX frontend attribute for specifying the type of computation.
extern const char kZkxComputeTypeAttr[];

// ZKX frontend attribute values for kZkxComputeTypeAttr.
extern const char kZkxComputeTypeHost[];

}  // namespace zkx

#endif  // ZKX_SIDE_EFFECT_UTIL_H_
