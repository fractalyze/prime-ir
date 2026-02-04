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

#ifndef ZKX_ARRAY4D_H_
#define ZKX_ARRAY4D_H_

#include <stdint.h>

#include <initializer_list>
#include <vector>

#include "zkx/array.h"

namespace zkx {

// Simple 4D array structure, similar in form to Array2D, for use primarily in
// testing and describing to ZKX APIs values in the 4D array structures used
// in convolutions.
//
// The data layout is, in order from major to minor:
//
//    First dimension: plane, batch, n1
//   Second dimension: depth, feature, z, n2
//    Third dimension: height, y, n3
//   Fourth dimension: width, x, n4
//
// These dimensions are referred to by various names, so that is why
// more than one name is given above. See operator() for the exact
// calculation of 1d indices from 4d indices.
template <typename T>
class Array4D : public Array<T> {
 public:
  Array4D() : Array<T>(std::vector<int64_t>{0, 0, 0, 0}) {}

  // Creates a 4D array, uninitialized values.
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width)
      : Array<T>(std::vector<int64_t>{planes, depth, height, width}) {}

  // Creates a 4D array, initialized to value.
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width, T value)
      : Array<T>(std::vector<int64_t>{planes, depth, height, width}, value) {}

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  Array4D(typename Array<T>::template InitializerList4D<T> values)
      : Array<T>(values) {}

  int64_t n4() const { return this->dim(3); }
  int64_t n3() const { return this->dim(2); }
  int64_t n2() const { return this->dim(1); }
  int64_t n1() const { return this->dim(0); }

  int64_t width() const { return this->dim(3); }
  int64_t height() const { return this->dim(2); }
  int64_t depth() const { return this->dim(1); }
  int64_t planes() const { return this->dim(0); }
};

}  // namespace zkx

#endif  // ZKX_ARRAY4D_H_
