// Copyright 2026 The PrimeIR Authors.
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

#include "prime_ir/Dialect/Field/IR/ExtensionFieldOperation.h"

namespace mlir::prime_ir::field {

template class ExtensionFieldOperation<2>;
template class ExtensionFieldOperation<3>;
template class ExtensionFieldOperation<4>;

template raw_ostream &operator<<(raw_ostream &os,
                                 const ExtensionFieldOperation<2> &op);
template raw_ostream &operator<<(raw_ostream &os,
                                 const ExtensionFieldOperation<3> &op);
template raw_ostream &operator<<(raw_ostream &os,
                                 const ExtensionFieldOperation<4> &op);

} // namespace mlir::prime_ir::field
