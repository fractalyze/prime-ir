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

// BabyBear field type aliases for AOT runtime.

!babybear = !field.pf<2013265921:i32>
!babybear_mont = !field.pf<2013265921:i32, true>
// BabyBear quartic extension: x⁴ - 11 (non-residue = 11)
!babybearx4 = !field.ef<4x!babybear, 11:i32>
!babybearx4m = !field.ef<4x!babybear_mont, 11:i32>
