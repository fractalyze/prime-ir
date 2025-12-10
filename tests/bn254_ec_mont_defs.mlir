// Copyright 2025 The ZKIR Authors.
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

#curve = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !PFm
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

#f2_elem = dense<[1, 2]>: tensor<2xi256>
#g2curve = #elliptic_curve.sw<#f2_elem, #f2_elem, (#f2_elem, #f2_elem)> : !QFm
!g2affine = !elliptic_curve.affine<#g2curve>
!g2jacobian = !elliptic_curve.jacobian<#g2curve>
!g2xyzz = !elliptic_curve.xyzz<#g2curve>
