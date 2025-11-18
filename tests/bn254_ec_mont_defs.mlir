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

#a = #field.pf.elem<0:i256> : !PFm
#b = #field.pf.elem<3:i256> : !PFm
#1 = #field.pf.elem<1:i256> : !PFm
#2 = #field.pf.elem<2:i256> : !PFm

#curve = #elliptic_curve.sw<#a, #b, (#1, #2)>
!affine = !elliptic_curve.affine<#curve>
!jacobian = !elliptic_curve.jacobian<#curve>
!xyzz = !elliptic_curve.xyzz<#curve>

#f2_elem = #field.f2.elem<#1, #2> : !QFm
#g2curve = #elliptic_curve.sw<#f2_elem, #f2_elem, (#f2_elem, #f2_elem)>
!g2affine = !elliptic_curve.affine<#g2curve>
!g2jacobian = !elliptic_curve.jacobian<#g2curve>
!g2xyzz = !elliptic_curve.xyzz<#g2curve>
