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

!PF = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!SF = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256>
!PFm = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256, true>
!SFm = !field.pf<21888242871839275222246405745257275088548364400416034343698204186575808495617:i256, true>

#beta = #field.pf.elem<21888242871839275222246405745257275088548364400416034343698204186575808495616:i256> : !PF
!QF = !field.f2<!PF, #beta>
#beta_mont = #field.pf.elem<21888242871839275222246405745257275088548364400416034343698204186575808495616:i256> : !PFm
!QFm = !field.f2<!PFm, #beta_mont>
