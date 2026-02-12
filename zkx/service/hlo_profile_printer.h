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

#ifndef ZKX_SERVICE_HLO_PROFILE_PRINTER_H_
#define ZKX_SERVICE_HLO_PROFILE_PRINTER_H_

#include <cstdint>
#include <string>

#include "zkx/service/hlo_profile_printer_data.pb.h"

namespace zkx {

// Pretty-print an array of profile counters using hlo_profile_printer_data.
std::string PrintHloProfile(
    const HloProfilePrinterData& hlo_profile_printer_data,
    const int64_t* counters, double clock_rate_ghz);

}  // namespace zkx

#endif  // ZKX_SERVICE_HLO_PROFILE_PRINTER_H_
