# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load("@rules_cc//cc:defs.bzl", "cc_library")

# TODO(rdzhabarov): Tighten visibility after migration is complete.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "bitmap",
    srcs = ["bitmap.cc"],
    hdrs = ["bitmap.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
    ],
    alwayslink = 1,
)

cc_library(
    name = "bits",
    hdrs = ["bits.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
    ],
)
