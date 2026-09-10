# Copyright 2025 The PrimeIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import subprocess
from pathlib import Path

from lit.formats import ShTest

config.name = "prime_ir"
config.test_format = ShTest()
config.suffixes = [".mlir", ".v"]

# lit executes relative to the directory
#
#   bazel-bin/tests/<test_target_name>.runfiles/<this repository>/
#
# which contains tools/ and tests/ directories and the binary targets built
# within them, brought in via the `data` attribute in the BUILD file. To
# manually inspect the filesystem in situ, add the following to this script and
# run `bazel test //tests:<target>`
#
#   import subprocess
#
#   print(subprocess.run(["pwd",]).stdout)
#   print(subprocess.run(["ls", "-l", os.environ["RUNFILES_DIR"]]).stdout)
#   print(subprocess.run([ "env", ]).stdout)
#
# Hence, to get lit to see tools like `prime-ir-opt`, we need to add the tools/
# subdirectory to the PATH environment variable.
#
# Bazel defines RUNFILES_DIR, which holds one directory per repository. Neither
# that directory's name nor this repository's own is a constant: on the WORKSPACE
# lane they are the apparent names (`prime_ir`, `llvm-project`), while under
# `--config=bzlmod` this repository is `_main` and an external one carries a
# canonical name that prefixes the apparent one (`_main~llvm_project~llvm-project`
# on Bazel 7, `+llvm_project+llvm-project` on Bazel 8). Both lanes are resolved
# below rather than spelled out.

runfiles_dir = Path(os.environ["RUNFILES_DIR"])

# lit runs with this repository's own runfiles directory as the working
# directory, which is what makes it findable without knowing its name.
main_repo_dir = Path.cwd()


def external_repo_dir(apparent_name):
  """Returns the runfiles directory of the external repository so named."""
  for path in sorted(runfiles_dir.iterdir()):
    if not path.is_dir():
      continue
    if path.name == apparent_name or path.name.endswith(
        ("~" + apparent_name, "+" + apparent_name)
    ):
      return path
  raise RuntimeError(
      f"no runfiles directory for @{apparent_name} in {runfiles_dir}"
  )


llvm_project_dir = external_repo_dir("llvm-project")
mlir_tools_path = llvm_project_dir.joinpath("mlir")

tool_paths = [
    mlir_tools_path,
    main_repo_dir.joinpath("tools"),
    llvm_project_dir.joinpath("llvm"),
]

config.environment["PATH"] = (
    ":".join(str(path) for path in tool_paths) + ":" + os.environ["PATH"]
)

substitutions = {
    "%mlir_lib_dir": str(mlir_tools_path),
    "%shlibext": ".so",
}

config.substitutions.extend(substitutions.items())


# CPU feature detection for platform-specific tests
def get_cpu_features():
  """Detect CPU features for conditional test execution."""
  features = set()
  system = platform.system()
  machine = platform.machine()

  if system == "Linux":
    if machine in ("x86_64", "i686", "i386"):
      # x86: read from /proc/cpuinfo
      try:
        with open("/proc/cpuinfo", "r") as f:
          for line in f:
            if line.startswith("flags"):
              flags = line.split(":")[1].strip().split()
              if "pclmulqdq" in flags:
                features.add("pclmulqdq")
              if "gfni" in flags:
                features.add("gfni")
              if "avx512f" in flags:
                features.add("avx512")
              break
      except (IOError, IndexError):
        pass
    elif machine in ("aarch64", "arm64"):
      # ARM: the "pmull" carryless-multiply lives behind the crypto/AES
      # extension, reported as "pmull" in /proc/cpuinfo's Features line.
      try:
        with open("/proc/cpuinfo", "r") as f:
          for line in f:
            if line.startswith("Features"):
              flags = line.split(":")[1].strip().split()
              if "pmull" in flags:
                features.add("pmull")
              break
      except (IOError, IndexError):
        pass
  elif system == "Darwin":
    if machine == "arm64":
      # macOS ARM: query the optional-feature flag via sysctl.
      try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm.FEAT_PMULL"],
            capture_output=True,
            text=True,
            check=False,
        )
        if out.stdout.strip() == "1":
          features.add("pmull")
      except (OSError, ValueError):
        pass

  # Add architecture features
  if machine in ("x86_64", "i686", "i386"):
    features.add("x86")
  elif machine in ("aarch64", "arm64"):
    features.add("arm")

  return features


# Add detected features to lit config
cpu_features = get_cpu_features()
for feature in cpu_features:
  config.available_features.add(feature)
