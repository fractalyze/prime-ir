# Copyright 2025 The ZKIR Authors.
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
from pathlib import Path

from lit.formats import ShTest

config.name = "zkir"
config.test_format = ShTest()
config.suffixes = [".mlir", ".v"]

# lit executes relative to the directory
#
#   bazel-bin/tests/<test_target_name>.runfiles/zkir/
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
# Hence, to get lit to see tools like `zkir-opt`, we need to add the tools/
# subdirectory to the PATH environment variable.
#
# Bazel defines RUNFILES_DIR which includes zkir/ and third party dependencies
# as their own directory. Generally, it seems that $PWD == $RUNFILES_DIR/zkir/

runfiles_dir = Path(os.environ["RUNFILES_DIR"])

mlir_tools_relpath = "llvm-project/mlir"
mlir_tools_path = runfiles_dir.joinpath(Path(mlir_tools_relpath))

tool_relpaths = [
    mlir_tools_relpath,
    "_main/tools",
    "llvm-project/llvm",
]

config.environment["PATH"] = (
    ":".join(str(runfiles_dir.joinpath(Path(path))) for path in tool_relpaths)
    + ":"
    + os.environ["PATH"]
)

substitutions = {
    "%mlir_lib_dir": str(mlir_tools_path),
    "%shlibext": ".so",
}

config.substitutions.extend(substitutions.items())
