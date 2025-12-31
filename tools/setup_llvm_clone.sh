#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: setup_llvm_clone.sh <llvm-project-dir>

Provide the directory that should contain your local llvm-project clone.
The script will clone llvm-project into that directory if it does not
already exist, or reuse the existing clone if it does.
EOF
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
fi

dest_dir=$1
repo_root=$(cd -- "$(dirname "$0")/.." && pwd -P)

if [[ ! -f $repo_root/WORKSPACE.bazel ]]; then
    echo "error: script must be run from within the zkir repository" >&2
    exit 1
fi

if [[ ! -d "$dest_dir/.git" ]]; then
    mkdir -p "$dest_dir"
    git clone https://github.com/llvm/llvm-project.git "$dest_dir"
fi

llvm_commit=$(awk -F'"' '/LLVM_COMMIT/ {print $2; exit}' "$repo_root/third_party/llvm-project/workspace.bzl")

if [[ -z $llvm_commit ]]; then
    echo "error: unable to locate LLVM_COMMIT in third_party/llvm-project/workspace.bzl" >&2
    exit 1
fi

(
    cd "$dest_dir"
    git fetch origin "$llvm_commit"
    # Force checkout to discard any local changes and previously applied patches.
    git checkout -f "$llvm_commit" -B zkir
    # Remove any untracked files and directories.
    git clean -fdx
)

patch_dir="$repo_root/third_party/llvm-project"
shopt -s nullglob
# List of patches in the order specified in workspace.bzl (lines 23-37)
patches=(
    "$patch_dir/cuda_runtime.patch"
    "$patch_dir/kernel_outlining.patch"
    "$patch_dir/nvptx_lowering.patch"
    "$patch_dir/owning_memref_free.patch"
    "$patch_dir/owning_memref_memset.patch"
    "$patch_dir/linalg_type_support.patch"
    "$patch_dir/tensor_type_support.patch"
    "$patch_dir/vector_type_support.patch"
    "$patch_dir/memref_folding.patch"
    "$patch_dir/lazy_linking.patch"
)
shopt -u nullglob

if [[ ${#patches[@]} -eq 0 ]]; then
    echo "warning: no patches found under $patch_dir" >&2
    exit 0
fi

(
    cd "$dest_dir"
    for patch in "${patches[@]}"; do
        patch_name=$(basename "$patch")
        if ! git apply --check -p1 "$patch" >/dev/null 2>&1; then
            echo "error: patch '$patch_name' cannot be applied with -p1." >&2
            exit 1
        fi
        git apply --verbose -p1 "$patch"
        if [[ -n $(git status --porcelain) ]]; then
            git add -A
            git commit -m "Apply $patch_name"
        fi
    done
)
