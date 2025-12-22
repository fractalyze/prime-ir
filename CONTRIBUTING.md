For the general contribution guidelines see
[fractalyze/CONTRIBUTING](https://github.com/fractalyze/.github/blob/main/CONTRIBUTING.md).

## Updating LLVM upstream patches

When touching files under `third_party/llvm-project`, follow this process to
refresh the patches we carry in this repository.

1. Run `tools/setup_llvm_clone.sh /absolute/path/to/local/llvm-project` to clone
   the canonical revision (it creates that directory if it does not already
   exist) and apply the current `zkir` patches automatically. The script accepts
   an existing clone as well, in which case it just checks out the correct
   commit, resets/creates a local `zkir` branch, reapplies the patches, and
   creates one commit per patch for easier editing.

1. Point Bazel to the local clone while you iterate:

   - Uncomment the `http_archive(name = "llvm-raw", …)` stanza in
     `WORKSPACE.bazel`.

   - Uncomment the `new_local_repository` entry:

     ```
     new_local_repository(
         name = "llvm-raw",
         build_file_content = "# empty",
         path = "/absolute/path/to/your/llvm-project",
     )
     ```

1. Implement and test your LLVM changes (see “Managing multiple patches” if you
   are extending an existing patch).

1. Produce an updated patch and copy it back under `third_party/llvm-project/`.
   To generate the patch, run this:

   ```sh
   git show HEAD > /path/to/zkir/repo/third_party/llvm-project/<descriptive_name>.patch
   ```

### Managing multiple patches

- Keep a dedicated local branch (created automatically by the helper script) so
  each patch remains its own commit and can be edited independently.
- When work must happen “within” an existing patch:
  - Reset to the commit that represents the patch:
    `git reset --hard <applying-patch-commit>`.
  - Implement your changes and create a temporary commit:
    `git commit -m "whatever"`.
  - Fold it back into the original commit
    (`git rebase -i <applying-patch-commit>^` and choose `fixup`).
  - Regenerate the patch: `git show HEAD`.
