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

1. Implement and test your changes within the LLVM source tree.

1. Produce a patch file using your preferred method:

   - `git diff` (Unstaged changes)
   - `git diff --cached` (Staged changes)
   - `git show HEAD --pretty=""` (The most recent commit)

1. Copy the generated patch file into `third_party/llvm-project/`.

### Extending patches

- Keep a dedicated local branch (created automatically by the helper script) so
  each patch remains its own commit and can be edited independently.

- When work must happen “within” an existing patch:

  1. **Enter Edit Mode:** Locate the commit for the specific patch you want to
     update. Use interactive rebase:

     ```shell
     git rebase -i <target-patch-commit>^
     ```

     Then, mark the target commit as `edit`.

  1. **Apply Changes:** Make the necessary code modifications.

  1. **Stage Changes:**

     ```shell
     git add --update
     ```

  1. **Amend the Commit:**

     ```shell
     git commit --amend
     ```

  1. **Regenerate the Patch:** Overwrite the existing patch file in your project
     directory:

     ```shell
     git show HEAD --pretty="" > /path/to/zkir/third_party/llvm-project/<patch_name>.patch
     ```

  1. **Finalize Rebase:** Return to the current `HEAD`:

     ```shell
     git rebase --continue
     ```
