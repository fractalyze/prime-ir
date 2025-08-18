# ZKIR Style Guide

## Introduction

This document defines the coding standards for C++ code in ZKIR.
The base guideline is the [LLVM Coding Standard], combined with the [Angular Commit Convention], with explicit project-specific modifications.
In addition to code style, this guide incorporates our rules for commit messages, pull requests, and IDE/editor setup.

---

## Core Principles

- **Readability:** Both code and commits should be immediately understandable.
- **Maintainability:** Code should be easy to refactor and extend.
- **Consistency:** Apply the same conventions across files and modules, except where external code (e.g., XLA) is imported.
- **Performance:** Prioritize clarity, but optimize carefully where latency and cost are critical.

---

## C++ Coding Style

The following are project-specific deviations and clarifications from the [LLVM Coding Standard].

### Static Methods

- For **static methods** implemented in `.cc` files, explicitly annotate with `// static`.

  ```c++
  // static
  uint64_t EnvTime::NowNanos() {
    // ...
  }
  ```

### File-Scoped Symbols

- Wrap **file-scoped functions, constants, and variables** inside an **anonymous namespace**.

  ```c++
  namespace {

  constexpr int kBufferSize = 1024;

  void HelperFunction() {
    // ...
  }

  }  // namespace
  ```

### Abseil

- Prefer **`std::string_view`** instead of `absl::string_view`.

### Header Inclusion

- **Avoid redundant includes**: Do not repeat headers in `.cc` files that are already included in the corresponding `.h`.

  ```c++
  // in a.h
  #include <stdint.h>

  // in a.cc
  #include "a.h"
  // #include <stdint.h>  // ❌ redundant
  ```

- **Include only required headers**. Remove unused includes.

### Raw Pointer Ownership

- When using a **raw pointer** (`T*`) in **class or struct members**, explicitly document ownership by adding an inline comment `// not owned` or `// owned`.
- Prefer `std::unique_ptr` or `std::shared_ptr` for owned resources.

Example:

```c++
class Prover {
 public:
  explicit Prover(Context* ctx) : ctx_(ctx) {}

 private:
  Context* ctx_; // not owned
  std::unique_ptr<Engine> engine_;
};
```

---

## Comment Style

- Non-trivial code changes must be accompanied by comments.
- Comments should explain **why** a change or design decision was made, not just what the code does.
- Use full sentences with proper punctuation.

---

## Bazel Style

- Every header included in a Bazel target must also be declared as a Bazel dependency.

---

## Testing

- **Framework**: Use gtest/gmock.
- **Coverage**: New features must include tests whenever applicable.
- **Completeness**: Always include boundary cases and error paths.
- **Determinism**: Tests must be deterministic and runnable independently (no hidden state dependencies).
- **Performance**: Add benchmarks for performance-critical code paths when appropriate.

---

## Collaboration Rules

### Commits (Angular Commit Convention)

- Must follow the [Commit Message Guideline].
- Format:

  ```
  <type>(<scope>): <summary>
  ```

  where `type` ∈ {build, chore, ci, docs, feat, fix, perf, refactor, style, test}.

- Commit body: explain **why** the change was made (minimum 20 characters).
- Footer: record breaking changes, deprecations, and related issues/PRs.
- Each commit must include only **minimal, logically related changes**. Avoid mixing style fixes with functional changes.

### Pull Requests

- Follow the [Pull Request Guideline].
- Commits must be **atomic** and independently buildable/testable.
- Provide context and links (short SHA for external references).

### File Formatting

- Every file must end with a single newline.
- No trailing whitespace.
- No extra blank lines at EOF.

---

## Tooling

- **Formatter:** `clang-format` (LLVM preset with project overrides). Refer to the [.clang-format] file in the repo.
- **Linter:** `clang-tidy`.
- **Pre-commit hooks:** Recommended for enforcing format and lint locally.
- **CI:** All PRs must pass lint, format, and tests before merge.

[LLVM Coding Standard]: https://llvm.org/docs/CodingStandards.html
[Angular Commit Convention]: https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md
[Commit Message Guideline]: https://github.com/zk-rabbit/.github/blob/main/COMMIT_MESSAGE_GUIDELINE.md
[Pull Request Guideline]: https://github.com/zk-rabbit/.github/blob/main/PULL_REQUEST_GUIDELINE.md
[.clang-format]: /.clang-format
