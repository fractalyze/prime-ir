# PrimeIR Style Guide

## Introduction

This document defines the coding standards for C++ code in PrimeIR. The base
guideline is the [LLVM Coding Standard], combined with the
[Angular Commit Convention], with explicit project-specific modifications. In
addition to code style, this guide incorporates our rules for commit messages,
pull requests, and IDE/editor setup.

______________________________________________________________________

## Core Principles

- **Readability:** Both code and commits should be immediately understandable.
- **Maintainability:** Code should be easy to refactor and extend.
- **Consistency:** Apply the same conventions across files and modules, except
  where external code (e.g., XLA) is imported.
- **Performance:** Prioritize clarity, but optimize carefully where latency and
  cost are critical.

______________________________________________________________________

## C++ Coding Style

The following are project-specific deviations and clarifications from the
[LLVM Coding Standard].

### Naming guide

- **Type names** (including classes, structs, enums, typedefs, etc) should be
  nouns and start with an upper-case letter (e.g. `TextFileReader`).

- **Variable names** should be nouns (as they represent state). The name should
  be camel case, and start with a lower-case letter (e.g. `leader` or `boats`).
  (This is different from the reference.)

- **Function names** should be verb phrases (as they represent actions), and
  command-like function should be imperative. The name should be camel case, and
  start with a lowercase letter (e.g. `openFile()` or `isFoo()`).

- **Enum declarations** (e.g. `enum Foo {...}`) are types, so they should follow
  the naming conventions for types. A common use for enums is as a discriminator
  for a union, or an indicator of a subclass. When an enum is used for something
  like this, it should have a Kind suffix (e.g. `ValueKind`).

### MLIR Type Variable Naming

| Type                          | Variable Name   |
| ----------------------------- | --------------- |
| `ExtensionFieldTypeInterface` | `efType`        |
| `PrimeFieldType`              | `pfType`        |
| Base field type (`Type`)      | `baseFieldType` |

### Static Methods

- For **static methods** implemented in `.cpp` files, explicitly annotate with
  `// static`.

  ```c++
  // static
  uint64_t EnvTime::nowNanos() {
    // ...
  }
  ```

### File-Scoped Symbols

- Wrap **file-scoped functions, constants, and variables** inside an **anonymous
  namespace**.

  ```c++
  namespace {

  constexpr int kBufferSize = 1024;

  void helperFunction() {
    // ...
  }

  }  // namespace
  ```

### Abseil

- Prefer **`std::string_view`** instead of `absl::string_view`.

### Field/ModArith Type Accessors

When working with `ModArithType` or `PrimeFieldType`:

| Purpose              | Method                                  |
| -------------------- | --------------------------------------- |
| Storage type         | `getStorageType()`                      |
| Storage bit width    | `getStorageBitWidth()`                  |
| Arithmetic bit width | `getModulus().getValue().getBitWidth()` |

**Do NOT use `getModulus().getType()`** for storage type. For binary fields
GF(2ⁿ), the modulus 2ⁿ requires n+1 bits but storage only needs n bits.
`getStorageType()` handles this automatically.

```c++
// ✅ Storage: use getStorageType() / getStorageBitWidth()
unsigned bitWidth = fieldType.getStorageBitWidth();
APInt nVal(bitWidth, n);
IntegerAttr::get(fieldType.getStorageType(), nVal);

// ✅ Arithmetic: use getModulus().getValue().getBitWidth()
APInt modulus = fieldType.getModulus().getValue();
APInt result = n.urem(modulus);

// ❌ Bad: using getModulus().getType() for storage
IntegerAttr::get(fieldType.getModulus().getType(), value);  // Wrong!
```

### Header Inclusion

- **Avoid redundant includes**: Do not repeat headers in `.cc` files that are
  already included in the corresponding `.h`.

  ```c++
  // in a.h
  #include <stdint.h>

  // in a.cc
  #include "a.h"
  // #include <stdint.h>  // ❌ redundant
  ```

- **Include only required headers**. Remove unused includes.

### Raw Pointer Ownership

- When using a **raw pointer** (`T*`) in **class or struct members**, explicitly
  document ownership by adding an inline comment `// not owned` or `// owned`.
- Prefer `std::unique_ptr` or `std::shared_ptr` for owned resources.

Example:

```c++
class Prover {
 public:
  explicit Prover(Context* ctx) : ctx(ctx) {}

 private:
  Context* ctx; // not owned
  std::unique_ptr<Engine> engine;
};
```

______________________________________________________________________

## TableGen

This section defines standards for defining MLIR Dialects and Passes,
particularly focusing on the use of `dependentDialects` in TableGen (`.td`)
files.

### Dialect: `dependentDialects` Management

See [Dialect#dependent-dialects]

When defining a Dialect, the `dependentDialects` field is used to record
dependencies on other dialects whose components (Operations, Attributes, or
Types) are **reused, relied upon, or constructed** by the current Dialect
itself.

Example:

```
def MyDialect : Dialect {
  // Here we register the Arithmetic and Func dialect as dependencies of our `MyDialect`.
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect"
  ];
}
```

For every Dialect listed in the `dependentDialects` of a Dialect, the
corresponding C++ header file **must** be included in the Dialect definition
file.

```c++
// IWYU pragma: begin_keep
// Headers needed for FieldDialect.cpp.inc
#include "mlir/IR/OperationSupport.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithDialect.h"
// IWYU pragma: end_keep
```

### Pass: `dependentDialects` Management

See [PassManager#tablegen-sepcification]

The `dependentDialects` list in a `Pass` definition must only include Dialects
for which the Pass **introduces new entities** (Operations, Attributes, Types,
etc.) during its execution.

- **Rule:** The list should contain only Dialects whose entities are **newly
  created or explicitly used to construct a transformation** by the Pass.
- **Avoid:** Do not include Dialects that are merely consumed, transformed, or
  required for general Pass setup.
  - *Example:* If a Pass transforms `tensor` ops into `memref` ops, and does not
    create new `tensor` ops, `tensor::TensorDialect` should not be listed as a
    dependent dialect.

For every Dialect listed in the `dependentDialects` of a Pass, the corresponding
C++ header file **must** be included in the Pass definition file (e.g.,
`*Pass.h`).

Example(for `TensorExtToTensorPass`):

```c++
// IWYU pragma: begin_keep
// Headers needed for TensorExtToTensor.h.inc
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep
```

______________________________________________________________________

## Comment Style

- Non-trivial code changes must be accompanied by comments.
- Comments explain **why** a change or design decision was made or explain the
  code for better readability.
- Use full sentences with proper punctuation.

______________________________________________________________________

## Bazel Style

- Every header included in a Bazel target must also be declared as a Bazel
  dependency.

______________________________________________________________________

## Testing

- **Framework**: Use gtest/gmock.
- **Coverage**: New features must include tests whenever applicable.
- **Completeness**: Always include boundary cases and error paths.
- **Determinism**: Tests must be deterministic and runnable independently (no
  hidden state dependencies).
- **Performance**: Add benchmarks for performance-critical code paths when
  appropriate.

______________________________________________________________________

## Collaboration Rules

### Commits (Angular Commit Convention)

- Must follow the [Commit Message Guideline].

- Format:

  ```
  <type>(<scope>): <summary>
  ```

  where `type` ∈ {build, chore, ci, docs, feat, fix, perf, refactor, style,
  test}.

- Commit body: explain **why** the change was made (minimum 20 characters).

- Footer: record breaking changes, deprecations, and related issues/PRs.

- Each commit must include only **minimal, logically related changes**. Avoid
  mixing style fixes with functional changes.

### Pull Requests

- Follow the [Pull Request Guideline].
- Commits must be **atomic** and independently buildable/testable.
- Provide context and links (short SHA for external references).

### File Formatting

- Every file must end with a single newline.
- No trailing whitespace.
- No extra blank lines at EOF.

______________________________________________________________________

## Tooling

- **Formatter:** `clang-format` (LLVM preset with project overrides). Refer to
  the [.clang-format] file in the repo.
- **Linter:** `clang-tidy`.
- **Pre-commit hooks:** Recommended for enforcing format and lint locally.
- **CI:** All PRs must pass lint, format, and tests before merge.

______________________________________________________________________

## License

Every file (that could be exceptional case, such as empty BUILD.bazel) should
have license notice in the top.

### Date of Creation

- **New Files**: For any new files created from now on, the copyright year
  should be set to 2026.
- **Refactored Files**: If a file is moved or renamed as part of a refactoring
  process, you may retain the original creation year from the source file.

[.clang-format]: /.clang-format
[angular commit convention]: https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md
[commit message guideline]: https://github.com/fractalyze/.github/blob/main/COMMIT_MESSAGE_GUIDELINE.md
[dialect#dependent-dialects]: https://mlir.llvm.org/docs/DefiningDialects/#dependent-dialects
[llvm coding standard]: https://llvm.org/docs/CodingStandards.html
[passmanager#tablegen-sepcification]: https://mlir.llvm.org/docs/PassManagement/#tablegen-specification
[pull request guideline]: https://github.com/fractalyze/.github/blob/main/PULL_REQUEST_GUIDELINE.md
