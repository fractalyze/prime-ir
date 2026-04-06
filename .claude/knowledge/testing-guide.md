# prime-ir Testing Guide

## Running Tests
```
bazel test //...
bazel test //tests/...
```

## Test Types
- LIT tests: MLIR pass verification (`.mlir` files)
- gtest: C++ unit tests (`*_test.cc`)

## Conventions
- Test files alongside source or in `tests/` subdirectory
- LIT tests use FileCheck for output verification
- Test both optimization correctness and algebraic identity preservation
