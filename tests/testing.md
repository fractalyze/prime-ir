# Testing Guide

## Overview

We use [Lit](https://llvm.org/docs/CommandGuide/lit.html) and
[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) for testing in
ZKIR.

Currently, we have 6 files denoting often-used definitions and functions:

- [default_print_utils.mlir](/tests/default_print_utils.mlir)
- [bn254_field_defs.mlir](/tests/bn254_field_defs.mlir)
- [bn254_ec_defs.mlir](/tests/bn254_ec_defs.mlir)
- [bn254_ec_utils.mlir](/tests/bn254_ec_utils.mlir)
- [bn254_ec_mont_defs.mlir](/tests/bn254_ec_mont_defs.mlir)
- [bn254_ec_mont_utils.mlir](/tests/bn254_ec_mont_utils.mlir)

Concatenate these to your test file as needed in your Lit commands. Note that
the order of concatenation is important!

```ex
         bn254_field_defs.mlir                      default_print_utils.mlir
                /     \
               ↙       ↘
bn254_ec_defs.mlir   bn254_ec_mont_defs.mlir
         |                      |
         ↓                      ↓
bn254_ec_utils.mlir  bn254_ec_mont_utils.mlir
```

## Examples

Here's a couple use case examples:

### Print Montgomery-form Elliptic Curve Points (runners)

```mlir
// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir \
// RUN:     %S/../../bn254_ec_mont_defs.mlir %S/../../bn254_ec_mont_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_bucket_acc -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../printI256%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_BUCKET_ACC < %t
```

### Print Standard-form Elliptic Curve Points (runners)

```mlir
// RUN: cat %S/../../default_print_utils.mlir %S/../../bn254_field_defs.mlir \
// RUN:     %S/../../bn254_ec_defs.mlir %S/../../bn254_ec_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_bucket_acc -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext,%S/../../printI256%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_BUCKET_ACC < %t
```

### Use Default Printing Utilities

```mlir
// RUN: cat %S/../../default_print_utils.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field -field-to-llvm \
// RUN:   | mlir-runner -e test_bucket_acc -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_TEST_BUCKET_ACC < %t
```

### Use BN254 Elliptic Curve Definitions

```mlir
// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_defs.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field \
// RUN:   | FileCheck %s -enable-var-scope
```

### Use BN254 Field Definitions

```mlir
// RUN: cat %S/../../bn254_field_defs.mlir %s \
// RUN:   | zkir-opt -elliptic-curve-to-field \
// RUN:   | FileCheck %s < %t
```
