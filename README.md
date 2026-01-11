# PrimeIR

[![CI](https://github.com/fractalyze/prime-ir/actions/workflows/ci.yml/badge.svg)](https://github.com/fractalyze/prime-ir/actions/workflows/ci.yml)

**PrimeIR** is an Intermediate Representation (IR) for cryptographic
computations built on
[MLIR (Multi-Level Intermediate Representation)](https://mlir.llvm.org/).
Originally developed for Zero-Knowledge (ZK) proving systems, PrimeIR has
evolved to support a broader range of cryptographic applications including
homomorphic encryption, digital signatures, and other prime field-based
protocols.

The core goal of PrimeIR is to enable automatic domain-specific optimizations
and efficiently support diverse, heterogeneous backends without any additional
fine-tuning for targets.

## Motivation

Cryptographic systems often require domain-specific arithmetic operations, such
as field multiplication or modular inversion. These operations are difficult to
express and optimize at the level of low-level IRs like LLVM IR.

In contrast, a high-level, cryptography-aware IR preserves algebraic structure
and developer intent, providing the following advantages:

- **Domain-specific optimization**: MLIR allows us to design a domain-specific
  language (DSL) that preserves mathematical semantics. This enables
  simplifications such as $-(-x) = x$.

- **Hardware abstraction**: MLIR is designed with extensibility in mind and
  provides dedicated dialects for various hardware targets—including
  [NVGPU](https://mlir.llvm.org/docs/Dialects/NVGPU/),
  [SPIR-V](https://mlir.llvm.org/docs/Dialects/SPIR-V/), and
  [AMDGPU](https://mlir.llvm.org/docs/Dialects/AMDGPU/). This allows for a clean
  separation between algorithmic logic and backend-specific code generation.

  PrimeIR builds on this advantage, allowing high-level computations and
  cryptographic operations to be expressed independently of their execution
  environment. This enables:

  - **Retargeting the same IR** to multiple hardware backends (CPU, GPU, FPGA)
    without rewriting the logic.

  - **Fine-grained control over lowering strategies**, such as using different
    pipelining and tiling strategies depending on the target hardware.

  - **Future-proof integration** with cryptographic accelerators or distributed
    computing platforms.

Instead of forcing cryptographic computation through a CPU-centric path, PrimeIR
makes it possible to define **hardware-aware but backend-agnostic IR** that
retains both performance and correctness across a wide range of devices.

## Status

- ✅: Complete
- 🟡: In Progress
- ⚪: Not Yet Started

### [ModArith](/prime_ir/Dialect/ModArith/IR/ModArithOps.td)

- ✅ Fast Montgomery Multiplication
- ✅ Bernstein-Yang Batch Inverse
- 🟡 Specialized SIMD
  - ✅ AVX512
  - 🟡 ARM Neon
  - ⚪ AVX2
- ⚪ DataFlow Analysis
  - Range Analysis
  - Montgomery Conversion Analysis

### [Field](/prime_ir/Dialect/Field/IR/FieldOps.td)

- ✅ Prime Field Operations(Add, Double, Sub, Negate, Mul, Inv, Pow, ...)
- ⚪ Binary Field Operations
- 🟡 Extension Field Operations
  - ✅ Quadratic Extension Field Operations
  - ✅ Cubic Extension Field Operations
  - ✅ Quartic Extension Field Operations
  - 🟡 Quintic Extension Field Operations

### [TensorExt](/prime_ir/Dialect/TensorExt/IR/TensorExtOps.td)

- ✅ Bit-reverse Canonicalization

### [Elliptic Curve](/prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.td)

- ✅ Group Operations(Add, Double, Sub, Negate, ScalarMul, ...)
- ✅ MSM

### [Poly](/prime_ir/Dialect/Poly/IR/PolyOps.td)

- ✅ NTT / INTT

## Prerequisite

1. Follow the [bazel installation guide](https://bazel.build/install).

1. Install optional dependencies for benchmark.

   - Ubuntu

   ```sh
   sudo apt install libomp-dev
   ```

   - Macos

   ```sh
   brew install libomp
   ```

## Build instructions

1. Clone the PrimeIR repo

   ```sh
   git clone https://github.com/fractalyze/prime-ir
   ```

1. Build PrimeIR

   ```sh
   bazel build //...
   ```

1. Run a test optimization:

   Create a test input file `negate.mlir`:

   ```sh
   cat > negate.mlir <<EOF
   !PF = !field.pf<11:i32>

   func.func @negate(%a: !PF) -> !PF  {
     %0 = field.negate %a : !PF
     %1 = field.negate %0 : !PF
     return %1 : !PF
   }
   EOF
   ```

   Run the optimizer:

   ```sh
   bazel run //tools:prime-ir-opt -- --canonicalize $(pwd)/negate.mlir
   ```

   Expected output:

   ```mlir
   !pf11 = !field.pf<11 : i32>
   module {
     func.func @negate(%arg0: !pf11) -> !pf11 {
       return %arg0 : !pf11
     }
   }
   ```

## Community

Building a substantial cryptographic compiler requires collaboration across the
broader ecosystem — and we'd love your help in shaping PrimeIR. See
[fractalyze/CONTRIBUTING.md](https://github.com/fractalyze/.github/blob/main/CONTRIBUTING.md)
for general guidelines, and refer to `CONTRIBUTING.md` in this repository if you
plan to update our vendored LLVM patches or need pointers on working with an
external LLVM checkout.

We use GitHub Issues and Pull Requests to coordinate development, and
longer-form discussions take place in
[Discussions](https://github.com/fractalyze/prime-ir/discussions).
