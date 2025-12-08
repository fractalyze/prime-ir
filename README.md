# ZKIR: Zero-Knowledge Intermediate Representation

[![CI](https://github.com/fractalyze/zkir/actions/workflows/ci.yml/badge.svg)](https://github.com/fractalyze/zkir/actions/workflows/ci.yml)

**ZKIR** is an Intermediate Representation (IR) designed specifically for
representing Zero-Knowledge (ZK) proving schemes, using
[MLIR (Multi-Level Intermediate Representation)](https://mlir.llvm.org/). The
core goal of ZKIR is to enable automatic domain-specific optimizations and
efficiently support diverse, heterogeneous proving backends without any
additional fine-tuning for targets.

## Motivation

ZK proving systems often require domain-specific arithmetic operations, such as
field multiplication or modular inversion. These operations are difficult to
express and optimize at the level of low-level IRs like LLVM IR.

In contrast, a high-level, ZK-specific IR preserves algebraic structure and
developer intent, providing the following advantages:

- **Domain-specific optimization**: MLIR allows us to design a domain-specific
  language (DSL) that preserves mathematical semantics. This enables
  simplifications such as $-(-x) = x$.

- **Hardware abstraction**: MLIR is designed with extensibility in mind and
  provides dedicated dialects for various hardware targets—including
  [NVGPU](https://mlir.llvm.org/docs/Dialects/NVGPU/),
  [SPIR-V](https://mlir.llvm.org/docs/Dialects/SPIR-V/), and
  [AMDGPU](https://mlir.llvm.org/docs/Dialects/AMDGPU/). This allows for a clean
  separation between algorithmic logic and backend-specific code generation.

  ZKIR builds on this advantage, allowing high-level computations and
  ZK-specific operations to be expressed independently of their execution
  environment. This enables:

  - **Retargeting the same IR** to multiple hardware backends (CPU, GPU, FPGA)
    without rewriting the logic.

  - **Fine-grained control over lowering strategies**, such as using different
    pipelining and tiling strategies depending on the target hardware.

  - **Future-proof integration** with ZK-dedicated accelerators or distributed
    proving platforms.

Instead of forcing ZK computation through a CPU-centric path, ZKIR makes it
possible to define **hardware-aware but backend-agnostic IR** that retains both
performance and correctness across a wide range of devices.

## Status

- ✅: Complete
- 🟡: In Progress
- ⚪: Not Yet Started

### [ModArith](/zkir/Dialect/ModArith/IR/ModArithOps.td)

- ✅ Fast Montgomery Multiplication
- ✅ Bernstein-Yang Batch Inverse
- 🟡 Specialized SIMD
  - ✅ AVX512
  - 🟡 ARM Neon
  - ⚪ AVX2
- ⚪ DataFlow Analysis
  - Range Analysis
  - Montgomery Conversion Analysis

### [Field](/zkir/Dialect/Field/IR/FieldOps.td)

- ✅ Prime Field Operations(Add, Double, Sub, Negate, Mul, Inv, Pow, ...)
- ⚪ Binary Field Operations
- 🟡 Extension Field Operations
  - ✅ Quadratic Extension Field Operations
  - ✅ Cubic Extension Field Operations
  - ✅ Quartic Extension Field Operations
  - 🟡 Quintic Extension Field Operations

### [TensorExt](/zkir/Dialect/TensorExt/IR/TensorExtOps.td)

- ✅ Bit-reverse Canonicalization

### [Elliptic Curve](/zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.td)

- ✅ Group Operations(Add, Double, Sub, Negate, ScalarMul, ...)
- ✅ MSM for CPU
- ⚪ MSM for GPU

### [Poly](/zkir/Dialect/Poly/IR/PolyOps.td)

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

1. Clone the ZKIR repo

   ```sh
   git clone https://github.com/fractalyze/zkir
   ```

1. Build ZKIR

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
   bazel run //tools:zkir-opt -- --canonicalize $(pwd)/negate.mlir
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

Building a substantial ZK compiler requires collaboration across the broader ZK
ecosystem — and we’d love your help in shaping ZKIR. See
[CONTRIBUTING.md](https://github.com/fractalyze/.github/blob/main/CONTRIBUTING.md)
for more details.

We use GitHub Issues and Pull Requests to coordinate development, and
longer-form discussions take place in the
[zkir-discuss](https://github.com/fractalyze/zkir/discussions).
