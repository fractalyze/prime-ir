# ZKIR: Zero-Knowledge Intermediate Representation

ZKIR is a set of high-level operations (HLOs) designed for zero-knowledge (ZK)
provers, inspired by [StableHLO](https://github.com/openxla/stablehlo). It
leverages
[MLIR (Multi-Level Intermediate Representation)](https://mlir.llvm.org/) to
enable domain-specific optimizations and support for heterogeneous proving
backends.

## Motivation

Zero-knowledge (ZK) proof systems often require domain-specific arithmetic
operations—such as field multiplication and modular inversion that are difficult
to express and optimize at low-level IRs like LLVM IR.

### Why not use LLVM IR?

LLVM IR is designed primarily for general-purpose CPU targets, which introduces
several limitations:

- **Lack of domain semantics**: Field operations like `-x` are reduced to
  generic integer operations. If the modulus is 11, the code will be as follows:

  ```mlir
   %0 = llvm.mlir.constant(11 : i32) : i32
   %1 = llvm.sub %0, %arg0 : i32
  ```

  This makes algebraic simplification and reasoning nearly impossible.

- **Loss of intent**: Without explicit semantic representation, higher-level
  optimizations or rewrites (e.g., canonicalizing double negations) become
  infeasible.

### Benefits of MLIR

In contrast, a high-level ZK-specific IR can preserve algebraic structure and
intent, enabling

- **Domain-specific optimization**. For example, simplifying `-(-x) = x` is
  trivial when the semantics are preserved. These kinds of rewrites are easy to
  express and safely apply at the IR level when working with well-defined field
  operations.

- **Hardware abstraction**: MLIR is designed with extensibility in mind and
  provides dedicated dialects for various hardware targets—including
  [NVGPU](https://mlir.llvm.org/docs/Dialects/NVGPU/),
  [SPIR-V](https://mlir.llvm.org/docs/Dialects/SPIR-V/), and
  [AMDGPU](https://mlir.llvm.org/docs/Dialects/AMDGPU/)—enabling a clean
  separation between algorithm logic and backend-specific code generation.

  ZKIR builds on this by allowing high-level computations and ZK-specific
  operations to be expressed independently of their execution environment. This
  enables:

  - Retargeting the same IR to multiple hardware backends (CPU, GPU, FPGA)
    without rewriting the logic.

  - Fine-grained control over lowering strategies—e.g., using different
    pipelining and tiling strategies depending on the target hardware.

  - Future integration with ZK-dedicated accelerators or distributed proving
    platforms.

Instead of forcing ZK computation through a CPU-centric path, ZKIR makes it
possible to define hardware-aware but backend-agnostic IR that retains both
performance and correctness across a wide range of devices.

## Vision

Currently, ZKIR defines only a subset of operations sufficient to represent part
of a proving scheme. Our long-term goal is to extend ZKIR to express arbitrary
proving schemes, enabling compiler-based optimization and hardware-accelerated
proving pipelines.

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
   git clone https://github.com/zk-rabbit/zkir
   ```

1. Build the ZKIR

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
ecosystem — and we’d love your help in shaping ZKIR.

We use GitHub Issues and Pull Requests to coordinate development, and
longer-form discussions take place in the
[zkir-discuss](https://github.com/zk-rabbit/zkir/discussions).
