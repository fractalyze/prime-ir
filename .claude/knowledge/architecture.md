# prime-ir Architecture

## Design Goals
- Domain-specific optimization preserving algebraic structure
- Hardware abstraction via MLIR's multi-level approach
- Retargeting same IR to CPU, GPU, FPGA without rewriting

## Lowering Pipeline
```
PrimeIR (high-level field ops)
  ↓ Domain-specific optimizations
PrimeIR (optimized)
  ↓ Hardware-specific lowering
LLVM IR / NVGPU / SPIR-V / AMDGPU
  ↓
Executable
```

## Key Dialects
- Field Dialect: finite field arithmetic primitives
- EllipticCurve Dialect: EC operations
- Integration with MLIR standard dialects (arith, linalg, etc.)

## Relationship to ZKX Pipeline
```
StableHLO → ZKIR → PrimeIR (this repo) → LLVM/GPU
```
