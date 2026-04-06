# CLAUDE.md

## Project Overview
PrimeIR is a cryptography-aware Intermediate Representation built on MLIR.
Enables domain-specific optimizations (e.g., -(-x) = x) and targets CPU, GPU,
FPGA backends. Middle stage of the ZKX compiler pipeline.

## Current Focus
Q2: E2E proving p99 ≤ 7s on 16 GPUs (excluding verification).
Sprint: E2E correctness — 5 test blocks Phase 1-3 bug-free.
Out of scope: Multi-zkVM 2nd backend, community building, internal tooling, external talks.

## Commands
- Build: `bazel build //...`
- Test all: `bazel test //...`
- Test suite: `bazel test //tests/...`

## Why Decisions
- MLIR over custom IR: leverage existing dialects (NVGPU, SPIR-V, AMDGPU) for hardware abstraction without reimplementing.
- Domain-specific algebraic optimizations: impossible at LLVM IR level where field structure is lost.
- PrimeField as degree-1 ExtensionField: unifies field operation handling, eliminates duplicate logic.

## Rules
- Do NOT modify lowering passes affecting cryptographic correctness without expert review.
- Do NOT call `BYInverter::Invert` without zero-initializing the output first.
- Do NOT check for zero AFTER calling inverse — check BEFORE (`inv(0) = 0` by ZK convention).
- Treat PrimeField as degree-1 ExtensionField to avoid duplicate handling logic.
- Always run `bazel test //...` before committing.

## Invisible Traps
- `inv(0) = 0` by ZK convention. zkir checks for zero before calling zk_dtypes, returns `ub.poison`. Forgetting the zero-check produces silently wrong field inversions.
- `BYInverter::Invert` returns false for non-invertible elements WITHOUT modifying output — if output isn't zero-initialized, you get garbage.
- Pipeline position: StableHLO → ZKIR → **PrimeIR** → LLVM/GPU. Algebraic identity optimizations here affect all downstream code generation.

## Knowledge Files
Read ONLY when relevant to your current task:
@.claude/knowledge/architecture.md — IR design, lowering pipeline
@.claude/knowledge/testing-guide.md — LIT test and gtest conventions
@.claude/knowledge/solutions.md — Past bug resolution patterns
