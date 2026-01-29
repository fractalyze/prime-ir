# ZKX: Zero Knowledge Accelerator

[![CI](https://github.com/fractalyze/zkx/actions/workflows/ci.yml/badge.svg)](https://github.com/fractalyze/zkx/actions/workflows/ci.yml)

ZKX is a computation framework inspired by
[XLA](https://github.com/openxla/xla). It compiles ZK-specific high-level
operations into efficient low-level code using
[PrimeIR](https://github.com/fractalyze/prime-ir) as its intermediate
representation. ZKX is optimized for CPUs and GPUs, with plans to extend support
to specialized ZK hardware for greater performance and portability.

## Prerequisite

1. Follow the [bazel installation guide](https://bazel.build/install).
1. (macOS) Apple Clang 17 or higher is recommended. Check your version with
   `clang --version`.

## Build instructions

1. Clone the ZKX repo

   ```sh
   git clone https://github.com/fractalyze/zkx
   ```

1. Build ZKX

   ```sh
   bazel build //...
   ```

1. Test ZKX

   ```sh
   bazel test //...
   ```

## Community

Building a substantial ZK compiler requires collaboration across the broader ZK
ecosystem — and we’d love your help in shaping ZKX. See
[CONTRIBUTING.md](https://github.com/fractalyze/.github/blob/main/CONTRIBUTING.md)
for more details.

We use GitHub Issues and Pull Requests to coordinate development, and
longer-form discussions take place in the
[zkx-discuss](https://github.com/fractalyze/zkx/discussions).

## Status

- ✅: **Complete**
- 🟡: In Progress
- ⚪: Not Yet Started

### Primitive Type

- ✅ Boolean
- ✅ Integer
- ⚪ Binary Field
- ✅ Koalabear
- ✅ Babybear
- ✅ Mersenne31
- ✅ Goldilocks
- ✅ Bn254

### HloPass

- ⚪: SPMD Partition
- ⚪: Algebraic Rewrite
- ⚪: Layout Assignment
- ⚪: Fusion

### Instruction for single machine

| HloOpcode              | CPU                     | GPU |
| ---------------------- | ----------------------- | --- |
| abs                    | ✅                      | ⚪  |
| add                    | ✅                      | ✅  |
| and                    | ✅                      | ⚪  |
| bitcast                | ✅                      | ⚪  |
| bitcast-convert        | ✅                      | ⚪  |
| broadcast              | ✅                      | ⚪  |
| call                   | ✅                      | ⚪  |
| clamp                  | ✅                      | ⚪  |
| count-leading-zeros    | ✅                      | ⚪  |
| compare                | ✅                      | ⚪  |
| concatenate            | ✅                      | ⚪  |
| conditional            | ✅                      | ⚪  |
| constant               | ✅                      | ✅  |
| convert                | ✅                      | ✅  |
| custom-call            | ✅                      | ⚪  |
| divide                 | ✅                      | ✅  |
| dot                    | ✅ (SpMV with CSR only) | ⚪  |
| dynamic-reshape        | ⚪                      | ⚪  |
| dynamic-slice          | ✅                      | ⚪  |
| dynamic-update-slice   | ✅                      | ⚪  |
| fusion                 | ✅                      | ✅  |
| fft                    | ✅                      | ⚪  |
| gather                 | 🟡                      | ⚪  |
| get-dimension-size     | ⚪                      | ⚪  |
| get-tuple-element      | ✅                      | ✅  |
| iota                   | ✅                      | ⚪  |
| inverse                | ✅                      | ⚪  |
| map                    | ✅                      | ⚪  |
| maximum                | ✅                      | ⚪  |
| minimum                | ✅                      | ⚪  |
| msm                    | ✅                      | ⚪  |
| multiply               | ✅                      | ✅  |
| negate                 | ✅                      | ✅  |
| not                    | ✅                      | ⚪  |
| or                     | ✅                      | ⚪  |
| pad                    | ✅                      | ⚪  |
| parameter              | ✅                      | ✅  |
| popcnt                 | ✅                      | ⚪  |
| power                  | ✅                      | ✅  |
| reduce                 | ✅                      | ⚪  |
| remainder              | ✅                      | ⚪  |
| reshape                | ✅                      | ⚪  |
| reverse                | ✅                      | ⚪  |
| scatter                | ✅                      | ⚪  |
| select                 | ✅                      | ⚪  |
| set-dimension-size     | ⚪                      | ⚪  |
| shift-left             | ✅                      | ⚪  |
| shift-right-arithmetic | ✅                      | ⚪  |
| shift-right-logical    | ✅                      | ⚪  |
| sign                   | ✅                      | ⚪  |
| slice                  | ✅                      | ✅  |
| sort                   | ✅                      | ⚪  |
| subtract               | ✅                      | ✅  |
| transpose              | ✅                      | ⚪  |
| tuple                  | ✅                      | ✅  |
| while                  | ✅                      | ⚪  |
| xor                    | ✅                      | ⚪  |
