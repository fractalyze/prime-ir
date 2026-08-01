# Copyright 2026 The PrimeIR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Derive and prove the tower<->flat basis constants in TowerFlatBasis.h.

The Fan-Paar tower (BinaryFieldCodeGen.cpp) is GF(2^(2^k)) =
subfield[X_k]/(X_k^2 + X_{k-1}*X_k + 1) with X_0 = 1; storage bit i of a
bf<k> encodes the multilinear monomial prod_j X_{j+1}^{i_j}. Every level
1..7 has a canonical flat basis GF(2)[y]/(f_k): level 3's flat basis IS
the AES basis (f_3 = x^8+x^4+x^3+x+1) and level 7's IS GHASH
(f_7 = x^128+x^7+x^2+x+1); the other levels use low-weight irreducibles
pinned here. The tower->flat isomorphism is GF(2)-linear — a bit-matrix
per level.

Proofs, per level:
  - f_k irreducibility (Rabin);
  - generator images satisfy the tower's defining relations
    B_j^2 + B_{j-1}*B_j + 1 = 0 — by the universal property of the
    iterated quadratic quotients this alone makes the induced map a field
    isomorphism (asserted during construction);
  - the multiplication homomorphism on ALL basis pairs for levels 1..5
    (complete by bilinearity), and on random samples for 6..7 where the
    exhaustive loop is slow and the construction proof already carries;
  - the two matrices are mutually inverse on every basis vector;
  - the clmad reduction schedules against plain long division: the
    single-limb two-fold schedule (mulFlatClmad) for n <= 32 and the
    two-limb schedule (product limbs + fold + spill fold) for n = 64.
    n = 128 uses the separately-proven GHASH reduction and is not
    re-proven here.

Prints the tables TowerFlatBasis.h carries: matrices for the widths with
C++ consumers (16/32 for the tower specialization and xla's additive-NTT
hoist, 64/128 for the wide tower specializations — 128-bit as lo/hi
uint64 pairs) and the per-level reduction constants. Levels 1..3 matrices
are derived and proven but not emitted: they have no consumer (the tower
recursion is cheaper than conversion at those widths), and this script
regenerates them on demand.

Run: python3 tools/derive_tower_flat_basis.py
"""

import random

# --- Tower arithmetic (mirrors BinaryFieldCodeGen::mulTower/mulXTower) ---


def mul_x_tower(a: int, level: int) -> int:
  """Multiply a (bf<level>) by the level generator X_level."""
  if level == 0:
    return a
  half = 1 << (level - 1)
  mask = (1 << half) - 1
  a0, a1 = a & mask, a >> half
  return a1 | ((a0 ^ mul_x_tower(a1, level - 1)) << half)


def mul_tower(a: int, b: int, level: int) -> int:
  if level == 0:
    return a & b
  half = 1 << (level - 1)
  mask = (1 << half) - 1
  a0, a1 = a & mask, a >> half
  b0, b1 = b & mask, b >> half
  m0 = mul_tower(a0, b0, level - 1)
  m1 = mul_tower(a1, b1, level - 1)
  m2 = mul_tower(a0 ^ a1, b0 ^ b1, level - 1)
  lo = m0 ^ m1
  hi = m2 ^ m0 ^ m1 ^ mul_x_tower(m1, level - 1)
  return lo | (hi << half)


# --- Flat arithmetic: GF(2)[y]/(f) with f given as an int bitmask ---

# Canonical flat moduli; is_irreducible() guards them in main(). Level 3 is
# pinned to the AES polynomial and level 7 to GHASH so `bf<3, flat>` /
# `bf<7, flat>` coincide with the existing named bases. Bit i = coeff of y^i.
FLAT_MODULUS = {
    1: (1 << 2) | (1 << 1) | 1,  # y^2+y+1
    2: (1 << 4) | (1 << 1) | 1,  # y^4+y+1
    3: (1 << 8) | (1 << 4) | (1 << 3) | (1 << 1) | 1,  # AES
    4: (1 << 16) | (1 << 5) | (1 << 3) | (1 << 1) | 1,  # y^16+y^5+y^3+y+1
    5: (1 << 32) | (1 << 7) | (1 << 3) | (1 << 2) | 1,  # y^32+y^7+y^3+y^2+1
    6: (1 << 64) | (1 << 4) | (1 << 3) | (1 << 1) | 1,  # y^64+y^4+y^3+y+1
    7: (1 << 128) | (1 << 7) | (1 << 2) | (1 << 1) | 1,  # GHASH
}


def f_low(level: int) -> int:
  """The modulus minus its leading term — the reduction fold constant."""
  return FLAT_MODULUS[level] ^ (1 << (1 << level))


def clmul(a: int, b: int) -> int:
  r = 0
  while b:
    if b & 1:
      r ^= a
    a <<= 1
    b >>= 1
  return r


def flat_reduce(a: int, level: int) -> int:
  f = FLAT_MODULUS[level]
  n = 1 << level
  for i in range(a.bit_length() - 1, n - 1, -1):
    if a >> i & 1:
      a ^= f << (i - n)
  return a


def mul_flat(a: int, b: int, level: int) -> int:
  return flat_reduce(clmul(a, b), level)


def mul_flat_clmad_schedule(a: int, b: int, level: int) -> int:
  """The single-limb fold schedule mulFlatClmad emits, in i64 semantics.

  p = clmul(a,b); t1 = p ^ clmul(p>>n, fLow); h2 = (t1>>n) ^ (p>>n);
  t2 = t1 ^ clmul(h2, fLow); result = trunc_n(t2). Every intermediate must
  fit 64 bits or clmad.lo would truncate information. Valid for n <= 32.
  """
  n = 1 << level
  low = f_low(level)
  p = clmul(a, b)
  hi = p >> n
  t1 = p ^ clmul(hi, low)
  h2 = (t1 >> n) ^ hi
  t2 = t1 ^ clmul(h2, low)
  assert max(p, t1, t2).bit_length() <= 64
  return t2 & ((1 << n) - 1)


def mul_flat64_clmad_schedule(a: int, b: int) -> int:
  """The two-limb schedule for n = 64 (5 clmads on hardware).

  (pLo, pHi) = 64x64 carry-less product limbs; fold pHi via
  y^64 == fLow, whose product spills at most deg(fLow) bits into a second
  limb, folded once more: r = pLo ^ lo(pHi*fLow) ^ lo(hi(pHi*fLow)*fLow).
  """
  low = f_low(6)
  mask = (1 << 64) - 1
  p = clmul(a, b)
  p_lo, p_hi = p & mask, p >> 64
  f1 = clmul(p_hi, low)
  r = p_lo ^ (f1 & mask)
  r ^= clmul(f1 >> 64, low) & mask
  return r


def pow_flat(a: int, e: int, level: int) -> int:
  r = 1
  while e:
    if e & 1:
      r = mul_flat(r, a, level)
    a = mul_flat(a, a, level)
    e >>= 1
  return r


def is_irreducible(f: int, n: int) -> bool:
  """Degree-n f is irreducible over GF(2) iff y^(2^n) = y mod f and
  gcd(y^(2^(n/p)) + y, f) = 1 for every prime p dividing n; n is a power
  of two here, so p = 2 is the only case."""

  def modmul(a, b):
    r = clmul(a, b)
    for i in range(r.bit_length() - 1, n - 1, -1):
      if r >> i & 1:
        r ^= f << (i - n)
    return r

  def y_pow_2exp(e):
    r = 2  # y
    for _ in range(e):
      r = modmul(r, r)
    return r

  if n == 1:
    return f in (0b10, 0b11)
  if y_pow_2exp(n) != 2:
    return False
  a, b = y_pow_2exp(n // 2) ^ 2, f
  while a:
    while a and a.bit_length() >= b.bit_length():
      a ^= b << (a.bit_length() - b.bit_length())
    a, b = b, a
    if b.bit_length() <= 1:
      break
  return b == 1


# --- Solve z^2 + z = c over the flat field (GF(2)-linear system) ---


def solve_artin_schreier(c: int, level: int) -> int:
  n = 1 << level
  cols = [mul_flat(1 << i, 1 << i, level) ^ (1 << i) for i in range(n)]
  basis = []  # (l, z) pairs, each l with a unique leading bit
  for i in range(n):
    l, z = cols[i], 1 << i
    while l:
      p = l.bit_length() - 1
      hit = next(
          (j for j, (bl, _) in enumerate(basis) if bl.bit_length() - 1 == p),
          None,
      )
      if hit is None:
        basis.append((l, z))
        break
      l ^= basis[hit][0]
      z ^= basis[hit][1]
  z = 0
  while c:
    p = c.bit_length() - 1
    hit = next(
        (j for j, (bl, _) in enumerate(basis) if bl.bit_length() - 1 == p), None
    )
    if hit is None:
      raise ValueError("z^2+z=c has no solution (trace(c) != 0)")
    c ^= basis[hit][0]
    z ^= basis[hit][1]
  return z


# --- Embedding: images B_j of the tower generators X_j in the flat field ---


def generator_images(level: int) -> list[int]:
  """B[j] = flat image of X_{j+1}, satisfying B_1^2 + B_1 + 1 = 0 and
  B_j^2 + B_{j-1}*B_j + 1 = 0. Substituting B = prev*z turns each
  quadratic into Artin-Schreier form z^2 + z = 1/prev^2. The relation
  asserts here are the load-bearing proof: images satisfying the tower's
  defining relations induce a field isomorphism by the universal property
  of the iterated quadratic quotients."""
  images = []
  prev = 1
  for _ in range(level):
    prev_sq = mul_flat(prev, prev, level)
    inv_prev_sq = pow_flat(prev_sq, (1 << (1 << level)) - 2, level)
    z = solve_artin_schreier(inv_prev_sq, level)
    b = mul_flat(prev, z, level)
    assert mul_flat(b, b, level) ^ mul_flat(prev, b, level) ^ 1 == 0
    images.append(b)
    prev = b
  return images


def basis_matrix(level: int) -> list[int]:
  """Column i of M = flat image of tower basis monomial i."""
  b = generator_images(level)
  cols = []
  for i in range(1 << level):
    v = 1
    for j in range(level):
      if i >> j & 1:
        v = mul_flat(v, b[j], level)
    cols.append(v)
  return cols


def apply_matrix(cols: list[int], x: int) -> int:
  r = 0
  for i, c in enumerate(cols):
    if x >> i & 1:
      r ^= c
  return r


def invert_matrix(cols: list[int], n: int) -> list[int]:
  basis = {}
  for i in range(n):
    l, z = cols[i], 1 << i
    while l:
      p = l.bit_length() - 1
      if p not in basis:
        basis[p] = (l, z)
        break
      l ^= basis[p][0]
      z ^= basis[p][1]
    else:
      raise ValueError("matrix not invertible")
  inv_cols = []
  for i in range(n):
    c, z = 1 << i, 0
    while c:
      p = c.bit_length() - 1
      bl, bz = basis[p]
      c ^= bl
      z ^= bz
    inv_cols.append(z)
  return inv_cols


def dump64(name: str, n: int, cols: list[int]) -> None:
  print(f"inline constexpr uint64_t {name}[{n}] = {{")
  for i in range(0, n, 4):
    print("    " + ", ".join(hex(c) for c in cols[i : i + 4]) + ",")
  print("};")


def dump128(name: str, cols: list[int]) -> None:
  mask = (1 << 64) - 1
  for suffix, shift in (("Lo", 0), ("Hi", 64)):
    print(f"inline constexpr uint64_t {name}{suffix}[128] = {{")
    for i in range(0, 128, 4):
      row = ", ".join(hex((c >> shift) & mask) for c in cols[i : i + 4])
      print(f"    {row},")
    print("};")


def prove_level(level: int) -> tuple[list[int], list[int]]:
  n = 1 << level
  f = FLAT_MODULUS[level]
  assert is_irreducible(f, n), f"modulus for level {level} not irreducible"
  cols = basis_matrix(level)
  inv = invert_matrix(cols, n)

  # Homomorphism: exhaustive basis pairs (complete by bilinearity) where
  # mul_tower is cheap; sampled at 6/7 where the construction proof in
  # generator_images already certifies the map.
  if level <= 5:
    pairs = [(1 << i, 1 << j) for i in range(n) for j in range(n)]
  else:
    rng = random.Random(392 + level)
    pairs = [(rng.getrandbits(n), rng.getrandbits(n)) for _ in range(200)]
    pairs += [(1 << i, 1 << (n - 1 - i)) for i in range(n)]
  for a, b in pairs:
    lhs = apply_matrix(cols, mul_tower(a, b, level))
    rhs = mul_flat(apply_matrix(cols, a), apply_matrix(cols, b), level)
    assert lhs == rhs, (level, a, b)
  for i in range(n):
    assert apply_matrix(inv, apply_matrix(cols, 1 << i)) == 1 << i

  # Reduction schedules against plain long division.
  rng = random.Random(392 + level)
  ops = (
      [(1 << i, 1 << j) for i in range(n) for j in range(n)] if n <= 32 else []
  )
  ops.append(((1 << n) - 1, (1 << n) - 1))
  ops += [(rng.getrandbits(n), rng.getrandbits(n)) for _ in range(1000)]
  if n <= 32:
    for a, b in ops:
      assert mul_flat_clmad_schedule(a, b, level) == mul_flat(a, b, level)
  elif n == 64:
    for a, b in ops:
      assert mul_flat64_clmad_schedule(a, b) == mul_flat(a, b, level)
  # n == 128 reduces via the separately-proven GHASH fold; not re-proven.
  return cols, inv


def main() -> None:
  emitted = {}
  for level in range(1, 8):
    emitted[level] = prove_level(level)
    print(
        f"// level {level} (n={1 << level}): f = {hex(FLAT_MODULUS[level])}"
        " proven"
    )

  print()
  lows = [0] + [f_low(level) for level in range(1, 8)]
  print("// BinaryFieldType::kCanonicalFlatModLow (FieldTypes.td) must equal:")
  print("//     " + ", ".join(hex(v) for v in lows))
  for level, name in ((4, "16"), (5, "32"), (6, "64")):
    cols, inv = emitted[level]
    dump64(f"kTowerToFlat{name}", 1 << level, cols)
    dump64(f"kFlatToTower{name}", 1 << level, inv)
  cols, inv = emitted[7]
  dump128("kTowerToFlat128", cols)
  dump128("kFlatToTower128", inv)


if __name__ == "__main__":
  main()
