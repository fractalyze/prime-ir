#pragma once

#include <cstdio>
#include <cstdint>
#include <cassert>
#include "icicle/utils/modifiers.h"
#include "icicle/math/storage.h"
#include "third_party/icicle/ptx.h"

namespace cuda_math {

  // Explicit-carry chain: replaces PTX carry-flag chaining (add.cc / addc)
  // with standard C++ widening arithmetic for clang CUDA compatibility.
  template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false>
  struct carry_chain {
    unsigned index;
    uint32_t carry;

    constexpr __device__ __forceinline__ carry_chain(uint32_t initial_carry = 0)
        : index(0), carry(CARRY_IN ? initial_carry : 0) {}

    __device__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT) {
        carry = 0;
        return ptx::add(x, y);
      }
      uint64_t sum = (uint64_t)x + y + carry;
      carry = (uint32_t)(sum >> 32);
      return (uint32_t)sum;
    }

    __device__ __forceinline__ uint32_t sub(const uint32_t x, const uint32_t y)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT) {
        carry = 0;
        return ptx::sub(x, y);
      }
      uint64_t diff = (uint64_t)x - y - carry;
      carry = (diff >> 63) ? 1u : 0u;
      return (uint32_t)diff;
    }

    __device__ __forceinline__ uint32_t mad_lo(const uint32_t x, const uint32_t y, const uint32_t z)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT) {
        carry = 0;
        return ptx::mad_lo(x, y, z);
      }
      // mad.lo.u32: d = lo(x * y) + z
      uint64_t prod_lo = (uint32_t)((uint64_t)x * y);
      uint64_t sum = prod_lo + (uint64_t)z + carry;
      carry = (uint32_t)(sum >> 32);
      return (uint32_t)sum;
    }

    __device__ __forceinline__ uint32_t mad_hi(const uint32_t x, const uint32_t y, const uint32_t z)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT) {
        carry = 0;
        return ptx::mad_hi(x, y, z);
      }
      // mad.hi.u32: d = hi(x * y) + z
      uint64_t prod_hi = (uint32_t)(((uint64_t)x * y) >> 32);
      uint64_t sum = prod_hi + (uint64_t)z + carry;
      carry = (uint32_t)(sum >> 32);
      return (uint32_t)sum;
    }
  };

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr __device__ __forceinline__ uint32_t add_sub_u32(const uint32_t* x, const uint32_t* y, uint32_t* r)
  {
    uint32_t carry = 0;
    for (unsigned i = 0; i < NLIMBS; i++) {
      if (SUBTRACT) {
        uint64_t diff = (uint64_t)x[i] - y[i] - carry;
        r[i] = (uint32_t)diff;
        carry = (diff >> 63) ? 1u : 0u;
      } else {
        uint64_t sum = (uint64_t)x[i] + y[i] + carry;
        r[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
    }
    if (!CARRY_OUT) return 0;
    return carry;
  }

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT, bool IS_U32 = true>
  static constexpr __device__ __forceinline__ uint32_t
  add_sub_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    return add_sub_u32<NLIMBS, SUBTRACT, CARRY_OUT>(x, y, r);
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    UNROLL
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  mul_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS, size_t start_i = 0)
  {
    UNROLL
    for (size_t i = start_i; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  // cmad_n: chained multiply-add with carry propagation.
  // Returns the final carry out.
  template <unsigned NLIMBS, bool CARRY_IN = false>
  static __device__ __forceinline__ uint32_t
  cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS, uint32_t optional_carry = 0)
  {
    uint32_t carry = 0;
    if (CARRY_IN) {
      // Original: add_cc(UINT32_MAX, optional_carry) sets carry if optional_carry > 0
      carry = (optional_carry > 0) ? 1u : 0u;
    }

    // acc[0] = lo(a[0] * bi) + acc[0] + carry
    {
      uint64_t prod_lo = (uint32_t)((uint64_t)a[0] * bi);
      uint64_t sum = prod_lo + (uint64_t)acc[0] + carry;
      acc[0] = (uint32_t)sum;
      carry = (uint32_t)(sum >> 32);
    }
    // acc[1] = hi(a[0] * bi) + acc[1] + carry
    {
      uint64_t prod_hi = (uint32_t)(((uint64_t)a[0] * bi) >> 32);
      uint64_t sum = prod_hi + (uint64_t)acc[1] + carry;
      acc[1] = (uint32_t)sum;
      carry = (uint32_t)(sum >> 32);
    }

    UNROLL
    for (size_t i = 2; i < n; i += 2) {
      {
        uint64_t prod_lo = (uint32_t)((uint64_t)a[i] * bi);
        uint64_t sum = prod_lo + (uint64_t)acc[i] + carry;
        acc[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
      {
        uint64_t prod_hi = (uint32_t)(((uint64_t)a[i] * bi) >> 32);
        uint64_t sum = prod_hi + (uint64_t)acc[i + 1] + carry;
        acc[i + 1] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
    }
    return carry;
  }

  template <unsigned NLIMBS, bool EVEN_PHASE>
  static __device__ __forceinline__ uint32_t cmad_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    uint32_t carry = 0;
    if (EVEN_PHASE) {
      // acc[0] = lo(a[0] * bi) + acc[0]
      {
        uint64_t prod_lo = (uint32_t)((uint64_t)a[0] * bi);
        uint64_t sum = prod_lo + (uint64_t)acc[0];
        acc[0] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
      // acc[1] = hi(a[0] * bi) + acc[1] + carry
      {
        uint64_t prod_hi = (uint32_t)(((uint64_t)a[0] * bi) >> 32);
        uint64_t sum = prod_hi + (uint64_t)acc[1] + carry;
        acc[1] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
    } else {
      // acc[1] = hi(a[0] * bi) + acc[1]
      {
        uint64_t prod_hi = (uint32_t)(((uint64_t)a[0] * bi) >> 32);
        uint64_t sum = prod_hi + (uint64_t)acc[1];
        acc[1] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
    }

    UNROLL
    for (size_t i = 2; i < n; i += 2) {
      {
        uint64_t prod_lo = (uint32_t)((uint64_t)a[i] * bi);
        uint64_t sum = prod_lo + (uint64_t)acc[i] + carry;
        acc[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
      {
        uint64_t prod_hi = (uint32_t)(((uint64_t)a[i] * bi) >> 32);
        uint64_t sum = prod_hi + (uint64_t)acc[i + 1] + carry;
        acc[i + 1] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
    }
    return carry;
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ void cmad_n_lsb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    uint32_t carry = 0;

    if (n > 1) {
      // acc[0] = lo(a[0] * bi) + acc[0]
      uint64_t prod_lo = (uint32_t)((uint64_t)a[0] * bi);
      uint64_t sum = prod_lo + (uint64_t)acc[0];
      acc[0] = (uint32_t)sum;
      carry = (uint32_t)(sum >> 32);
    } else {
      acc[0] = ptx::mad_lo(a[0], bi, acc[0]);
      return;
    }

    size_t i;
    UNROLL
    for (i = 1; i < n - 1; i += 2) {
      // acc[i] = hi(a[i-1] * bi) + acc[i] + carry
      {
        uint64_t prod_hi = (uint32_t)(((uint64_t)a[i - 1] * bi) >> 32);
        uint64_t sum = prod_hi + (uint64_t)acc[i] + carry;
        acc[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
      if (i == n - 2) {
        // last: no carry out needed
        uint64_t prod_lo = (uint32_t)((uint64_t)a[i + 1] * bi);
        acc[i + 1] = (uint32_t)(prod_lo + (uint64_t)acc[i + 1] + carry);
        carry = 0;
      } else {
        uint64_t prod_lo = (uint32_t)((uint64_t)a[i + 1] * bi);
        uint64_t sum = prod_lo + (uint64_t)acc[i + 1] + carry;
        acc[i + 1] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
    }
    if (i == n - 1) {
      // acc[i] = hi(a[i-1] * bi) + acc[i] + carry (no carry out)
      uint64_t prod_hi = (uint32_t)(((uint64_t)a[i - 1] * bi) >> 32);
      acc[i] = (uint32_t)(prod_hi + (uint64_t)acc[i] + carry);
    }
  }

  template <unsigned NLIMBS, bool CARRY_OUT = false, bool CARRY_IN = false>
  static __device__ __forceinline__ uint32_t mad_row(
    uint32_t* odd,
    uint32_t* even,
    const uint32_t* a,
    uint32_t bi,
    size_t n = NLIMBS,
    uint32_t ci = 0,
    uint32_t di = 0,
    uint32_t carry_for_high = 0,
    uint32_t carry_for_low = 0)
  {
    uint32_t odd_carry = cmad_n<NLIMBS, CARRY_IN>(odd, a + 1, bi, n - 2, carry_for_low);

    // odd[n-2] = lo(a[n-1] * bi) + ci + odd_carry
    {
      uint64_t prod_lo = (uint32_t)((uint64_t)a[n - 1] * bi);
      uint64_t sum = prod_lo + (uint64_t)ci + odd_carry;
      odd[n - 2] = (uint32_t)sum;
      odd_carry = (uint32_t)(sum >> 32);
    }
    // odd[n-1] = hi(a[n-1] * bi) + di + odd_carry
    {
      uint64_t prod_hi = (uint32_t)(((uint64_t)a[n - 1] * bi) >> 32);
      uint64_t sum = prod_hi + (uint64_t)di + odd_carry;
      odd[n - 1] = (uint32_t)sum;
      odd_carry = (uint32_t)(sum >> 32);
    }
    uint32_t cr = CARRY_OUT ? odd_carry : 0;

    uint32_t even_carry = cmad_n<NLIMBS>(even, a, bi, n);
    if (CARRY_OUT) {
      uint64_t sum = (uint64_t)odd[n - 1] + carry_for_high + even_carry;
      odd[n - 1] = (uint32_t)sum;
      cr += (uint32_t)(sum >> 32);
    } else {
      odd[n - 1] += carry_for_high + even_carry;
    }
    return cr;
  }

  template <unsigned NLIMBS, bool EVEN_PHASE>
  static __device__ __forceinline__ void
  mad_row_msb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    uint32_t odd_carry = cmad_n_msb<NLIMBS, !EVEN_PHASE>(odd, EVEN_PHASE ? a : (a + 1), bi, n - 2);

    // madc_lo_cc: lo(a[n-1] * bi) + 0 + odd_carry
    unsigned idx1 = EVEN_PHASE ? (n - 1) : (n - 2);
    {
      uint64_t prod_lo = (uint32_t)((uint64_t)a[n - 1] * bi);
      uint64_t sum = prod_lo + odd_carry;
      odd[idx1] = (uint32_t)sum;
      odd_carry = (uint32_t)(sum >> 32);
    }
    // madc_hi (terminal): hi(a[n-1] * bi) + 0 + odd_carry
    unsigned idx2 = EVEN_PHASE ? n : (n - 1);
    {
      uint64_t prod_hi = (uint32_t)(((uint64_t)a[n - 1] * bi) >> 32);
      odd[idx2] = (uint32_t)(prod_hi + odd_carry);
    }

    uint32_t even_carry = cmad_n_msb<NLIMBS, EVEN_PHASE>(even, EVEN_PHASE ? (a + 1) : a, bi, n - 1);
    odd[idx2] += even_carry;
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  mad_row_lsb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    if (bi != 0) {
      if (n > 1) cmad_n_lsb<NLIMBS>(odd, a + 1, bi, n - 1);
      cmad_n_lsb<NLIMBS>(even, a, bi, n);
    }
    return;
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ uint32_t
  mul_n_and_add(uint32_t* acc, const uint32_t* a, const uint32_t bi, const uint32_t* extra, size_t n = (NLIMBS >> 1))
  {
    uint32_t carry = 0;

    // acc[0] = lo(a[0] * bi) + extra[0]
    {
      uint64_t prod_lo = (uint32_t)((uint64_t)a[0] * bi);
      uint64_t sum = prod_lo + (uint64_t)extra[0];
      acc[0] = (uint32_t)sum;
      carry = (uint32_t)(sum >> 32);
    }

    UNROLL
    for (size_t i = 1; i < n - 1; i += 2) {
      // acc[i] = hi(a[i-1] * bi) + extra[i] + carry
      {
        uint64_t prod_hi = (uint32_t)(((uint64_t)a[i - 1] * bi) >> 32);
        uint64_t sum = prod_hi + (uint64_t)extra[i] + carry;
        acc[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
      // acc[i+1] = lo(a[i+1] * bi) + extra[i+1] + carry
      {
        uint64_t prod_lo = (uint32_t)((uint64_t)a[i + 1] * bi);
        uint64_t sum = prod_lo + (uint64_t)extra[i + 1] + carry;
        acc[i + 1] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
      }
    }

    // acc[n-1] = hi(a[n-2] * bi) + extra[n-1] + carry
    {
      uint64_t prod_hi = (uint32_t)(((uint64_t)a[n - 2] * bi) >> 32);
      uint64_t sum = prod_hi + (uint64_t)extra[n - 1] + carry;
      acc[n - 1] = (uint32_t)sum;
      carry = (uint32_t)(sum >> 32);
    }
    return carry;
  }

  /**
   * This method multiplies `a` and `b` (both assumed to have NLIMBS / 2 limbs) and adds `in1` and `in2` (NLIMBS limbs
   * each) to the result which is written to `even`.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  multiply_and_add_short_raw(const uint32_t* a, const uint32_t* b, uint32_t* even, uint32_t* in1, uint32_t* in2)
  {
    __align__(16) uint32_t odd[NLIMBS - 2];
    uint32_t first_row_carry = mul_n_and_add<NLIMBS>(even, a, b[0], in1);
    uint32_t carry = mul_n_and_add<NLIMBS>(odd, a + 1, b[0], &in2[1]);

    size_t i;
    UNROLL
    for (i = 2; i < ((NLIMBS >> 1) - 1); i += 2) {
      carry = mad_row<NLIMBS, true, false>(
        &even[i], &odd[i - 2], a, b[i - 1], NLIMBS >> 1, in1[(NLIMBS >> 1) + i - 2], in1[(NLIMBS >> 1) + i - 1], carry);
      carry = mad_row<NLIMBS, true, false>(
        &odd[i], &even[i], a, b[i], NLIMBS >> 1, in2[(NLIMBS >> 1) + i - 1], in2[(NLIMBS >> 1) + i], carry);
    }
    mad_row<NLIMBS, false, true>(
      &even[NLIMBS >> 1], &odd[(NLIMBS >> 1) - 2], a, b[(NLIMBS >> 1) - 1], NLIMBS >> 1, in1[NLIMBS - 2],
      in1[NLIMBS - 1], carry, first_row_carry);

    // merge |even| and |odd| plus the parts of `in2` we haven't added yet (first and last limbs)
    uint32_t merge_carry = 0;
    {
      uint64_t sum = (uint64_t)even[0] + in2[0];
      even[0] = (uint32_t)sum;
      merge_carry = (uint32_t)(sum >> 32);
    }
    for (i = 0; i < (NLIMBS - 2); i++) {
      uint64_t sum = (uint64_t)even[i + 1] + odd[i] + merge_carry;
      even[i + 1] = (uint32_t)sum;
      merge_carry = (uint32_t)(sum >> 32);
    }
    even[i + 1] += in2[i + 1] + merge_carry;
  }

  /**
   * This method multiplies `a` and `b` and writes the result into `even`. It assumes that `a` and `b` are NLIMBS/2
   * limbs long. The usual schoolbook algorithm is used.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void multiply_short_raw(const uint32_t* a, const uint32_t* b, uint32_t* even)
  {
    __align__(16) uint32_t odd[NLIMBS - 2];
    mul_n<NLIMBS>(even, a, b[0], NLIMBS >> 1);
    mul_n<NLIMBS>(odd, a + 1, b[0], NLIMBS >> 1);
    mad_row<NLIMBS>(&even[2], &odd[0], a, b[1], NLIMBS >> 1);

    size_t i;
    UNROLL
    for (i = 2; i < ((NLIMBS >> 1) - 1); i += 2) {
      mad_row<NLIMBS>(&odd[i], &even[i], a, b[i], NLIMBS >> 1);
      mad_row<NLIMBS>(&even[i + 2], &odd[i], a, b[i + 1], NLIMBS >> 1);
    }
    // merge |even| and |odd|
    uint32_t merge_carry = 0;
    {
      uint64_t sum = (uint64_t)even[1] + odd[0];
      even[1] = (uint32_t)sum;
      merge_carry = (uint32_t)(sum >> 32);
    }
    for (i = 1; i < NLIMBS - 2; i++) {
      uint64_t sum = (uint64_t)even[i + 1] + odd[i] + merge_carry;
      even[i + 1] = (uint32_t)sum;
      merge_carry = (uint32_t)(sum >> 32);
    }
    even[i + 1] += merge_carry;
  }

  /**
   * This method multiplies `as` and `bs` and writes the (wide) result into `rs`.
   * Implements subtractive Karatsuba.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  multiply_raw(const storage<NLIMBS>& as, const storage<NLIMBS>& bs, storage<2 * NLIMBS>& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* r = rs.limbs;
    if constexpr (NLIMBS > 2) {
      multiply_short_raw<NLIMBS>(a, b, r);
      multiply_short_raw<NLIMBS>(&a[NLIMBS >> 1], &b[NLIMBS >> 1], &r[NLIMBS]);
      __align__(16) uint32_t middle_part[NLIMBS];
      __align__(16) uint32_t diffs[NLIMBS];
      uint32_t carry1 = add_sub_u32<(NLIMBS >> 1), true, true>(&a[NLIMBS >> 1], a, diffs);
      uint32_t carry2 = add_sub_u32<(NLIMBS >> 1), true, true>(b, &b[NLIMBS >> 1], &diffs[NLIMBS >> 1]);
      multiply_and_add_short_raw<NLIMBS>(diffs, &diffs[NLIMBS >> 1], middle_part, r, &r[NLIMBS]);
      if (carry1)
        add_sub_u32<(NLIMBS >> 1), true, false>(
          &middle_part[NLIMBS >> 1], &diffs[NLIMBS >> 1], &middle_part[NLIMBS >> 1]);
      if (carry2) add_sub_u32<(NLIMBS >> 1), true, false>(&middle_part[NLIMBS >> 1], diffs, &middle_part[NLIMBS >> 1]);
      // Add middle part to result
      uint32_t mid_carry = add_sub_u32<NLIMBS, false, true>(&r[NLIMBS >> 1], middle_part, &r[NLIMBS >> 1]);

      // Propagate carry to highest limbs
      for (size_t i = NLIMBS + (NLIMBS >> 1); i < 2 * NLIMBS; i++) {
        uint64_t sum = (uint64_t)r[i] + mid_carry;
        r[i] = (uint32_t)sum;
        mid_carry = (uint32_t)(sum >> 32);
      }
    } else if (NLIMBS == 2) {
      auto a_128b = static_cast<__uint128_t>(*(uint64_t*)(as.limbs));
      auto b_64b = *(uint64_t*)bs.limbs;
      __uint128_t r_128b = a_128b * b_64b;
      r[0] = r_128b;
      r[1] = r_128b >> 32;
      r[2] = r_128b >> 64;
      r[3] = r_128b >> 96;

    } else if (NLIMBS == 1) {
      r[0] = ptx::mul_lo(a[0], b[0]);
      r[1] = ptx::mul_hi(a[0], b[0]);
    }
  }

  /**
   * A function that computes wide product that's correct for the higher NLIMBS + 1 limbs with a small maximum error.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  multiply_msb_raw(const storage<NLIMBS>& as, const storage<NLIMBS>& bs, storage<2 * NLIMBS>& rs)
  {
    if constexpr (NLIMBS > 1) {
      const uint32_t* a = as.limbs;
      const uint32_t* b = bs.limbs;
      uint32_t* even = rs.limbs;
      __align__(16) uint32_t odd[2 * NLIMBS - 2];

      even[NLIMBS - 1] = ptx::mul_hi(a[NLIMBS - 2], b[0]);
      odd[NLIMBS - 2] = ptx::mul_lo(a[NLIMBS - 1], b[0]);
      odd[NLIMBS - 1] = ptx::mul_hi(a[NLIMBS - 1], b[0]);
      size_t i;
      UNROLL
      for (i = 2; i < NLIMBS - 1; i += 2) {
        mad_row_msb<NLIMBS, true>(&even[NLIMBS - 2], &odd[NLIMBS - 2], &a[NLIMBS - i - 1], b[i - 1], i + 1);
        mad_row_msb<NLIMBS, false>(&odd[NLIMBS - 2], &even[NLIMBS - 2], &a[NLIMBS - i - 2], b[i], i + 2);
      }
      mad_row<NLIMBS>(&even[NLIMBS], &odd[NLIMBS - 2], a, b[NLIMBS - 1]);

      // merge |even| and |odd|
      uint32_t merge_carry = 0;
      {
        uint64_t sum = (uint64_t)even[NLIMBS - 1] + odd[NLIMBS - 2];
        even[NLIMBS - 1] = (uint32_t)sum;
        merge_carry = (uint32_t)(sum >> 32);
      }
      for (i = NLIMBS - 1; i < 2 * NLIMBS - 2; i++) {
        uint64_t sum = (uint64_t)even[i + 1] + odd[i] + merge_carry;
        even[i + 1] = (uint32_t)sum;
        merge_carry = (uint32_t)(sum >> 32);
      }
      even[i + 1] += merge_carry;
    } else {
      multiply_raw<NLIMBS>(as, bs, rs);
    }
  }

  /**
   * A function that computes the low half of the fused multiply-and-add.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void multiply_and_add_lsb_neg_modulus_raw(
    const storage<NLIMBS>& as, const storage<NLIMBS>& neg_mod, const storage<NLIMBS>& cs, storage<NLIMBS>& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = neg_mod.limbs;
    const uint32_t* c = cs.limbs;
    uint32_t* even = rs.limbs;

    if constexpr (NLIMBS > 2) {
      __align__(16) uint32_t odd[NLIMBS - 1];
      size_t i;
      if (b[0] == UINT32_MAX) {
        add_sub_u32<NLIMBS, true, false>(c, a, even);
        for (i = 0; i < NLIMBS - 1; i++)
          odd[i] = a[i];
      } else {
        mul_n_and_add<NLIMBS>(even, a, b[0], c, NLIMBS);
        mul_n<NLIMBS>(odd, a + 1, b[0], NLIMBS - 1);
      }
      mad_row_lsb<NLIMBS>(&even[2], &odd[0], a, b[1], NLIMBS - 1);
      UNROLL
      for (i = 2; i < NLIMBS - 1; i += 2) {
        mad_row_lsb<NLIMBS>(&odd[i], &even[i], a, b[i], NLIMBS - i);
        mad_row_lsb<NLIMBS>(&even[i + 2], &odd[i], a, b[i + 1], NLIMBS - i - 1);
      }

      // merge |even| and |odd|
      uint32_t merge_carry = 0;
      {
        uint64_t sum = (uint64_t)even[1] + odd[0];
        even[1] = (uint32_t)sum;
        merge_carry = (uint32_t)(sum >> 32);
      }
      for (i = 1; i < NLIMBS - 2; i++) {
        uint64_t sum = (uint64_t)even[i + 1] + odd[i] + merge_carry;
        even[i + 1] = (uint32_t)sum;
        merge_carry = (uint32_t)(sum >> 32);
      }
      even[i + 1] += odd[i] + merge_carry;
    } else if (NLIMBS == 2) {
      uint64_t res = *(uint64_t*)a * *(uint64_t*)b + *(uint64_t*)c;
      even[0] = res;
      even[1] = res >> 32;

    } else if (NLIMBS == 1) {
      even[0] = ptx::mad_lo(a[0], b[0], c[0]);
    }
  }

  /**
   * @brief Return upper/lower half of x (into r).
   */
  template <unsigned NLIMBS, bool HIGHER = false>
  static __device__ __forceinline__ void get_half_32(const uint32_t* x, uint32_t* r)
  {
    for (unsigned i = 0; i < NLIMBS; i++) {
      r[i] = x[HIGHER ? i + NLIMBS : i];
    }
  }

  template <unsigned NLIMBS, unsigned SLACK_BITS>
  static constexpr __device__ __forceinline__ void
  get_higher_with_slack(const storage<2 * NLIMBS>& xs, storage<NLIMBS>& out)
  {
    UNROLL
    for (unsigned i = 0; i < NLIMBS; i++) {
      out.limbs[i] = __funnelshift_lc(xs.limbs[i + NLIMBS - 1], xs.limbs[i + NLIMBS], 2 * SLACK_BITS);
    }
  }

  /**
   * Barrett reduction.
   */
  template <unsigned NLIMBS, unsigned SLACK_BITS, unsigned NOF_REDUCTIONS>
  static constexpr __device__ __forceinline__ storage<NLIMBS> barrett_reduce(
    const storage<2 * NLIMBS>& xs,
    const storage<NLIMBS>& ms,
    const storage<NLIMBS>& mod1,
    const storage<NLIMBS>& mod2,
    const storage<NLIMBS>& neg_mod)
  {
    storage<2 * NLIMBS> l = {};
    storage<NLIMBS> r = {};

    storage<NLIMBS> xs_hi = {};
    get_higher_with_slack<NLIMBS, SLACK_BITS>(xs, xs_hi);
    multiply_msb_raw<NLIMBS>(xs_hi, ms, l);
    storage<NLIMBS> l_hi = {};
    storage<NLIMBS> xs_lo = {};
    get_half_32<NLIMBS, true>(l.limbs, l_hi.limbs);
    get_half_32<NLIMBS, false>(xs.limbs, xs_lo.limbs);
    multiply_and_add_lsb_neg_modulus_raw(l_hi, neg_mod, xs_lo, r);
    if constexpr (NOF_REDUCTIONS == 2) {
      storage<NLIMBS> r_reduced = {};
      const auto borrow = add_sub_limbs<NLIMBS, true, true>(r, mod2, r_reduced);
      if (!borrow) return r_reduced;
    }
    storage<NLIMBS> r_reduced = {};
    const auto borrow = add_sub_limbs<NLIMBS, true, true>(r, mod1, r_reduced);
    return borrow ? r : r_reduced;
  }

  template <unsigned NLIMBS>
  static constexpr __device__ __forceinline__ bool is_equal(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t limbs_or = x[0] ^ y[0];
    UNROLL
    for (unsigned i = 1; i < NLIMBS; i++)
      limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
  }

  template <unsigned NLIMBS>
  static constexpr __device__ __forceinline__ bool is_zero(const storage<NLIMBS>& xs)
  {
    uint32_t limbs_or = 0;
    for (unsigned i = 0; i < NLIMBS; i++)
      limbs_or |= xs.limbs[i];
    return limbs_or == 0;
  }

  template <unsigned NLIMBS, unsigned BITS>
  static constexpr __device__ __forceinline__ storage<NLIMBS> right_shift(const storage<NLIMBS>& xs)
  {
    if constexpr (BITS == 0)
      return xs;
    else {
      constexpr unsigned BITS32 = BITS % 32;
      constexpr unsigned LIMBS_GAP = BITS / 32;
      storage<NLIMBS> out{};
      if constexpr (LIMBS_GAP < NLIMBS - 1) {
        for (unsigned i = 0; i < NLIMBS - LIMBS_GAP - 1; i++)
          out.limbs[i] = (xs.limbs[i + LIMBS_GAP] >> BITS32) + (xs.limbs[i + LIMBS_GAP + 1] << (32 - BITS32));
      }
      if constexpr (LIMBS_GAP < NLIMBS) out.limbs[NLIMBS - LIMBS_GAP - 1] = (xs.limbs[NLIMBS - 1] >> BITS32);
      return out;
    }
  }

  static constexpr __device__ __forceinline__ void index_err(uint32_t index, uint32_t max_index)
  {
    if (index > max_index) {
      printf("CUDA ERROR: field.h: index out of range: given index - %u > max index - %u", index, max_index);
      assert(false);
    }
  }

  // Assumes the number is even!
  template <unsigned NLIMBS>
  static constexpr __device__ __forceinline__ void div2(const storage<NLIMBS>& xs, storage<NLIMBS>& rs)
  {
    const uint32_t* x = xs.limbs;
    uint32_t* r = rs.limbs;
    if constexpr (NLIMBS > 1) {
      UNROLL
      for (unsigned i = 0; i < NLIMBS - 1; i++) {
        r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
      }
    }
    r[NLIMBS - 1] = x[NLIMBS - 1] >> 1;
  }
} // namespace cuda_math
