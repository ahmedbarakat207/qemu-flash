/*
 *  x86 memory access helpers
 *
 *  Copyright (c) 2003 Fabrice Bellard
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#include "qemu/osdep.h"
#include "cpu.h"
#include "exec/helper-proto.h"
#include "accel/tcg/cpu-ldst.h"
#include "qemu/int128.h"
#include "qemu/atomic128.h"
#include "tcg/tcg.h"
#include "helper-tcg.h"
#include "accel/tcg/probe.h"
#include "exec/target_page.h"

void helper_boundw(CPUX86State *env, target_ulong a0, int v)
{
    int low, high;

    low = cpu_ldsw_le_data_ra(env, a0, GETPC());
    high = cpu_ldsw_le_data_ra(env, a0 + 2, GETPC());
    v = (int16_t)v;
    if (v < low || v > high) {
        if (env->hflags & HF_MPX_EN_MASK) {
            env->bndcs_regs.sts = 0;
        }
        raise_exception_ra(env, EXCP05_BOUND, GETPC());
    }
}

void helper_boundl(CPUX86State *env, target_ulong a0, int v)
{
    int low, high;

    low = cpu_ldl_le_data_ra(env, a0, GETPC());
    high = cpu_ldl_le_data_ra(env, a0 + 4, GETPC());
    if (v < low || v > high) {
        if (env->hflags & HF_MPX_EN_MASK) {
            env->bndcs_regs.sts = 0;
        }
        raise_exception_ra(env, EXCP05_BOUND, GETPC());
    }
}

/*
 * Hardware-accelerated memory zeroing / filling (rep stos).
 * Fast-paths contiguous guest RAM ranges via host native memset/SIMD.
 */
void helper_fast_rep_stos(CPUX86State *env, int ot, int aflag, target_ulong val)
{
    if (env->df < 0) {
        return;
    }
    if (!(env->cr[0] & CR0_PE_MASK)) {
        return;
    }
    if (!(env->hflags & HF_CS64_MASK)) {
        if (env->segs[R_ES].base != 0) {
            return;
        }
    }

    target_ulong cx_mask = MAKE_64BIT_MASK(0, 8 << aflag);
    target_ulong count = env->regs[R_ECX] & cx_mask;
    if (count < 8) {
        return;
    }

    int mmu_idx = cpu_mmu_index(env_cpu(env), false);
    if (ot == MO_8) {
        val = (uint8_t)val;
    } else if (ot == MO_16) {
        val = (uint16_t)val;
    } else if (ot == MO_32) {
        val = (uint32_t)val;
    }

    CPUState *cs = env_cpu(env);

    while (count > 0) {
        if (unlikely(qatomic_read(&cs->interrupt_request) ||
                     qatomic_read(&cs->exit_request))) {
            break;
        }

        target_ulong cur_edi = env->regs[R_EDI] & cx_mask;
        size_t bytes_to_page = TARGET_PAGE_SIZE - (cur_edi & (TARGET_PAGE_SIZE - 1));
        size_t max_bytes = (size_t)count << ot;
        size_t chunk_bytes = bytes_to_page < max_bytes ? bytes_to_page : max_bytes;

        chunk_bytes &= ~((1 << ot) - 1);
        if (chunk_bytes == 0) {
            break;
        }

        void *hptr = probe_write(env, cur_edi, chunk_bytes, mmu_idx, GETPC());
        if (!hptr) {
            break;
        }

        if (ot == MO_8) {
            memset(hptr, (uint8_t)val, chunk_bytes);
        } else if (val == 0) {
            memset(hptr, 0, chunk_bytes);
        } else if (ot == MO_16) {
            uint16_t *p = (uint16_t *)hptr;
            size_t n = chunk_bytes >> 1;
            uint16_t v16 = (uint16_t)val;
            for (size_t i = 0; i < n; i++) {
                p[i] = v16;
            }
        } else if (ot == MO_32) {
            uint32_t *p = (uint32_t *)hptr;
            size_t n = chunk_bytes >> 2;
            uint32_t v32 = (uint32_t)val;
            for (size_t i = 0; i < n; i++) {
                p[i] = v32;
            }
        } else if (ot == MO_64) {
            uint64_t *p = (uint64_t *)hptr;
            size_t n = chunk_bytes >> 3;
            for (size_t i = 0; i < n; i++) {
                p[i] = val;
            }
        }

        size_t chunk_elems = chunk_bytes >> ot;
        count -= chunk_elems;
        if (aflag == MO_16) {
            env->regs[R_EDI] = (env->regs[R_EDI] & ~0xffffull) |
                               ((cur_edi + chunk_bytes) & 0xffffull);
            env->regs[R_ECX] = (env->regs[R_ECX] & ~0xffffull) |
                               (count & 0xffffull);
        } else if (aflag == MO_32) {
            env->regs[R_EDI] = (uint32_t)(cur_edi + chunk_bytes);
            env->regs[R_ECX] = (uint32_t)count;
        } else {
            env->regs[R_EDI] = cur_edi + chunk_bytes;
            env->regs[R_ECX] = count;
        }
    }
}

/*
 * Hardware-accelerated memory copying (rep movs).
 * Fast-paths contiguous guest RAM blocks using host native memmove.
 */
void helper_fast_rep_movs(CPUX86State *env, int ot, int aflag, int ovr_seg)
{
    if (env->df < 0) {
        return;
    }
    if (!(env->cr[0] & CR0_PE_MASK)) {
        return;
    }
    int seg = ovr_seg < 0 ? R_DS : ovr_seg;
    if (env->hflags & HF_CS64_MASK) {
        if (seg == R_FS || seg == R_GS) {
            return;
        }
    } else {
        if (env->segs[R_ES].base != 0 || env->segs[seg].base != 0) {
            return;
        }
    }

    target_ulong cx_mask = MAKE_64BIT_MASK(0, 8 << aflag);
    target_ulong count = env->regs[R_ECX] & cx_mask;
    if (count < 8) {
        return;
    }

    int mmu_idx = cpu_mmu_index(env_cpu(env), false);
    CPUState *cs = env_cpu(env);

    while (count > 0) {
        if (unlikely(qatomic_read(&cs->interrupt_request) ||
                     qatomic_read(&cs->exit_request))) {
            break;
        }

        target_ulong cur_edi = env->regs[R_EDI] & cx_mask;
        target_ulong cur_esi = env->regs[R_ESI] & cx_mask;

        size_t bytes_to_page_dst = TARGET_PAGE_SIZE - (cur_edi & (TARGET_PAGE_SIZE - 1));
        size_t bytes_to_page_src = TARGET_PAGE_SIZE - (cur_esi & (TARGET_PAGE_SIZE - 1));
        size_t page_limit = bytes_to_page_dst < bytes_to_page_src ? bytes_to_page_dst : bytes_to_page_src;
        size_t max_bytes = (size_t)count << ot;
        size_t chunk_bytes = page_limit < max_bytes ? page_limit : max_bytes;

        chunk_bytes &= ~((1 << ot) - 1);
        if (chunk_bytes == 0) {
            break;
        }

        void *hptr_src = probe_read(env, cur_esi, chunk_bytes, mmu_idx, GETPC());
        if (!hptr_src) {
            break;
        }
        void *hptr_dst = probe_write(env, cur_edi, chunk_bytes, mmu_idx, GETPC());
        if (!hptr_dst) {
            break;
        }

        memmove(hptr_dst, hptr_src, chunk_bytes);

        size_t chunk_elems = chunk_bytes >> ot;
        count -= chunk_elems;
        if (aflag == MO_16) {
            env->regs[R_EDI] = (env->regs[R_EDI] & ~0xffffull) |
                               ((cur_edi + chunk_bytes) & 0xffffull);
            env->regs[R_ESI] = (env->regs[R_ESI] & ~0xffffull) |
                               ((cur_esi + chunk_bytes) & 0xffffull);
            env->regs[R_ECX] = (env->regs[R_ECX] & ~0xffffull) |
                               (count & 0xffffull);
        } else if (aflag == MO_32) {
            env->regs[R_EDI] = (uint32_t)(cur_edi + chunk_bytes);
            env->regs[R_ESI] = (uint32_t)(cur_esi + chunk_bytes);
            env->regs[R_ECX] = (uint32_t)count;
        } else {
            env->regs[R_EDI] = cur_edi + chunk_bytes;
            env->regs[R_ESI] = cur_esi + chunk_bytes;
            env->regs[R_ECX] = count;
        }
    }
}
