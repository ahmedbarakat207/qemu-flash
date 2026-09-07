/*
 * QEMU High-Level Emulation (HLE) Library Shims
 *
 * Direct SysV x86_64 -> Host AAPCS64 / C calling convention thunks for
 * high-performance math (libm), string/memory, and crypto routines.
 *
 * Copyright (c) 2026 QEMU contributors
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hle-thunks.h"

#ifdef TARGET_I386
#include "qemu/osdep.h"
#include "cpu.h"
#else
/* Standalone / Mock definitions for testing and non-target builds */
#ifndef R_EAX
enum {
    R_EAX = 0,
    R_ECX = 1,
    R_EDX = 2,
    R_EBX = 3,
    R_ESP = 4,
    R_EBP = 5,
    R_ESI = 6,
    R_EDI = 7,
    R_R8  = 8,
    R_R9  = 9,
    R_R10 = 10,
    R_R11 = 11,
    R_R12 = 12,
    R_R13 = 13,
    R_R14 = 14,
    R_R15 = 15,
};
#endif

typedef union {
    uint8_t  _b[64];
    uint32_t _l[16];
    uint64_t _q[8];
    float    _s[16];
    double   _d[8];
} MockZMMReg;

typedef struct MockCPUX86State {
    uint64_t regs[16];
    uint64_t eip;
    uint64_t eflags;
    uint8_t  _pad[1024];
    MockZMMReg xmm_regs[32];
} MockCPUX86State;
#endif

/* -------------------------------------------------------------------------
 * Register and Memory Access Helpers
 * ------------------------------------------------------------------------- */

static inline uint64_t get_gpr(void *env, int reg)
{
#ifdef TARGET_I386
    CPUX86State *s = (CPUX86State *)env;
    return s->regs[reg];
#else
    MockCPUX86State *s = (MockCPUX86State *)env;
    return s->regs[reg];
#endif
}

static inline void set_gpr(void *env, int reg, uint64_t val)
{
#ifdef TARGET_I386
    CPUX86State *s = (CPUX86State *)env;
    s->regs[reg] = val;
#else
    MockCPUX86State *s = (MockCPUX86State *)env;
    s->regs[reg] = val;
#endif
}

static inline double get_xmm_double(void *env, int reg)
{
#ifdef TARGET_I386
    CPUX86State *s = (CPUX86State *)env;
    return s->xmm_regs[reg]._d_ZMMReg[0];
#else
    MockCPUX86State *s = (MockCPUX86State *)env;
    return s->xmm_regs[reg]._d[0];
#endif
}

static inline void set_xmm_double(void *env, int reg, double val)
{
#ifdef TARGET_I386
    CPUX86State *s = (CPUX86State *)env;
    s->xmm_regs[reg]._d_ZMMReg[0] = val;
#else
    MockCPUX86State *s = (MockCPUX86State *)env;
    s->xmm_regs[reg]._d[0] = val;
#endif
}

static inline float get_xmm_float(void *env, int reg)
{
#ifdef TARGET_I386
    CPUX86State *s = (CPUX86State *)env;
    return s->xmm_regs[reg]._s_ZMMReg[0];
#else
    MockCPUX86State *s = (MockCPUX86State *)env;
    return s->xmm_regs[reg]._s[0];
#endif
}

static inline void set_xmm_float(void *env, int reg, float val)
{
#ifdef TARGET_I386
    CPUX86State *s = (CPUX86State *)env;
    s->xmm_regs[reg]._s_ZMMReg[0] = val;
#else
    MockCPUX86State *s = (MockCPUX86State *)env;
    s->xmm_regs[reg]._s[0] = val;
#endif
}

static inline void simulate_ret(void *env, void *guest_base)
{
    uint64_t rsp = get_gpr(env, R_ESP);
    uint64_t *host_rsp;
    if (guest_base) {
        host_rsp = (uint64_t *)((uintptr_t)guest_base + (uintptr_t)rsp);
    } else {
        host_rsp = (uint64_t *)(uintptr_t)rsp;
    }
    uint64_t ret_addr = *host_rsp;
    set_gpr(env, R_ESP, rsp + 8);
#ifdef TARGET_I386
    CPUX86State *s = (CPUX86State *)env;
    s->eip = ret_addr;
#else
    MockCPUX86State *s = (MockCPUX86State *)env;
    s->eip = ret_addr;
#endif
}

static inline void *g2h_ptr(uint64_t gaddr, void *guest_base)
{
    if (guest_base) {
        return (void *)((uintptr_t)guest_base + (uintptr_t)gaddr);
    }
    return (void *)(uintptr_t)gaddr;
}

/* -------------------------------------------------------------------------
 * Self-contained SHA-256 & MD5 Implementation for Crypto Thunks
 * ------------------------------------------------------------------------- */

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buffer[64];
} HleSha256Ctx;

#define ROR32(v, n) (((v) >> (n)) | ((v) << (32 - (n))))
#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROR32(x, 2) ^ ROR32(x, 13) ^ ROR32(x, 22))
#define EP1(x) (ROR32(x, 6) ^ ROR32(x, 11) ^ ROR32(x, 25))
#define SIG0(x) (ROR32(x, 7) ^ ROR32(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROR32(x, 17) ^ ROR32(x, 19) ^ ((x) >> 10))

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static void hle_sha256_transform(HleSha256Ctx *ctx, const uint8_t data[64])
{
    uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];
    int i;
    for (i = 0; i < 16; ++i) {
        m[i] = ((uint32_t)data[i * 4] << 24) |
               ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8) |
               ((uint32_t)data[i * 4 + 3]);
    }
    for (; i < 64; ++i) {
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + K256[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void hle_sha256_init_internal(HleSha256Ctx *ctx)
{
    ctx->count = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

static void hle_sha256_update_internal(HleSha256Ctx *ctx, const uint8_t *data, size_t len)
{
    size_t i = 0;
    size_t idx = (size_t)(ctx->count & 63);
    ctx->count += len;
    if (idx > 0) {
        size_t part = 64 - idx;
        if (len < part) {
            memcpy(&ctx->buffer[idx], data, len);
            return;
        }
        memcpy(&ctx->buffer[idx], data, part);
        hle_sha256_transform(ctx, ctx->buffer);
        i = part;
    }
    for (; i + 63 < len; i += 64) {
        hle_sha256_transform(ctx, &data[i]);
    }
    if (i < len) {
        memcpy(ctx->buffer, &data[i], len - i);
    }
}

static void hle_sha256_final_internal(uint8_t hash[32], HleSha256Ctx *ctx)
{
    uint64_t total_bits = ctx->count * 8;
    size_t idx = (size_t)(ctx->count & 63);
    ctx->buffer[idx++] = 0x80;
    if (idx > 56) {
        memset(&ctx->buffer[idx], 0, 64 - idx);
        hle_sha256_transform(ctx, ctx->buffer);
        idx = 0;
    }
    memset(&ctx->buffer[idx], 0, 56 - idx);
    for (int i = 0; i < 8; ++i) {
        ctx->buffer[63 - i] = (uint8_t)(total_bits >> (i * 8));
    }
    hle_sha256_transform(ctx, ctx->buffer);
    for (int i = 0; i < 8; ++i) {
        hash[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
        hash[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        hash[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        hash[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

/* -------------------------------------------------------------------------
 * HLE Thunk Handler Implementations
 * ------------------------------------------------------------------------- */

/* Macro for 1-arg double math thunks */
#define DEF_MATH_1ARG_D(name, fn)                                      \
static bool thunk_##name(void *env, void *gbase)                       \
{                                                                      \
    double x = get_xmm_double(env, 0);                                 \
    set_xmm_double(env, 0, fn(x));                                     \
    simulate_ret(env, gbase);                                          \
    return true;                                                       \
}

/* Macro for 2-arg double math thunks */
#define DEF_MATH_2ARG_D(name, fn)                                      \
static bool thunk_##name(void *env, void *gbase)                       \
{                                                                      \
    double x = get_xmm_double(env, 0);                                 \
    double y = get_xmm_double(env, 1);                                 \
    set_xmm_double(env, 0, fn(x, y));                                  \
    simulate_ret(env, gbase);                                          \
    return true;                                                       \
}

/* Macro for 1-arg float math thunks */
#define DEF_MATH_1ARG_F(name, fn)                                      \
static bool thunk_##name(void *env, void *gbase)                       \
{                                                                      \
    float x = get_xmm_float(env, 0);                                   \
    set_xmm_float(env, 0, fn(x));                                      \
    simulate_ret(env, gbase);                                          \
    return true;                                                       \
}

/* Macro for 2-arg float math thunks */
#define DEF_MATH_2ARG_F(name, fn)                                      \
static bool thunk_##name(void *env, void *gbase)                       \
{                                                                      \
    float x = get_xmm_float(env, 0);                                   \
    float y = get_xmm_float(env, 1);                                   \
    set_xmm_float(env, 0, fn(x, y));                                   \
    simulate_ret(env, gbase);                                          \
    return true;                                                       \
}

DEF_MATH_1ARG_D(sin, sin)
DEF_MATH_1ARG_D(cos, cos)
DEF_MATH_1ARG_D(tan, tan)
DEF_MATH_1ARG_D(asin, asin)
DEF_MATH_1ARG_D(acos, acos)
DEF_MATH_1ARG_D(atan, atan)
DEF_MATH_2ARG_D(atan2, atan2)
DEF_MATH_1ARG_D(exp, exp)
DEF_MATH_1ARG_D(exp2, exp2)
DEF_MATH_1ARG_D(log, log)
DEF_MATH_1ARG_D(log2, log2)
DEF_MATH_1ARG_D(log10, log10)
DEF_MATH_2ARG_D(pow, pow)
DEF_MATH_1ARG_D(sqrt, sqrt)
DEF_MATH_1ARG_D(cbrt, cbrt)
DEF_MATH_2ARG_D(hypot, hypot)
DEF_MATH_1ARG_D(fabs, fabs)
DEF_MATH_1ARG_D(floor, floor)
DEF_MATH_1ARG_D(ceil, ceil)
DEF_MATH_1ARG_D(round, round)
DEF_MATH_1ARG_D(trunc, trunc)
DEF_MATH_2ARG_D(fmod, fmod)

DEF_MATH_1ARG_F(sinf, sinf)
DEF_MATH_1ARG_F(cosf, cosf)
DEF_MATH_1ARG_F(tanf, tanf)
DEF_MATH_1ARG_F(asinf, asinf)
DEF_MATH_1ARG_F(acosf, acosf)
DEF_MATH_1ARG_F(atanf, atanf)
DEF_MATH_2ARG_F(atan2f, atan2f)
DEF_MATH_1ARG_F(expf, expf)
DEF_MATH_1ARG_F(exp2f, exp2f)
DEF_MATH_1ARG_F(logf, logf)
DEF_MATH_1ARG_F(log2f, log2f)
DEF_MATH_1ARG_F(log10f, log10f)
DEF_MATH_2ARG_F(powf, powf)
DEF_MATH_1ARG_F(sqrtf, sqrtf)
DEF_MATH_1ARG_F(cbrtf, cbrtf)
DEF_MATH_2ARG_F(hypotf, hypotf)
DEF_MATH_1ARG_F(fabsf, fabsf)
DEF_MATH_1ARG_F(floorf, floorf)
DEF_MATH_1ARG_F(ceilf, ceilf)
DEF_MATH_1ARG_F(roundf, roundf)
DEF_MATH_1ARG_F(truncf, truncf)
DEF_MATH_2ARG_F(fmodf, fmodf)

/* String & Memory thunks */
static bool thunk_strlen(void *env, void *gbase)
{
    uint64_t s_addr = get_gpr(env, R_EDI);
    const char *s = (const char *)g2h_ptr(s_addr, gbase);
    size_t len = strlen(s);
    set_gpr(env, R_EAX, (uint64_t)len);
    simulate_ret(env, gbase);
    return true;
}

static bool thunk_memcpy(void *env, void *gbase)
{
    uint64_t dst_addr = get_gpr(env, R_EDI);
    uint64_t src_addr = get_gpr(env, R_ESI);
    uint64_t n = get_gpr(env, R_EDX);
    void *dst = g2h_ptr(dst_addr, gbase);
    const void *src = g2h_ptr(src_addr, gbase);
    memcpy(dst, src, (size_t)n);
    set_gpr(env, R_EAX, dst_addr);
    simulate_ret(env, gbase);
    return true;
}

static bool thunk_memset(void *env, void *gbase)
{
    uint64_t dst_addr = get_gpr(env, R_EDI);
    int c = (int)get_gpr(env, R_ESI);
    uint64_t n = get_gpr(env, R_EDX);
    void *dst = g2h_ptr(dst_addr, gbase);
    memset(dst, c, (size_t)n);
    set_gpr(env, R_EAX, dst_addr);
    simulate_ret(env, gbase);
    return true;
}

static bool thunk_memmove(void *env, void *gbase)
{
    uint64_t dst_addr = get_gpr(env, R_EDI);
    uint64_t src_addr = get_gpr(env, R_ESI);
    uint64_t n = get_gpr(env, R_EDX);
    void *dst = g2h_ptr(dst_addr, gbase);
    const void *src = g2h_ptr(src_addr, gbase);
    memmove(dst, src, (size_t)n);
    set_gpr(env, R_EAX, dst_addr);
    simulate_ret(env, gbase);
    return true;
}

static bool thunk_memcmp(void *env, void *gbase)
{
    uint64_t s1_addr = get_gpr(env, R_EDI);
    uint64_t s2_addr = get_gpr(env, R_ESI);
    uint64_t n = get_gpr(env, R_EDX);
    const void *s1 = g2h_ptr(s1_addr, gbase);
    const void *s2 = g2h_ptr(s2_addr, gbase);
    int diff = memcmp(s1, s2, (size_t)n);
    set_gpr(env, R_EAX, (uint64_t)(int64_t)diff);
    simulate_ret(env, gbase);
    return true;
}

/* Crypto thunks */
static bool thunk_sha256_init(void *env, void *gbase)
{
    uint64_t ctx_addr = get_gpr(env, R_EDI);
    HleSha256Ctx *ctx = (HleSha256Ctx *)g2h_ptr(ctx_addr, gbase);
    hle_sha256_init_internal(ctx);
    set_gpr(env, R_EAX, 1); /* OpenSSL success = 1 */
    simulate_ret(env, gbase);
    return true;
}

static bool thunk_sha256_update(void *env, void *gbase)
{
    uint64_t ctx_addr = get_gpr(env, R_EDI);
    uint64_t data_addr = get_gpr(env, R_ESI);
    uint64_t len = get_gpr(env, R_EDX);
    HleSha256Ctx *ctx = (HleSha256Ctx *)g2h_ptr(ctx_addr, gbase);
    const uint8_t *data = (const uint8_t *)g2h_ptr(data_addr, gbase);
    hle_sha256_update_internal(ctx, data, (size_t)len);
    set_gpr(env, R_EAX, 1);
    simulate_ret(env, gbase);
    return true;
}

static bool thunk_sha256_final(void *env, void *gbase)
{
    uint64_t md_addr = get_gpr(env, R_EDI);
    uint64_t ctx_addr = get_gpr(env, R_ESI);
    uint8_t *md = (uint8_t *)g2h_ptr(md_addr, gbase);
    HleSha256Ctx *ctx = (HleSha256Ctx *)g2h_ptr(ctx_addr, gbase);
    hle_sha256_final_internal(md, ctx);
    set_gpr(env, R_EAX, 1);
    simulate_ret(env, gbase);
    return true;
}

/* -------------------------------------------------------------------------
 * Descriptor Registry Table
 * ------------------------------------------------------------------------- */

static HleThunkDesc g_thunk_table[HLE_THUNK_COUNT] = {
    [HLE_THUNK_NONE] = { HLE_THUNK_NONE, NULL, NULL, 0, NULL },

    /* Double math */
    [HLE_THUNK_SIN] = { HLE_THUNK_SIN, "sin", "libm.so.6", 0, thunk_sin },
    [HLE_THUNK_COS] = { HLE_THUNK_COS, "cos", "libm.so.6", 0, thunk_cos },
    [HLE_THUNK_TAN] = { HLE_THUNK_TAN, "tan", "libm.so.6", 0, thunk_tan },
    [HLE_THUNK_ASIN] = { HLE_THUNK_ASIN, "asin", "libm.so.6", 0, thunk_asin },
    [HLE_THUNK_ACOS] = { HLE_THUNK_ACOS, "acos", "libm.so.6", 0, thunk_acos },
    [HLE_THUNK_ATAN] = { HLE_THUNK_ATAN, "atan", "libm.so.6", 0, thunk_atan },
    [HLE_THUNK_ATAN2] = { HLE_THUNK_ATAN2, "atan2", "libm.so.6", 0, thunk_atan2 },
    [HLE_THUNK_EXP] = { HLE_THUNK_EXP, "exp", "libm.so.6", 0, thunk_exp },
    [HLE_THUNK_EXP2] = { HLE_THUNK_EXP2, "exp2", "libm.so.6", 0, thunk_exp2 },
    [HLE_THUNK_LOG] = { HLE_THUNK_LOG, "log", "libm.so.6", 0, thunk_log },
    [HLE_THUNK_LOG2] = { HLE_THUNK_LOG2, "log2", "libm.so.6", 0, thunk_log2 },
    [HLE_THUNK_LOG10] = { HLE_THUNK_LOG10, "log10", "libm.so.6", 0, thunk_log10 },
    [HLE_THUNK_POW] = { HLE_THUNK_POW, "pow", "libm.so.6", 0, thunk_pow },
    [HLE_THUNK_SQRT] = { HLE_THUNK_SQRT, "sqrt", "libm.so.6", 0, thunk_sqrt },
    [HLE_THUNK_CBRT] = { HLE_THUNK_CBRT, "cbrt", "libm.so.6", 0, thunk_cbrt },
    [HLE_THUNK_HYPOT] = { HLE_THUNK_HYPOT, "hypot", "libm.so.6", 0, thunk_hypot },
    [HLE_THUNK_FABS] = { HLE_THUNK_FABS, "fabs", "libm.so.6", 0, thunk_fabs },
    [HLE_THUNK_FLOOR] = { HLE_THUNK_FLOOR, "floor", "libm.so.6", 0, thunk_floor },
    [HLE_THUNK_CEIL] = { HLE_THUNK_CEIL, "ceil", "libm.so.6", 0, thunk_ceil },
    [HLE_THUNK_ROUND] = { HLE_THUNK_ROUND, "round", "libm.so.6", 0, thunk_round },
    [HLE_THUNK_TRUNC] = { HLE_THUNK_TRUNC, "trunc", "libm.so.6", 0, thunk_trunc },
    [HLE_THUNK_FMOD] = { HLE_THUNK_FMOD, "fmod", "libm.so.6", 0, thunk_fmod },

    /* Single math */
    [HLE_THUNK_SINF] = { HLE_THUNK_SINF, "sinf", "libm.so.6", 0, thunk_sinf },
    [HLE_THUNK_COSF] = { HLE_THUNK_COSF, "cosf", "libm.so.6", 0, thunk_cosf },
    [HLE_THUNK_TANF] = { HLE_THUNK_TANF, "tanf", "libm.so.6", 0, thunk_tanf },
    [HLE_THUNK_ASINF] = { HLE_THUNK_ASINF, "asinf", "libm.so.6", 0, thunk_asinf },
    [HLE_THUNK_ACOSF] = { HLE_THUNK_ACOSF, "acosf", "libm.so.6", 0, thunk_acosf },
    [HLE_THUNK_ATANF] = { HLE_THUNK_ATANF, "atanf", "libm.so.6", 0, thunk_atanf },
    [HLE_THUNK_ATAN2F] = { HLE_THUNK_ATAN2F, "atan2f", "libm.so.6", 0, thunk_atan2f },
    [HLE_THUNK_EXPF] = { HLE_THUNK_EXPF, "expf", "libm.so.6", 0, thunk_expf },
    [HLE_THUNK_EXP2F] = { HLE_THUNK_EXP2F, "exp2f", "libm.so.6", 0, thunk_exp2f },
    [HLE_THUNK_LOGF] = { HLE_THUNK_LOGF, "logf", "libm.so.6", 0, thunk_logf },
    [HLE_THUNK_LOG2F] = { HLE_THUNK_LOG2F, "log2f", "libm.so.6", 0, thunk_log2f },
    [HLE_THUNK_LOG10F] = { HLE_THUNK_LOG10F, "log10f", "libm.so.6", 0, thunk_log10f },
    [HLE_THUNK_POWF] = { HLE_THUNK_POWF, "powf", "libm.so.6", 0, thunk_powf },
    [HLE_THUNK_SQRTF] = { HLE_THUNK_SQRTF, "sqrtf", "libm.so.6", 0, thunk_sqrtf },
    [HLE_THUNK_CBRTF] = { HLE_THUNK_CBRTF, "cbrtf", "libm.so.6", 0, thunk_cbrtf },
    [HLE_THUNK_HYPOTF] = { HLE_THUNK_HYPOTF, "hypotf", "libm.so.6", 0, thunk_hypotf },
    [HLE_THUNK_FABSF] = { HLE_THUNK_FABSF, "fabsf", "libm.so.6", 0, thunk_fabsf },
    [HLE_THUNK_FLOORF] = { HLE_THUNK_FLOORF, "floorf", "libm.so.6", 0, thunk_floorf },
    [HLE_THUNK_CEILF] = { HLE_THUNK_CEILF, "ceilf", "libm.so.6", 0, thunk_ceilf },
    [HLE_THUNK_ROUNDF] = { HLE_THUNK_ROUNDF, "roundf", "libm.so.6", 0, thunk_roundf },
    [HLE_THUNK_TRUNCF] = { HLE_THUNK_TRUNCF, "truncf", "libm.so.6", 0, thunk_truncf },
    [HLE_THUNK_FMODF] = { HLE_THUNK_FMODF, "fmodf", "libm.so.6", 0, thunk_fmodf },

    /* Memory / String */
    [HLE_THUNK_MEMCPY] = { HLE_THUNK_MEMCPY, "memcpy", "libc.so.6", 0, thunk_memcpy },
    [HLE_THUNK_MEMSET] = { HLE_THUNK_MEMSET, "memset", "libc.so.6", 0, thunk_memset },
    [HLE_THUNK_MEMMOVE] = { HLE_THUNK_MEMMOVE, "memmove", "libc.so.6", 0, thunk_memmove },
    [HLE_THUNK_MEMCMP] = { HLE_THUNK_MEMCMP, "memcmp", "libc.so.6", 0, thunk_memcmp },
    [HLE_THUNK_STRLEN] = { HLE_THUNK_STRLEN, "strlen", "libc.so.6", 0, thunk_strlen },

    /* Crypto */
    [HLE_THUNK_SHA256_INIT] = { HLE_THUNK_SHA256_INIT, "SHA256_Init", "libcrypto.so", 0, thunk_sha256_init },
    [HLE_THUNK_SHA256_UPDATE] = { HLE_THUNK_SHA256_UPDATE, "SHA256_Update", "libcrypto.so", 0, thunk_sha256_update },
    [HLE_THUNK_SHA256_FINAL] = { HLE_THUNK_SHA256_FINAL, "SHA256_Final", "libcrypto.so", 0, thunk_sha256_final },
};

static bool g_hle_initialized = false;
static size_t g_hle_active_count = 0;

void hle_thunk_init(void)
{
    g_hle_initialized = true;
}

void hle_thunk_reset(void)
{
    for (int i = 0; i < HLE_THUNK_COUNT; ++i) {
        g_thunk_table[i].guest_addr = 0;
    }
    g_hle_active_count = 0;
}

bool hle_thunk_register_address(const char *name, uint64_t guest_addr)
{
    if (!name || guest_addr == 0) {
        return false;
    }
    for (int i = 1; i < HLE_THUNK_COUNT; ++i) {
        if (g_thunk_table[i].name && strcmp(g_thunk_table[i].name, name) == 0) {
            if (g_thunk_table[i].guest_addr == 0) {
                g_hle_active_count++;
            }
            g_thunk_table[i].guest_addr = guest_addr;
            return true;
        }
    }
    return false;
}

const HleThunkDesc *hle_thunk_lookup(uint64_t guest_addr)
{
    if (guest_addr == 0) {
        return NULL;
    }
    for (int i = 1; i < HLE_THUNK_COUNT; ++i) {
        if (g_thunk_table[i].guest_addr == guest_addr) {
            return &g_thunk_table[i];
        }
    }
    return NULL;
}

const HleThunkDesc *hle_thunk_lookup_name(const char *name)
{
    if (!name) {
        return NULL;
    }
    for (int i = 1; i < HLE_THUNK_COUNT; ++i) {
        if (g_thunk_table[i].name && strcmp(g_thunk_table[i].name, name) == 0) {
            return &g_thunk_table[i];
        }
    }
    return NULL;
}

const HleThunkDesc *hle_thunk_get(HleThunkId id)
{
    if (id <= HLE_THUNK_NONE || id >= HLE_THUNK_COUNT) {
        return NULL;
    }
    return &g_thunk_table[id];
}

size_t hle_thunk_registered_count(void)
{
    return g_hle_active_count;
}

bool hle_thunk_dispatch(void *env, uint64_t guest_addr, void *guest_base)
{
    const HleThunkDesc *desc = hle_thunk_lookup(guest_addr);
    if (!desc || !desc->handler) {
        return false;
    }
    return desc->handler(env, guest_base);
}
