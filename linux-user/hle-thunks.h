/*
 * QEMU High-Level Emulation (HLE) Library Shims
 *
 * Direct SysV x86_64 -> Host AAPCS64 / C calling convention thunks for
 * high-performance math (libm), string/memory, and crypto routines.
 *
 * Copyright (c) 2026 QEMU contributors
 */

#ifndef HLE_THUNKS_H
#define HLE_THUNKS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum HleThunkId {
    HLE_THUNK_NONE = 0,

    /* Math (libm) - double */
    HLE_THUNK_SIN,
    HLE_THUNK_COS,
    HLE_THUNK_TAN,
    HLE_THUNK_ASIN,
    HLE_THUNK_ACOS,
    HLE_THUNK_ATAN,
    HLE_THUNK_ATAN2,
    HLE_THUNK_EXP,
    HLE_THUNK_EXP2,
    HLE_THUNK_LOG,
    HLE_THUNK_LOG2,
    HLE_THUNK_LOG10,
    HLE_THUNK_POW,
    HLE_THUNK_SQRT,
    HLE_THUNK_CBRT,
    HLE_THUNK_HYPOT,
    HLE_THUNK_FABS,
    HLE_THUNK_FLOOR,
    HLE_THUNK_CEIL,
    HLE_THUNK_ROUND,
    HLE_THUNK_TRUNC,
    HLE_THUNK_FMOD,

    /* Math (libm) - single (float) */
    HLE_THUNK_SINF,
    HLE_THUNK_COSF,
    HLE_THUNK_TANF,
    HLE_THUNK_ASINF,
    HLE_THUNK_ACOSF,
    HLE_THUNK_ATANF,
    HLE_THUNK_ATAN2F,
    HLE_THUNK_EXPF,
    HLE_THUNK_EXP2F,
    HLE_THUNK_LOGF,
    HLE_THUNK_LOG2F,
    HLE_THUNK_LOG10F,
    HLE_THUNK_POWF,
    HLE_THUNK_SQRTF,
    HLE_THUNK_CBRTF,
    HLE_THUNK_HYPOTF,
    HLE_THUNK_FABSF,
    HLE_THUNK_FLOORF,
    HLE_THUNK_CEILF,
    HLE_THUNK_ROUNDF,
    HLE_THUNK_TRUNCF,
    HLE_THUNK_FMODF,

    /* String / Memory */
    HLE_THUNK_MEMCPY,
    HLE_THUNK_MEMSET,
    HLE_THUNK_MEMMOVE,
    HLE_THUNK_MEMCMP,
    HLE_THUNK_STRLEN,

    /* Cryptography / Hashing */
    HLE_THUNK_SHA256_INIT,
    HLE_THUNK_SHA256_UPDATE,
    HLE_THUNK_SHA256_FINAL,
    HLE_THUNK_MD5_INIT,
    HLE_THUNK_MD5_UPDATE,
    HLE_THUNK_MD5_FINAL,

    HLE_THUNK_COUNT
} HleThunkId;

typedef struct HleThunkDesc {
    HleThunkId id;
    const char *name;
    const char *lib_name;
    uint64_t guest_addr;
    bool (*handler)(void *env, void *guest_base);
} HleThunkDesc;

/* Initialize internal HLE dispatch tables */
void hle_thunk_init(void);

/* Reset address bindings (e.g. between runs) */
void hle_thunk_reset(void);

/* Register a resolved guest function address for a known symbol name */
bool hle_thunk_register_address(const char *name, uint64_t guest_addr);

/* Lookup an intercepted thunk by guest address */
const HleThunkDesc *hle_thunk_lookup(uint64_t guest_addr);

/* Lookup an intercepted thunk by symbol name */
const HleThunkDesc *hle_thunk_lookup_name(const char *name);

/* Lookup by descriptor ID */
const HleThunkDesc *hle_thunk_get(HleThunkId id);

/* Return number of currently registered/active thunk addresses */
size_t hle_thunk_registered_count(void);

/* Dispatch an HLE thunk directly on the x86 guest CPU environment.
 * Automatically marshals arguments from guest regs / XMM, calls the host
 * native implementation, places results in RAX / XMM0, and simulates x86 ret.
 * Returns true if dispatched, false if not found.
 */
bool hle_thunk_dispatch(void *env, uint64_t guest_addr, void *guest_base);

#ifdef __cplusplus
}
#endif

#endif /* HLE_THUNKS_H */
