/* LLVM tier-2 runtime interface (sketch, header-only, unused).
 *
 * Long-term integration point for an LLVM tier-2 translator:
 * hot traces detected at run time are re-emitted as LLVM IR, optimized,
 * JIT-compiled with ORC, and executed in place of TCG code until
 * invalidated.  Nothing includes this yet; it documents the intended
 * boundary so the off-line prototype (contrib/llvm-tier2) grows toward
 * a fixed target.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef TCG_LLVM_TIER2_H
#define TCG_LLVM_TIER2_H

#include <stdbool.h>
#include <stdint.h>

typedef struct TranslationBlock TranslationBlock;

/* Opaque per-region handle owned by the tier-2 compiler thread. */
typedef struct Tier2Region Tier2Region;

/* Initialize/shutdown the ORC JIT context.  No-op until wired. */
void tier2_init(void);
void tier2_shutdown(void);

/*
 * Hotness accounting, called with a TB that just completed execution.
 * Returns true if the TB's region crossed the compile threshold and a
 * tier-2 compile was queued.  Must be cheap: a single counter bump.
 */
bool tier2_notify_exec(TranslationBlock *tb);

/*
 * Look up tier-2 code for @tb.  Returns the executable entry point, or
 * NULL to keep executing TCG code.  The returned code observes the same
 * CPU-state contract as the TB it replaces.
 */
void *tier2_lookup(const TranslationBlock *tb);

/*
 * Drop all tier-2 code derived from @tb.  Called from the existing TB
 * invalidation paths (it must dominate: no tier-2 code may outlive the
 * TB it was compiled from).
 */
void tier2_invalidate(TranslationBlock *tb);

#endif /* TCG_LLVM_TIER2_H */
