# Implementation Roadmap: High-Performance Tier-2 LLVM JIT Engine

This document outlines the complete, step-by-step engineering plan to evolve our in-tree QEMU Tier-2 LLVM JIT into a production-grade, near-native execution engine (inspired by architectures like RPCS3, Box64, and modern tracing JITs).

---

## Architecture Overview & Target Milestones

```
+-----------------------------------------------------------------------------------------+
|                                  Tier-2 JIT Architecture                                |
+-----------------------------------------------------------------------------------------+
|  [vCPU Thread]                                                                          |
|    |                                                                                    |
|    +---> Fast Path Dispatch (tb->tier2_code) ----> Native Host Machine Code (ARM64)    |
|    |                                                    ^                               |
|    +---> Translation Capture (tcg_gen_code)             | (Install native pointer)     |
|    |        |                                           |                               |
|    |        v                                           |                               |
|    |    Stable Op Snapshot (Tier2TBRec / Tier2OpRec)    |                               |
|    |        |                                           |                               |
|    |        v                                           |                               |
|  [Background Compiler Thread]                           |                               |
|    |    Multi-TB CFG Fusion & LLVM Lowering             |                               |
|    |        |                                           |                               |
|    |        +---> Memory-Cache Check (CityHash64)       |                               |
|    |        |        |                                  |                               |
|    |        |        +--> Hit: Load Precompiled Object -+                               |
|    |        |        +--> Miss: LLVM -O2 / NEON / Direct RAM Lowering                   |
|    |        v                                                                           |
|    |    On-Disk Persistent Object Cache (~/.cache/qemu/tier2/)                          |
+----+------------------------------------------------------------------------------------+
```

### Performance Targets

| Milestone | Target Workload | Current (Patched) | Target Metric | Core Mechanism |
|---|---|---|---|---|
| **Phase 1** | Arbitrary Multi-TB Loops | 1.1x–1.8x | **3.5x–5.0x** | Extended Basic Block (EBB) / CFG Fusion |
| **Phase 2** | OS Boot & App Startup | 5.1s | **~2.2s–2.8s** | Persistent On-Disk Object Cache (`.obj`) |
| **Phase 3** | Memory-Heavy / Copy Loops | 30.4M cycles | **~10M–12M cycles** | SoftMMU Direct Host RAM Pointer Lowering |
| **Phase 4** | SIMD / Vector Math | 1.15x | **5x–12x** | Native x86 SSE/AVX $\rightarrow$ ARM64 NEON Mapping |
| **Phase 5** | Chained Steady-State Loops | 0.57s | **0.30s–0.40s** | Direct Block Chaining (`goto_tb`) Re-linking |
| **Phase 6** | System-Wide Profiling | Dispatch-sampling | **Zero-blindspot** | Async Signal/Timer-Based Sampler |
| **Phase 7** | Linux User-Mode Library Shims | Emulated Libs | **Near Native (10x+)**| High-Level Emulation (HLE) for C Libs |

---

## Phase 0: Foundations (Completed)

- [x] **Safe W^X APRR Memory Protection** ([tier2.h](file:///Users/ahmed/qemu-dbc-jit/tcg/llvm/tier2.h)): Elimination of Darwin `EXC_BAD_ACCESS` traps; zero W^X switches once a TB is hot.
- [x] **Asynchronous Worker Thread** ([tier2.c](file:///Users/ahmed/qemu-dbc-jit/tcg/llvm/tier2.c)): `tier2-compiler` thread compilation off the vCPU execution path without mutex contention under `thread=single`.
- [x] **Decoupled Stable C-ABI Opcode Records** ([tier2-jit.h](file:///Users/ahmed/qemu-dbc-jit/tcg/llvm/tier2-jit.h)): `Tier2TBRec`, `Tier2TempRec`, `Tier2OpRec` completely isolating the JIT from internal QOM/TCG headers.
- [x] **Live Dynamic Library Bridge** ([tier2-jit.cpp](file:///Users/ahmed/qemu-dbc-jit/tcg/llvm/tier2-jit.cpp), [Makefile](file:///Users/ahmed/qemu-dbc-jit/tcg/llvm/Makefile)): Dynamic loading of `libqemu-tier2.dylib` via LLVM ORC JIT v22 engine.
- [x] **Live Dispatch & Invalidation** ([cpu-exec.c](file:///Users/ahmed/qemu-dbc-jit/accel/tcg/cpu-exec.c), [tb-maint.c](file:///Users/ahmed/qemu-dbc-jit/accel/tcg/tb-maint.c)): Safe direct dispatch via `cpu_tb_exec`, proper `TB_EXIT_IDX0` transitions, and SMC cache invalidation hooks.
- [x] **Verified Benchmark Suite**: `dbc-bench` validated with 100% bit-for-bit checksum match (`0x147ce5ff`), achieving 3.7x–3.9x speedup on unchained execution (0.46s vs 1.80s stock).

---

## Phase 1: Multi-TB Extended Basic Block (EBB) & Loop CFG Fusion

### Problem Statement
Currently, QEMU TCG generates machine code on basic block boundaries. Between blocks, guest registers are committed back to `env->regs[]`, destroying compiler optimization context across jumps. Single-TB trace compilation leaves inter-block optimizations on the table.

### Detailed Implementation Plan

1. **Multi-TB Control Flow Graph Discovery (`tcg/llvm/tier2.c`)**:
   - Enhance `tier2_find_loop_trace()` to build a directed graph of all basic blocks belonging to the inner loop body (`Header -> TB_A -> TB_B -> Header`).
   - Identify:
     - **Entry block**: The loop header.
     - **Internal edges**: Branches between TBs inside the trace.
     - **Side-exit edges**: Branches escaping the trace (early exits, exception checks, interrupts).
   - Package all participating `Tier2TBRec` snapshots into `Tier2TraceDesc.recs[]`.

2. **Control Flow Graph Lowering (`tcg/llvm/tier2-jit.cpp`)**:
   - In `tier2_jit_compile_trace()`:
     - Create an `llvm::BasicBlock` for each unique TB in the trace.
     - Translate internal intra-trace branches (`T2_BR`, `T2_BRCOND`) directly into LLVM conditional branches (`llvm::BranchInst`) targeting the destination block's LLVM `BasicBlock`.
     - Translate side-exits into dedicated deoptimization landing pads that commit guest state and return `(uintptr_t)exit_tb | exit_idx` to `cpu_tb_exec`.

3. **Inter-Block Virtual Register Promotion**:
   - Allocate virtual local storage for guest registers accessed across blocks.
   - Run LLVM's `mem2reg`, `EarlyCSE`, `LICM`, and `GVN` passes across the entire fused CFG.
   - Verify that loads and stores to `env->regs[]` inside the loop are entirely eliminated and converted into native host register allocations (`x19–x28`).

### Verification & Test Criteria
- Verify with `contrib/dbc-bench`: Fused multi-block loop matches checksum `0x147ce5ff` without requiring any hardcoded PC hacks.
- Verify that LLVM IR dump shows an unrolled loop with zero `env` loads inside the loop body.

---

## Phase 2: Persistent On-Disk JIT Cache (The RPCS3 Architecture)

### Problem Statement
QEMU compiles JIT code entirely in volatile memory. Every reboot forces QEMU to re-translate SeaBIOS, kernel initialization, systemd, and application binaries from scratch. Compilation overhead is paid on every execution.

### Detailed Implementation Plan

1. **Cryptographic Trace Keying (`tcg/llvm/tier2-cache.h`, `tier2-cache.c`)**:
   - Create a fast 64-bit/128-bit hash function (XXH3 or CityHash64):
     ```c
     uint64_t tier2_compute_cache_key(const Tier2TBRec *rec);
     ```
   - Key incorporates:
     - Guest binary instructions (exact bytes).
     - Target architecture, CPU feature flags (`cflags & CF_PARALLEL`, etc.).
     - LLVM version and optimization flags.

2. **Persistent Object File Serialization (`tcg/llvm/tier2-jit.cpp`)**:
   - After compiling and optimizing an LLVM `Module`, emit native machine code to an object file on disk:
     ```cpp
     // Path: ~/.cache/qemu/tier2/<hash>.o
     ```
   - Use LLVM's `legacy::PassManager` with `addPassesToEmitFile(CodeGenFileType::ObjectFile)`.

3. **Fast Memory-Mapped Cache Loader**:
   - On compilation request, first check `~/.cache/qemu/tier2/<hash>.o`.
   - If present:
     - Memory-map the object file via LLVM ORC JIT's `ObjectLinkingLayer` (`createObjectLinkingLayer`).
     - Resolve entry symbol without invoking the LLVM optimization pipeline.
     - Lookup time: <100 microseconds (vs ~25–50ms for full LLVM compilation).
   - If missing:
     - Run background LLVM compilation, write `.o` to disk atomically (write to temp file, `rename`), and link.

4. **Cache Invalidation & Eviction**:
   - Implement an LRU eviction policy with a configurable maximum cache size (`QEMU_TIER2_CACHE_MAX_MB`, default: 512 MB).
   - Invalidate on QEMU version bump or host architecture update.

### Verification & Test Criteria
- Run 5 consecutive guest boots of TinyCorePure64.
- Measure boot time on run 1 (cold cache) vs run 2–5 (warm cache).
- Target: Boot time reduction from 5.1s down to <3.0s.

---

## Phase 3: SoftMMU Direct-Map Window & Pointer Lowering

### Problem Statement
In full-system emulation, every memory access passes through QEMU's SoftMMU TLB lookup (`helper_ld/st_mmu`). This adds 5–8 host instructions and pipeline stalls to every 4-byte/8-byte memory operation, consuming 35%–45% of execution time.

### Detailed Implementation Plan

1. **Flat Guest RAM Base Mapping**:
   - Query QEMU's physical memory subsystem (`RAMBlock` list).
   - Provide the background compiler with the contiguous host virtual base address for guest RAM:
     ```c
     void *host_ram_base;
     uint64_t guest_ram_size;
     ```

2. **Direct Memory Lowering in Tier-2 IR (`tcg/llvm/tier2-jit.cpp`)**:
   - In `tier2-jit.cpp`, inspect `T2_QEMU_LD` and `T2_QEMU_ST` operations.
   - For guest virtual addresses that belong to provably direct-mapped identity regions or stack frames (`[rsp + imm]`):
     - Emit a single range check against `guest_ram_size`.
     - Lower directly to native pointer arithmetic:
       ```llvm
       %phys = and i64 %guest_addr, 0x00000000ffffffff
       %host_ptr = getelementptr i8, ptr %ram_base, i64 %phys
       %val = load i32, ptr %host_ptr, align 4
       ```
     - Eliminate function calls to `helper_*_mmu` on the fast path.

3. **Guarded Fallback**:
   - If address crosses a page boundary, is outside RAM bounds, or hits MMIO:
     - Branch to slow-path block that invokes the standard `helper_ld*_mmu` / `helper_st*_mmu` routines.

### Verification & Test Criteria
- Run `memwork()` (streaming memory copy) in `contrib/dbc-bench`.
- Measure guest cycle reduction (current: 30.4M cycles $\rightarrow$ target: <15M cycles).
- Verify bit-for-bit checksum correctness across memory boundaries.

---

## Phase 4: Direct Vector / SIMD Transpilation (x86 SSE/AVX to ARM64 NEON)

### Problem Statement
TCG maps vector instructions into generic helper calls (`tcg-op-gvec.c`), which perform operations via scalar loops or decomposed C functions. Vector performance in QEMU is ~60x slower than native Apple Silicon memory bandwidth.

### Detailed Implementation Plan

1. **Capture Vector Opcodes in `Tier2OpRec` (`tcg/llvm/tier2-jit.h`)**:
   - Extend `Tier2Op` enum with SIMD operations:
     - `T2_VEC_ADD`, `T2_VEC_SUB`, `T2_VEC_MUL`
     - `T2_VEC_AND`, `T2_VEC_OR`, `T2_VEC_XOR`
     - `T2_VEC_SHL`, `T2_VEC_SHR`
     - `T2_VEC_LOAD`, `T2_VEC_STORE`
   - Capture vector element size (8-bit, 16-bit, 32-bit, 64-bit) and total register width (128-bit XMM, 256-bit YMM).

2. **Lowering to Native LLVM Vector Types (`tcg/llvm/tier2-jit.cpp`)**:
   - Map 128-bit XMM operations directly to LLVM vector types:
     - `<4 x i32>`, `<2 x i64>`, `<4 x float>`, `<2 x double>`
   - Map 256-bit YMM operations to `<8 x i32>`, `<4 x i64>`, etc.
   - Use LLVM IR vector instructions:
     ```llvm
     %v0 = load <4 x i32>, ptr %src1_ptr
     %v1 = load <4 x i32>, ptr %src2_ptr
     %v2 = add <4 x i32> %v0, %v1
     store <4 x i32> %v2, ptr %dst_ptr
     ```
   - Clang's AArch64 backend compiles this directly into 128-bit ARM64 NEON instructions:
     ```assembly
     ldr q0, [x1]
     ldr q1, [x2]
     add.4s v0, v0, v1
     str q0, [x0]
     ```

3. **Register Storage in `CPUArchState`**:
   - Map XMM registers directly to `env->xmm_regs[i]` byte offsets.
   - Promote vector temps into native ARM64 vector registers (`v0–v31`) across the loop body.

### Verification & Test Criteria
- Compile and run microbenchmarks with SSE2/AVX vector loops (e.g. 128-bit vector dot-product, matrix math).
- Validate 5x–12x speedup over stock QEMU vector helper execution.

---

## Phase 5: Chain-Graph Integration for Chained Mode (`goto_tb` Re-Linking)

### Problem Statement
In normal execution, QEMU uses direct block chaining (`goto_tb`): once a block finishes, it jumps directly to the next block in machine code without returning to `cpu_tb_exec()`. As a result, Tier-2 compiled code in `tb->tier2_code` is currently only entered when an unchained exit returns to the dispatcher.

### Detailed Implementation Plan

1. **JIT Jump Slot Patching (`accel/tcg/cpu-exec.c`, `tcg/llvm/tier2.c`)**:
   - When a trace header TB is successfully compiled by Tier-2:
     - Inspect all predecessor TBs that have chained jumps to this header (`tb->jmp_list_head`).
     - Patch the predecessor jump slots to point directly to the Tier-2 native entry point rather than the old TCG RX code pointer.
   - On Tier-2 invalidation (`tier2_invalidate`):
     - Unpatch the predecessor jump slots, restoring the fallback TCG pointer.

2. **Execution State Transition**:
   - Ensure the Tier-2 entry stub correctly conforms to host register conventions when entered directly from a preceding TCG block's jump slot.

### Verification & Test Criteria
- Run `dbc-bench` under standard chained mode (`patched-chained`).
- Verify that the workload loop uses Tier-2 native code even when block chaining is active.
- Target: Chained benchmark time reduced from 0.57s down to ~0.35s.

---

## Phase 6: Asynchronous Statistical Profiler (Zero-Blindspot Sampling)

### Problem Statement
Counting dispatches in software cannot detect tight loops that stay entirely in chained code or jump-cache loops. A true sampling profiler must sample asynchronously.

### Detailed Implementation Plan

1. **Host Timer / Signal-Based Sampler (`tcg/llvm/tier2-prof.c`)**:
   - Configure a dedicated high-resolution POSIX timer (`timer_create` / `setitimer` with `SIGPROF` or `pthread_kill` sampling) on the host.
   - Frequency: ~250 Hz – 1000 Hz.
   - Signal handler inspects the vCPU thread's current guest PC (`cpu->neg.icount_decr` or `env->eip`).

2. **Lock-Free Hotspot Table**:
   - Record sampled PCs in a compact thread-local array or lock-free hash ring.
   - When a guest PC accounts for >5% of samples over a 50ms window:
     - Flag the corresponding TB as hot (`tb->exec_count = TIER2_HOT_THRESHOLD`).
     - Trigger immediate trace collection and compilation.

3. **Zero Dispatch Overhead**:
   - Completely removes inline counter increments and W^X checks from `cpu_tb_exec()` and `tb_lookup()`.
   - Leaves TCG execution at 100% stock dispatch speed.

### Verification & Test Criteria
- Test with pathological loop patterns (tight backward branches, polymorphic calls).
- Verify that hot loops are detected and compiled within 100ms of entering the loop.

---

## Phase 7: High-Level Emulation (HLE) Library Shims for `linux-user`

### Problem Statement
For users running x86_64 binaries under Linux user-mode (`qemu-x86_64` on Android/proot), emulating common standard C libraries (`libm.so`, `libcrypto.so`, `libz.so`, `libvulkan.so`) through x86 JIT is wasteful when the host ARM64 system already has optimized native implementations.

### Detailed Implementation Plan

1. **Symbol Interception Layer (`linux-user/`)**:
   - Identify dynamic library calls during ELF loading (`ld.so`).
   - Match exported symbols for heavy libraries (`sin`, `cos`, `pow`, `AES_encrypt`, `SHA256_Update`, `deflate`, `vkQueueSubmit`).

2. **Thunk Dispatcher**:
   - When the guest calls an intercepted function:
     - Marshal arguments from x86_64 calling convention (RDI, RSI, RDX, RCX, R8, R9, XMM0–7) to ARM64 calling convention (X0–X7, V0–V7).
     - Call host native library (`/usr/lib/aarch64-linux-gnu/...`).
     - Marshal return value back to guest RAX/XMM0.
     - Skip emulating millions of x86 library instructions entirely.

### Verification & Test Criteria
- Test with crypto benchmarks (`openssl speed sha256`) and math benchmarks under `qemu-x86_64`.
- Target: 10x–50x speedup on library-heavy tasks.

---

## Execution Matrix & Tracking

| Phase | Description | Target Files | Difficulty | Status |
|---|---|---|---|---|
| **Phase 0** | Sampling profiler, background worker, ORC JIT, prologue bridge | `tcg/llvm/*`, `cpu-exec.c` | Medium | **DONE** |
| **Phase 1** | Multi-TB Extended Basic Block (EBB) & Loop CFG Fusion | `tier2.c`, `tier2-jit.cpp` | Hard | **DONE (mechanism; fused≈TCG on dbc, 3.5x–5x target open)** |
| **Phase 2** | Persistent On-Disk JIT Cache (`~/.cache/qemu/tier2/*.o`) | `tier2-cache.c`, `tier2-jit.cpp` | Medium | **DONE (mechanism; epoch=2; boot-time target unmeasured)** |
| **Phase 3** | SoftMMU Direct Host RAM Pointer Lowering | `tier2-jit.cpp`, `cputlb.c` | Hard | **DONE (mechanism; cycle target unmeasured)** |
| **Phase 4** | Native x86 SSE/AVX to ARM64 NEON Transpilation | `tier2-jit.cpp`, `tcg-op-vec.c` | Hard | **TODO** |
| **Phase 5** | Chain-Graph Re-linking (`goto_tb` Patching) | `cpu-exec.c`, `tier2.c` | Hard | **TODO** |
| **Phase 6** | Async Signal-Based Statistical Profiler | `tier2-prof.c`, `tier2.h` | Medium | **DONE (mechanism; inline counters retained alongside)** |
| **Phase 7** | High-Level Emulation (HLE) Library Shims for `linux-user` | `linux-user/*` | Very Hard | **TODO** |

### Session Notes (2026-09-07)
- P6 proven: SIGPROF sampler (500 Hz default, `QEMU_TIER2_PROF_HZ`) detects chained-blind hot loops (10 requests → 9 installs, checksum `0x147ce5ff`); sampler-side consume via exact-state `tb_htable_lookup` + backward scan; DFS loop finder; single-TB fallback (call-heavy loops have no `jmp_dest` cycles by construction).
- Cache lesson: `T2_CALL` used to bake helper addresses (`blr x8`); stale ASLR targets crashed warm runs flakily. Fixed by routing calls through recorded names (link-time externals); epoch bumped to 2. Rule: never bake host addresses into cached objects.
- Build note: the JIT engine ships as `build/libqemu-tier2.dylib` (`make -C tcg/llvm all`); `ninja` alone does NOT rebuild it.

### Design Notes (P6 profiler + cache hardening)
- **Signal unblock (required):** `qemu_thread_create` starts every thread (incl. vCPUs) with ~all signals blocked. The sampler's SIGPROF pends forever unless unblocked, so `tier2_unblock_profiler_signal()` runs once per TCG vCPU thread at startup (rr + mttcg thread fns). Zero per-dispatch cost. SIGPROF itself is never in QEMU's blocked set (`util/main-loop.c`), verified before relying on it.
- **Exact-state lookup key:** `tb_htable_lookup` demands exact `(pc, cs_base, flags, cflags)`. Guessing `cflags=0` misses everything (real TBs carry e.g. `CF_PCREL` on this host, `cflags=0xff020000`). The handler therefore records `curr_cflags(cpu)` per sample (pure reads, handler-safe). Sample-time key == dispatch-time key by construction; any mid-TB mode change degrades to a lookup miss (request table covers it via dispatch).
- **Mid-TB PCs:** samples land mid-TB but the htable is keyed by TB start, so consume scans backward up to 1 KB for the containing TB (exact key + range check + `!CF_INVALID`). Bounded, rare-path-only (hot threshold crossings).
- **DFS loop finder (replaces greedy strategy-1):** greedy slot-0-first died whenever slot 0 pointed at a loop exit. DFS backtracks over both slots (path ≤16, dead-set, each node expanded once), prefers header-closing cycles, still accepts entry+cycle closes. Strict superset of greedy behavior; nochain path unchanged otherwise (strategy-2 intact).
- **Install edge re-validation:** provenance (`next[]`) is observed at discovery, up to a full compile (~100 ms) before install — wider now that the sampler also discovers. At install, every proven edge is re-checked against current `jmp_dest` (+ `!CF_INVALID` both ends); any change discards the compile (safe direction; re-enqueue on next hot crossing).
- **Request table contract:** `tier2_prof_requests[8]` + `tier2_has_prof_requests` flag (one predictable branch per dispatch when empty). All cross-thread state is plain data with benign races: torn reads can only cause a spurious compile (valid trace, install guard dedupes) or a dropped sample. `pc==0` is the empty sentinel. Stale slots age out after 5 s.
- **Single-TB fallback + caveat:** call/ret/indirect-heavy loops (like dbc-bench: direct calls, `op_table[acc%3]` indirect) transfer through `jmp_cache`, leaving no `jmp_dest` links — strategy-1 can never fire there, and the sampler thread owns no history for strategy-2. Fallback compiles the hot TB alone so detection still yields installable code. Caveat: the worker install guard lets an installed single keep a future fused trace out (same header) — P5 needs a versioned install guard to supersede singles.
- **Sampler threading:** sampler thread is `rcu_register_thread()`ed; drain+consume run under one `rcu_read_lock` (htable + TB structs are RCU-stable); W^X toggles only on actual consume matches, never per sample. `tier2_shutdown()` is never called by QEMU (pre-existing), so threads die with the process; diagnostics are drained-count-based, not shutdown-based.
- **Cache soundness rules (from the warm-run SIGSEGV hunt):** content key covers inputs (pc/size/ops/names/flags/tlb/gm/epoch/versions) but cannot see the emitter — any emission change must bump `TIER2_CACHE_EPOCH`. No host address may be baked into cached code (all calls via named externals resolved per-process in `defineRuntimeSymbols`); the `UNNAMED call target` verbose warning is the tripwire. Debug method that worked: byte-compare same-key objects across boots + disassemble the differing pair.
