/* Branch-heavy freestanding benchmark for TCG direct block chaining.
 *
 * Workload mixes:
 *  - tight loops with backward direct branches (goto_tb candidates),
 *  - direct calls/returns,
 *  - alternating taken/not-taken forward branches (both goto_tb slots),
 *  - indirect calls through a volatile function pointer (goto_ptr path,
 *    never directly chained: baseline cost reference).
 *
 * Prints RDTSC delta + checksum over serial, then either exits QEMU via
 * isa-debug-exit (default) or halts for `info jit` inspection when the
 * multiboot cmdline contains "mode=idle".
 */

#include <stdint.h>
#include <stddef.h>

#define SERIAL_PORT  0x3F8
#define DEBUG_EXIT   0xF4

static inline void outb(uint16_t port, uint8_t v)
{
    __asm__ volatile ("outb %0, %1" : : "a"(v), "Nd"(port));
}

static inline uint8_t inb(uint16_t port)
{
    uint8_t v;
    __asm__ volatile ("inb %1, %0" : "=a"(v) : "Nd"(port));
    return v;
}

static inline uint64_t rdtsc(void)
{
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static void serial_init(void)
{
    outb(SERIAL_PORT + 1, 0x00);
    outb(SERIAL_PORT + 3, 0x80);
    outb(SERIAL_PORT + 0, 0x03);
    outb(SERIAL_PORT + 1, 0x00);
    outb(SERIAL_PORT + 3, 0x03);
    outb(SERIAL_PORT + 2, 0xC7);
    outb(SERIAL_PORT + 4, 0x0B);
}

static void serial_putc(char c)
{
    while ((inb(SERIAL_PORT + 5) & 0x20) == 0) {
    }
    outb(SERIAL_PORT, (uint8_t)c);
}

static void serial_puts(const char *s)
{
    while (*s) {
        if (*s == '\n') {
            serial_putc('\r');
        }
        serial_putc(*s++);
    }
}

static void serial_puthex64(uint64_t v)
{
    static const char dig[] = "0123456789abcdef";
    int i;
    serial_puts("0x");
    for (i = 15; i >= 0; i--) {
        serial_putc(dig[(v >> (i * 4)) & 0xf]);
    }
}

static void serial_putdec(uint32_t v)
{
    char buf[12];
    int i = 0;
    if (v == 0) {
        serial_putc('0');
        return;
    }
    while (v > 0) {
        buf[i++] = (char)('0' + v % 10);
        v /= 10;
    }
    while (i > 0) {
        serial_putc(buf[--i]);
    }
}

/* ---- workload ---- */

#ifndef ITERS
#define ITERS 400000
#endif
#ifndef MEMITERS
#define MEMITERS 24
#endif
#define MEMSZ (1u << 20)

static uint8_t src_buf[MEMSZ] __attribute__((aligned(4096)));
static uint8_t dst_buf[MEMSZ] __attribute__((aligned(4096)));

static uint32_t leaf_add(uint32_t a, uint32_t b) { return a + b; }
static uint32_t leaf_xor(uint32_t a, uint32_t b) { return a ^ b; }
static uint32_t leaf_mul(uint32_t a, uint32_t b) { return a * b + 1; }

typedef uint32_t (*binop_t)(uint32_t, uint32_t);
static binop_t volatile op_table[3] = { leaf_add, leaf_xor, leaf_mul };

/* Memory-bound: streaming copy through guest RAM (softmmu + fences). */
static void __attribute__((noinline)) blkcopy(uint8_t *d, const uint8_t *s,
                                              uint32_t n)
{
    while (n--) {
        *d++ = *s++;
    }
}

static uint32_t __attribute__((noinline)) memwork(void)
{
    uint32_t k, sum = 0;
    for (k = 0; k < MEMITERS; k++) {
        blkcopy(dst_buf, src_buf, MEMSZ);
        sum += dst_buf[k & (MEMSZ - 1)] + dst_buf[(k * 2654435761u) & (MEMSZ - 1)];
    }
    return sum;
}

/* Force the compiler to keep distinct call sites and real branches. */
static uint32_t __attribute__((noinline)) workload(uint32_t seed)
{
    uint32_t acc = seed;
    uint32_t i;

    for (i = 0; i < ITERS; i++) {
        /* direct calls */
        acc += leaf_add(acc, i);
        acc ^= leaf_xor(acc >> 3, i * 2654435761u);

        /* alternating forward branches: exercises both goto_tb slots */
        if (acc & 0x80000000u) {
            acc = acc * 3 + 1;
        } else {
            acc = acc + 0x9e3779b9u;
        }
        if ((i & 1) == 0) {
            acc = (acc << 5) | (acc >> 27);
        } else {
            acc = (acc >> 7) | (acc << 25);
        }

        /* indirect call: goto_ptr path, never directly chained */
        acc += op_table[acc % 3](acc, i);

        /* data-dependent loop trip: short direct-branch chains */
        if ((acc & 7) == 0) {
            acc += leaf_mul(acc, 0x1234567u);
        }
    }
    return acc;
}

struct multiboot_info {
    uint32_t flags;
    uint32_t mem_lower;
    uint32_t mem_upper;
    uint32_t boot_device;
    uint32_t cmdline;
};

static int has_opt(const char *cmd, const char *opt)
{
    const char *p;
    int i;
    if (!cmd) {
        return 0;
    }
    for (p = cmd; *p; p++) {
        for (i = 0; opt[i] && p[i] == opt[i]; i++) {
        }
        if (opt[i] == '\0' && (p[i] == '\0' || p[i] == ' ')) {
            return 1;
        }
    }
    return 0;
}

void kmain(uint32_t magic, struct multiboot_info *mbi)
{
    const char *cmdline = NULL;
    uint64_t t0, t1;
    uint32_t sum;

    (void)magic;
    serial_init();
    if (mbi && (mbi->flags & (1u << 2))) {
        cmdline = (const char *)(uintptr_t)mbi->cmdline;
    }

    serial_puts("DBC-BENCH start iters=");
    serial_putdec(ITERS);
    serial_puts(" memiters=");
    serial_putdec(MEMITERS);
    serial_puts("\n");

    /*
     * Pure-compute benchmark: mask the PIC and clear IF so no hardware
     * timer IRQ ever breaks a TB chain (a usermode-style emulator such
     * as Box64 never sees device interrupts either).  Serial/debug-exit
     * below are polled port I/O and need no IRQs.  Re-enable IF when the
     * workload is done so the idle loop can still halt politely.
     */
    outb(0x21, 0xFF);
    outb(0xA1, 0xFF);
    __asm__ volatile ("cli");

    t0 = rdtsc();
    sum = workload(0x12345u);
    t1 = rdtsc();

    serial_puts("DBC-BENCH done cycles=");
    serial_puthex64(t1 - t0);
    serial_puts(" sum=");
    serial_puthex64(sum);
    serial_puts("\n");

    /* init src pattern (unmeasured), then time the streaming copy */
    for (uint32_t i = 0; i < MEMSZ; i++) {
        src_buf[i] = (uint8_t)(i * 31 + 7);
    }
    __asm__ volatile ("cli");
    t0 = rdtsc();
    sum = memwork();
    t1 = rdtsc();
    __asm__ volatile ("sti");

    serial_puts("DBC-MEM done cycles=");
    serial_puthex64(t1 - t0);
    serial_puts(" sum=");
    serial_puthex64(sum);
    serial_puts("\n");

    if (cmdline && has_opt(cmdline, "mode=idle")) {
        serial_puts("DBC-BENCH idling for info jit\n");
        for (;;) {
            __asm__ volatile ("hlt");
        }
    }

    /* isa-debug-exit: guest status 0x00 -> QEMU exit code 1, check serial. */
    outb(DEBUG_EXIT, 0x00);
    for (;;) {
        __asm__ volatile ("hlt");
    }
}
