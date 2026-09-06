/* Multiboot1 header + 32-bit protected-mode entry.
 * Built with i686-elf-gcc, loaded by QEMU -kernel (no BIOS needed).
 */
    .code32
    .section .multiboot, "a"
    .align 4
    .long 0x1BADB002              /* magic */
    .long 0x00000003              /* flags: align modules + mem info */
    .long -(0x1BADB002 + 0x00000003)

    .section .bss, "aw", @nobits
    .align 16
stack_bottom:
    .skip 16384
stack_top:

    .section .text
    .global _start
    .type _start, @function
_start:
    movl $stack_top, %esp
    pushl %ebx                    /* multiboot info pointer */
    pushl %eax                    /* multiboot magic */
    call kmain
hang:
    hlt
    jmp hang
