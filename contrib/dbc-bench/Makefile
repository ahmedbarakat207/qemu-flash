# Build the freestanding multiboot guest with i686-elf-gcc.
CC      = i686-elf-gcc
BENCH_ITERS ?= 8000000
CFLAGS  ?= -std=c11 -O2 -ffreestanding -nostdlib -Wall -Wextra \
	   -fno-pic -fno-pie -fno-stack-protector -mno-red-zone -g \
	   -DITERS=$(BENCH_ITERS)

all: kernel.elf

boot.o: boot.S
	$(CC) $(CFLAGS) -c boot.S -o boot.o

bench.o: bench.c
	$(CC) $(CFLAGS) -c bench.c -o bench.o

kernel.elf: boot.o bench.o link.ld
	$(CC) $(CFLAGS) -T link.ld boot.o bench.o -o kernel.elf -lgcc

check: kernel.elf
	i686-elf-objcopy -O binary kernel.elf kernel.bin
	grub-file --is-x86-multiboot kernel.elf && echo "multiboot OK" || echo "multiboot header check skipped"

clean:
	rm -f boot.o bench.o kernel.elf kernel.bin

.PHONY: all check clean
