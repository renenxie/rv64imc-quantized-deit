/* link.ld — Spike bare-metal 記憶體配置
 * Spike 預設 DRAM 從 0x80000000 開始（預設 2 GiB）。 */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY {
  DRAM (rwx) : ORIGIN = 0x80000000, LENGTH = 256M
}

SECTIONS {
  .text : {
    *(.text.init)
    *(.text .text.*)
  } > DRAM

  .rodata : ALIGN(16) { *(.rodata .rodata.* .srodata .srodata.*) } > DRAM

  /* Spike 以符號名稱 tohost / fromhost 找到 HTIF 通訊位址 */
  .tohost : ALIGN(64) { *(.tohost) } > DRAM

  .data : ALIGN(16) {
    *(.data .data.*)
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > DRAM

  .bss (NOLOAD) : ALIGN(16) {
    __bss_start = .;
    *(.sbss .sbss.* .bss .bss.* COMMON)
    . = ALIGN(8);
    __bss_end = .;
  } > DRAM

  /* 256 KiB stack */
  .stack (NOLOAD) : ALIGN(16) {
    . += 256K;
    __stack_top = .;
  } > DRAM

  _end = .;
}
