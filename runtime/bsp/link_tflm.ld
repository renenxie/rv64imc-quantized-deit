/* link_tflm.ld — 給 LiteRT Micro（C++）用的 Spike bare-metal 記憶體配置
 * 與 src/common/link.ld 相同的起點（0x80000000），另外加上 C++ 需要的
 * .init_array（靜態建構子）與 newlib 用的 heap。 */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY {
  DRAM (rwx) : ORIGIN = 0x80000000, LENGTH = 512M
}

SECTIONS {
  .text : {
    *(.text.init)
    *(.text .text.*)
  } > DRAM

  .rodata : ALIGN(16) { *(.rodata .rodata.* .srodata .srodata.*) } > DRAM

  .init_array : ALIGN(8) {
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN(__preinit_array_end = .);
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array .ctors))
    PROVIDE_HIDDEN(__init_array_end = .);
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array .dtors))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } > DRAM

  .eh_frame : { KEEP(*(.eh_frame)) } > DRAM

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

  /* 1 MiB stack */
  .stack (NOLOAD) : ALIGN(16) {
    . += 1M;
    __stack_top = .;
  } > DRAM

  /* newlib 的 heap（printf 等會用到少量）；LiteRT Micro 本身不用 malloc */
  .heap (NOLOAD) : ALIGN(16) {
    __heap_start = .;
    . += 16M;
    __heap_end = .;
  } > DRAM

  _end = .;
  end = .;
}
