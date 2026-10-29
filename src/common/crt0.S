# crt0.S — 最小 bare-metal 啟動程式（在 Spike 的 M-mode 執行，不需要 pk）
# Spike 開機 ROM 會跳到 0x80000000，也就是 _start。

    .section .text.init, "ax"
    .globl _start
_start:
    # 設定 global pointer（relaxation 需要）
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    # 設定 stack
    la      sp, __stack_top

    # 清空 .bss
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:
    # 設定 trap handler：發生例外（例如非法指令）時印出原因並結束
    la      t0, trap_entry
    csrw    mtvec, t0

    call    main
    call    exit          # 以 main 的回傳值結束
3:  j       3b

    .align 2
trap_entry:
    csrr    a0, mcause
    csrr    a1, mepc
    csrr    a2, mtval
    call    handle_trap
4:  j       4b
