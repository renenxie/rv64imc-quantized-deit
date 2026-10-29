// perf.h — 讀取 RISC-V 效能計數器
// 在 Spike 上 cycle 與 instret 相同（Spike 不模擬時序），
// 所以期中以 instret（已退役指令數）為主要指標。
#ifndef PERF_H
#define PERF_H

#include <stdint.h>

static inline uint64_t rdinstret(void) {
    uint64_t x;
    __asm__ volatile("rdinstret %0" : "=r"(x));
    return x;
}

static inline uint64_t rdcycle(void) {
    uint64_t x;
    __asm__ volatile("rdcycle %0" : "=r"(x));
    return x;
}

#endif
