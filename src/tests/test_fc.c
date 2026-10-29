// test_fc.c — 驗證 FC kernel 並量測動態指令數
//
// 編譯時用 -DTEST_DATA='"fc_small.h"' 指定測試資料，-DKERNEL=qfc_b0 指定要測的版本。
// 輸出一行以 "RESULT" 開頭、方便腳本解析的結果。
int printf(const char *fmt, ...);  // 由 src/common/syscalls.c 提供（bare-metal）
#include <stdint.h>
#include "perf.h"
#include "qfc.h"
#include TEST_DATA

#ifndef KERNEL
#define KERNEL qfc_b0
#endif
#define STR_(x) #x
#define STR(x) STR_(x)

static int8_t out_buf[T_M * T_N];

int main(void) {
    qfc_params_t p = {
        .M = T_M, .K = T_K, .N = T_N,
        .in_zp = T_IN_ZP, .out_zp = T_OUT_ZP,
        .act_min = T_ACT_MIN, .act_max = T_ACT_MAX,
        .quant_mult = t_qm, .shift = t_shift,
    };

    uint64_t i0 = rdinstret();
    KERNEL(&p, t_in, t_w, t_bias, out_buf);
    uint64_t i1 = rdinstret();

    int errors = 0;
    for (int i = 0; i < T_M * T_N; i++) {
        if (out_buf[i] != t_golden[i]) {
            if (errors < 5)
                printf("  mismatch @%d (m=%d,n=%d): got %d, expect %d\n",
                       i, i / T_N, i % T_N, out_buf[i], t_golden[i]);
            errors++;
        }
    }

    uint64_t insts = i1 - i0;
    uint64_t macs = (uint64_t)T_M * T_K * T_N;
    uint64_t outs = (uint64_t)T_M * T_N;
    // 不用浮點：以 x1000 定點印出比值
    printf("RESULT kernel=%s M=%d K=%d N=%d insts=%lu macs=%lu "
           "mac_per_inst_x1000=%lu inst_per_out=%lu status=%s errors=%d\n",
           STR(KERNEL), T_M, T_K, T_N, (unsigned long)insts, (unsigned long)macs,
           (unsigned long)(macs * 1000 / insts), (unsigned long)(insts / outs),
           errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}
