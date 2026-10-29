// test_common.h — 各測試共用：逐位元比對 + 印出 RESULT 行
#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdint.h>
#include "perf.h"

int printf(const char *fmt, ...);  // 由 src/common/syscalls.c 提供（bare-metal）

static int check_and_report(const char *op, const int8_t *got, const int8_t *expect,
                            int n, uint64_t insts, uint64_t macs) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (got[i] != expect[i]) {
            if (errors < 5)
                printf("  mismatch @%d: got %d, expect %d\n", i, got[i], expect[i]);
            errors++;
        }
    }
    // 不用浮點：比值以 x1000 定點表示
    printf("RESULT op=%s outputs=%d insts=%lu inst_per_out_x1000=%lu",
           op, n, (unsigned long)insts, (unsigned long)(insts * 1000 / n));
    if (macs)
        printf(" macs=%lu mac_per_inst_x1000=%lu",
               (unsigned long)macs, (unsigned long)(macs * 1000 / insts));
    printf(" status=%s errors=%d\n", errors ? "FAIL" : "PASS", errors);
    return errors ? 1 : 0;
}

#endif
