// test_gelu.c — GELU 查表 kernel 測試
#include "ops.h"
#include "test_common.h"
#include TEST_DATA

static int8_t out_buf[T_N];

int main(void) {
    uint64_t i0 = rdinstret();
    qgelu_b0(t_in, out_buf, T_N, t_lut);
    uint64_t i1 = rdinstret();
    return check_and_report("gelu", out_buf, t_golden, T_N, i1 - i0, 0);
}
