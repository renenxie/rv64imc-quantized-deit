// test_softmax.c — Softmax kernel 測試
#include "ops.h"
#include "test_common.h"
#include TEST_DATA

static int8_t out_buf[T_ROWS * T_L];

int main(void) {
    uint64_t i0 = rdinstret();
    qsoftmax_b0(t_in, out_buf, T_ROWS, T_L, t_explut);
    uint64_t i1 = rdinstret();
    return check_and_report("softmax", out_buf, t_golden, T_ROWS * T_L, i1 - i0, 0);
}
