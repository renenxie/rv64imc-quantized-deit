// test_add.c — 殘差相加 kernel 測試
#include "ops.h"
#include "test_common.h"
#include TEST_DATA

static int8_t out_buf[T_N];

int main(void) {
    qadd_params_t p = {
        .zp1 = T_ZP1, .zp2 = T_ZP2, .out_zp = T_OZP, .left_shift = T_LSHIFT,
        .m1 = T_M1, .m2 = T_M2, .m_out = T_MO,
        .s1 = T_S1, .s2 = T_S2, .s_out = T_SO,
        .act_min = -128, .act_max = 127,
    };
    uint64_t i0 = rdinstret();
    qadd_b0(&p, t_a, t_b, out_buf, T_N);
    uint64_t i1 = rdinstret();
    return check_and_report("add", out_buf, t_golden, T_N, i1 - i0, 0);
}
