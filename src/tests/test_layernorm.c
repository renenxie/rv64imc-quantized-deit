// test_layernorm.c — LayerNorm kernel 測試
#include "ops.h"
#include "test_common.h"
#include TEST_DATA

static int8_t out_buf[T_ROWS * T_C];

int main(void) {
    qlayernorm_params_t p = {
        .C = T_C, .in_zp = T_IN_ZP, .out_zp = T_OUT_ZP, .eps = T_EPS,
        .G = t_G, .B = t_B,
    };
    uint64_t i0 = rdinstret();
    qlayernorm_b0(&p, t_in, out_buf, T_ROWS);
    uint64_t i1 = rdinstret();
    return check_and_report("layernorm", out_buf, t_golden, T_ROWS * T_C, i1 - i0, 0);
}
