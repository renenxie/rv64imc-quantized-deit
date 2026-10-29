// test_matmul.c — Attention 矩陣乘法 kernel 測試（Q·Kᵀ 或 P·V）
#include "ops.h"
#include "test_common.h"
#include TEST_DATA

static int8_t out_buf[T_M * T_N];

int main(void) {
    qmatmul_params_t p = {
        .M = T_M, .K = T_K, .N = T_N,
        .a_zp = T_AZP, .b_zp = T_BZP, .out_zp = T_OZP,
        .quant_mult = T_QM, .shift = T_SHIFT,
        .act_min = -128, .act_max = 127,
    };
    uint64_t i0 = rdinstret();
    qmatmul_b0(&p, t_a, t_b, out_buf);
    uint64_t i1 = rdinstret();
    return check_and_report("matmul", out_buf, t_golden, T_M * T_N, i1 - i0,
                            (uint64_t)T_M * T_K * T_N);
}
