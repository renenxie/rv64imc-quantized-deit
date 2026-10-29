// fp_counter.c — 計算「軟體浮點函式」被呼叫的次數，用來證明推論時沒有任何浮點運算
//
// RV64IMC 沒有浮點硬體，C/C++ 裡的 float/double 運算會被編譯成呼叫 libgcc 的軟體浮點
// 函式（__mulsf3、__adddf3…）或 libm（expf、sqrtf…）。連結時用 -Wl,--wrap=<函式>，
// 讓每次呼叫先經過這裡的計數器再轉給真正的實作。
// main.cc 在 Invoke() 前後讀取 g_fp_calls：差值必須是 0，代表推論過程完全沒有浮點運算；
// 預設的重播模式（離線參數表）下，整個程式結束時 g_fp_calls 也必須是 0。
#include <stdint.h>

uint64_t g_fp_calls;

#define WRAP2(ret, name, t1, t2)                     \
    ret __real_##name(t1, t2);                       \
    ret __wrap_##name(t1 a, t2 b) { g_fp_calls++; return __real_##name(a, b); }
#define WRAP1(ret, name, t1)                         \
    ret __real_##name(t1);                           \
    ret __wrap_##name(t1 a) { g_fp_calls++; return __real_##name(a); }

// 單精度（float）四則與比較
WRAP2(float, __addsf3, float, float)
WRAP2(float, __subsf3, float, float)
WRAP2(float, __mulsf3, float, float)
WRAP2(float, __divsf3, float, float)
WRAP2(long, __eqsf2, float, float)
WRAP2(long, __nesf2, float, float)
WRAP2(long, __ltsf2, float, float)
WRAP2(long, __lesf2, float, float)
WRAP2(long, __gtsf2, float, float)
WRAP2(long, __gesf2, float, float)
WRAP2(long, __unordsf2, float, float)
// 雙精度（double）四則與比較
WRAP2(double, __adddf3, double, double)
WRAP2(double, __subdf3, double, double)
WRAP2(double, __muldf3, double, double)
WRAP2(double, __divdf3, double, double)
WRAP2(long, __eqdf2, double, double)
WRAP2(long, __nedf2, double, double)
WRAP2(long, __ltdf2, double, double)
WRAP2(long, __ledf2, double, double)
WRAP2(long, __gtdf2, double, double)
WRAP2(long, __gedf2, double, double)
WRAP2(long, __unorddf2, double, double)
// 整數 ↔ 浮點轉換
WRAP1(int, __fixsfsi, float)
WRAP1(long, __fixsfdi, float)
WRAP1(unsigned, __fixunssfsi, float)
WRAP1(unsigned long, __fixunssfdi, float)
WRAP1(int, __fixdfsi, double)
WRAP1(long, __fixdfdi, double)
WRAP1(unsigned, __fixunsdfsi, double)
WRAP1(unsigned long, __fixunsdfdi, double)
WRAP1(float, __floatsisf, int)
WRAP1(float, __floatdisf, long)
WRAP1(float, __floatunsisf, unsigned)
WRAP1(float, __floatundisf, unsigned long)
WRAP1(double, __floatsidf, int)
WRAP1(double, __floatdidf, long)
WRAP1(double, __floatunsidf, unsigned)
WRAP1(double, __floatundidf, unsigned long)
WRAP1(double, __extendsfdf2, float)
WRAP1(float, __truncdfsf2, double)
WRAP1(float, __negsf2, float)
WRAP1(double, __negdf2, double)
// 常用數學函式（有些內部只用整數位元運算，所以另外計數）
WRAP1(float, sqrtf, float)
WRAP1(double, sqrt, double)
WRAP1(float, expf, float)
WRAP1(double, exp, double)
WRAP1(float, logf, float)
WRAP1(double, log, double)
WRAP1(float, tanhf, float)
WRAP1(double, tanh, double)
WRAP1(float, erfcf, float)
WRAP1(double, erfc, double)
WRAP1(float, roundf, float)
WRAP1(double, round, double)
WRAP1(float, floorf, float)
WRAP1(double, floor, double)
WRAP1(float, ceilf, float)
WRAP1(double, ceil, double)
WRAP2(float, powf, float, float)
WRAP2(double, pow, double, double)
WRAP2(float, frexpf, float, int*)
WRAP2(double, frexp, double, int*)
WRAP2(float, ldexpf, float, int)
WRAP2(double, ldexp, double, int)
