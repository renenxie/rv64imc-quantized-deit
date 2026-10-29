// syscalls.c — bare-metal 執行期支援：HTIF 輸出、printf、exit、memcpy/memset
//
// 透過 Spike 的 HTIF（tohost/fromhost）把字串交給主機印出，完全不依賴 pk / newlib。
// printf 只支援整數格式（%d %i %u %x %s %c %%，可加 l / ll），確保不會引入浮點指令。
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64)));
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64)));

#define SYS_write 64

static uint64_t htif_syscall(uint64_t n, uint64_t a0, uint64_t a1, uint64_t a2) {
    static volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
    magic_mem[0] = n;
    magic_mem[1] = a0;
    magic_mem[2] = a1;
    magic_mem[3] = a2;
    __sync_synchronize();
    tohost = (uint64_t)(uintptr_t)magic_mem;
    while (fromhost == 0) ;
    fromhost = 0;
    __sync_synchronize();
    return magic_mem[0];
}

__attribute__((noreturn)) void exit(int code) {
    tohost = ((uint64_t)code << 1) | 1;
    for (;;) ;
}

// ---- 輸出緩衝 ----
static char obuf[256];
static size_t olen;

static void flush(void) {
    if (olen) {
        htif_syscall(SYS_write, 1, (uint64_t)(uintptr_t)obuf, olen);
        olen = 0;
    }
}

static void outc(char c) {
    obuf[olen++] = c;
    if (c == '\n' || olen == sizeof(obuf)) flush();
}

static void outs(const char *s) {
    while (*s) outc(*s++);
}

static void outnum(uint64_t v, int base, int neg) {
    char tmp[24];
    int i = 0;
    if (neg) outc('-');
    do {
        int d = (int)(v % base);
        tmp[i++] = (char)(d < 10 ? '0' + d : 'a' + d - 10);
        v /= base;
    } while (v);
    while (i) outc(tmp[--i]);
}

int printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    for (; *fmt; fmt++) {
        if (*fmt != '%') { outc(*fmt); continue; }
        fmt++;
        int lng = 0;
        while (*fmt == 'l') { lng++; fmt++; }
        switch (*fmt) {
        case 'd': case 'i': {
            int64_t v = lng ? va_arg(ap, int64_t) : va_arg(ap, int);
            outnum(v < 0 ? (uint64_t)(-v) : (uint64_t)v, 10, v < 0);
            break;
        }
        case 'u': outnum(lng ? va_arg(ap, uint64_t) : va_arg(ap, unsigned), 10, 0); break;
        case 'x': outnum(lng ? va_arg(ap, uint64_t) : va_arg(ap, unsigned), 16, 0); break;
        case 's': outs(va_arg(ap, const char *)); break;
        case 'c': outc((char)va_arg(ap, int)); break;
        case '%': outc('%'); break;
        default:  outc('%'); outc(*fmt); break;
        }
    }
    va_end(ap);
    flush();
    return 0;
}

void handle_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval) {
    printf("TRAP mcause=%lu mepc=0x%lx mtval=0x%lx\n", mcause, mepc, mtval);
    exit(128 + (int)(mcause & 0x3f));
}

// GCC 可能會自動產生 memcpy / memset 呼叫
void *memcpy(void *dst, const void *src, size_t n) {
    char *d = dst;
    const char *s = src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, size_t n) {
    char *d = dst;
    while (n--) *d++ = (char)c;
    return dst;
}
