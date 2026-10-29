// newlib_stubs.c — 讓 newlib（C 標準函式庫）在 Spike bare-metal 上運作的系統呼叫
//
// LiteRT Micro 是 C++，會用到 newlib 的 vfprintf、memcpy 等函式；
// newlib 底層需要 _write、_exit、_sbrk 等「系統呼叫」，這裡用 Spike 的 HTIF 實作。
// （與 src/common/syscalls.c 相同的 HTIF 機制，但不自己實作 printf，改用 newlib 的。）
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>

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

int _write(int fd, const void *buf, size_t len) {
    (void)fd;  // stdout 與 stderr 都送到主機的標準輸出
    htif_syscall(SYS_write, 1, (uint64_t)(uintptr_t)buf, len);
    return (int)len;
}

__attribute__((noreturn)) void _exit(int code) {
    tohost = ((uint64_t)code << 1) | 1;
    for (;;) ;
}

extern char __heap_start[], __heap_end[];
static char *heap_ptr = __heap_start;

void *_sbrk(ptrdiff_t incr) {
    if (heap_ptr + incr > __heap_end) {
        errno = ENOMEM;
        return (void *)-1;
    }
    char *prev = heap_ptr;
    heap_ptr += incr;
    return prev;
}

int _close(int fd) { (void)fd; return -1; }
int _read(int fd, void *buf, size_t len) { (void)fd; (void)buf; (void)len; return 0; }
int _lseek(int fd, int off, int whence) { (void)fd; (void)off; (void)whence; return 0; }
int _isatty(int fd) { (void)fd; return 1; }
int _fstat(int fd, struct stat *st) { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _getpid(void) { return 1; }
int _kill(int pid, int sig) { (void)pid; (void)sig; errno = EINVAL; return -1; }

// crt0.S 的例外處理會呼叫這裡（例如執行到非法指令）
void handle_trap(uint64_t mcause, uint64_t mepc, uint64_t mtval) {
    static const char hex[] = "0123456789abcdef";
    char msg[96] = "TRAP mcause=0x";
    int p = 14;
    uint64_t v[3] = {mcause, mepc, mtval};
    const char *lab[3] = {"", " mepc=0x", " mtval=0x"};
    for (int k = 0; k < 3; k++) {
        for (const char *s = lab[k]; *s; s++) msg[p++] = *s;
        for (int i = 60; i >= 0; i -= 4) msg[p++] = hex[(v[k] >> i) & 0xf];
    }
    msg[p++] = '\n';
    _write(1, msg, p);
    _exit(128 + (int)(mcause & 0x3f));
}
