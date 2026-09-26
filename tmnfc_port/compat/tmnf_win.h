/* Windows (MinGW via zig cc) shims for TMNF-C's few Linux calls. Force-included. */
#pragma once
#ifdef _WIN32
#include <pthread.h>
#include <string.h>
#include <errno.h>
#define CPU_SETSIZE 1024
typedef struct { unsigned char bits[CPU_SETSIZE / 8]; } cpu_set_t;
#define CPU_ZERO(s) memset((s), 0, sizeof(cpu_set_t))
#define CPU_SET(c, s) ((s)->bits[(c) / 8] |= (unsigned char)(1u << ((c) % 8)))
#define CPU_ISSET(c, s) (((s)->bits[(c) / 8] >> ((c) % 8)) & 1u)
/* Pinning is an optimisation only: accept and ignore it. */
static inline int pthread_setaffinity_np(pthread_t t, size_t n, const cpu_set_t *s) { (void)t; (void)n; (void)s; return 0; }
static inline int pthread_getaffinity_np(pthread_t t, size_t n, cpu_set_t *s) { (void)t; memset(s, 0xff, n); return 0; }
#endif
#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(align, size) _aligned_malloc((size), (align))
#define TMNF_ALIGNED_FREE(p) _aligned_free(p)
#else
#define TMNF_ALIGNED_FREE(p) free(p)
#endif
#ifdef _WIN32
/* __builtin_cpu_supports needs libgcc's __cpu_model, which zig's Windows runtime lacks. */
#include <cpuid.h>
#include <immintrin.h>
static inline int tmnf_cpu_supports(const char *f)
{
	unsigned a, b, c, d;
	if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27)))   /* OSXSAVE */
		return 0;
	unsigned long long xcr0 = _xgetbv(0);
	if (!__get_cpuid_count(7, 0, &a, &b, &c, &d))
		return 0;
	if (!strcmp(f, "avx2"))
		return (xcr0 & 6) == 6 && (b & (1u << 5));
	if (!strcmp(f, "avx512f"))
		return (xcr0 & 0xe6) == 0xe6 && (b & (1u << 16));
	if (!strcmp(f, "avx512dq"))
		return (xcr0 & 0xe6) == 0xe6 && (b & (1u << 17));
	return 0;
}
#define __builtin_cpu_supports(f) tmnf_cpu_supports(f)
#endif
