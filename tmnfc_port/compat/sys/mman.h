/* mmap of a whole read-only file (MAP_PRIVATE, private copy): read it into memory. */
#pragma once
#include <stdlib.h>
#include <io.h>
#include <sys/types.h>
#include <fcntl.h>
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_PRIVATE 2
#define MAP_FAILED ((void *)-1)
static inline void *mmap(void *a, size_t len, int prot, int flags, int fd, long off)
{
	(void)a; (void)prot; (void)flags;
	unsigned char *p = (unsigned char *)malloc(len ? len : 1);
	if (!p) return MAP_FAILED;
	_setmode(fd, _O_BINARY);  /* open() defaults to text mode on Windows */
	if (_lseeki64(fd, off, SEEK_SET) < 0) { free(p); return MAP_FAILED; }
	size_t got = 0;
	while (got < len) {
		int n = _read(fd, p + got, (unsigned)((len - got) > (1u << 30) ? (1u << 30) : (len - got)));
		if (n <= 0) { free(p); return MAP_FAILED; }
		got += (size_t)n;
	}
	return p;
}
static inline int munmap(void *p, size_t len) { (void)len; free(p); return 0; }
static inline int mprotect(void *p, size_t len, int prot) { (void)p; (void)len; (void)prot; return 0; }
