/* futex(2) on top of WaitOnAddress / WakeByAddress (Windows 8+). Only the calls vec_env.c makes. */
#pragma once
#include <windows.h>
#include <errno.h>
#include <stdint.h>
#define SYS_futex 202
static inline long tmnf_futex(uint32_t *addr, int op, uint32_t val)
{
	if (op == 128) {              /* FUTEX_WAIT_PRIVATE */
		uint32_t expected = val;
		if (*(volatile uint32_t *)addr != expected) { errno = EAGAIN; return -1; }
		WaitOnAddress(addr, &expected, sizeof(expected), INFINITE);
		return 0;
	}
	if ((int)val >= 0x7fffffff || (int)val > 1)
		WakeByAddressAll(addr);
	else
		WakeByAddressSingle(addr);
	return 0;
}
#define syscall(nr, addr, op, val, ...) tmnf_futex((addr), (op), (uint32_t)(val))
