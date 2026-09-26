/* Load a DLL into the running TmForever.exe (32-bit build: LoadLibraryA has the same address
 * in every 32-bit process of the session). Nothing in the game folder changes.
 *
 *     inject.exe C:\full\path\TMNFTracer.dll [pid]
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DWORD find_game(void)
{
	PROCESSENTRY32 e;
	DWORD pid = 0;
	HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (snap == INVALID_HANDLE_VALUE)
		return 0;
	e.dwSize = sizeof(e);
	if (Process32First(snap, &e)) {
		do {
			if (_stricmp(e.szExeFile, "TmForever.exe") == 0) {
				if (pid != 0) {
					fprintf(stderr, "inject: several TmForever.exe running, pass a pid\n");
					CloseHandle(snap);
					return 0;
				}
				pid = e.th32ProcessID;
			}
		} while (Process32Next(snap, &e));
	}
	CloseHandle(snap);
	return pid;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: inject DLL [pid]\n");
		return 2;
	}
	const char *dll = argv[1];
	DWORD pid = argc > 2 ? (DWORD)strtoul(argv[2], NULL, 10) : find_game();
	if (pid == 0) {
		fprintf(stderr, "inject: no game process\n");
		return 1;
	}
	HANDLE proc = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
	if (proc == NULL) {
		fprintf(stderr, "inject: OpenProcess(%lu) failed: %lu\n", pid, GetLastError());
		return 1;
	}
	size_t n = strlen(dll) + 1;
	void *remote = VirtualAllocEx(proc, NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
	if (remote == NULL || !WriteProcessMemory(proc, remote, dll, n, NULL)) {
		fprintf(stderr, "inject: cannot write the path: %lu\n", GetLastError());
		return 1;
	}
	FARPROC load = GetProcAddress(GetModuleHandleA("kernel32.dll"), "LoadLibraryA");
	HANDLE thread = CreateRemoteThread(proc, NULL, 0, (LPTHREAD_START_ROUTINE)load, remote, 0, NULL);
	if (thread == NULL) {
		fprintf(stderr, "inject: CreateRemoteThread failed: %lu\n", GetLastError());
		return 1;
	}
	WaitForSingleObject(thread, 30000);
	DWORD module = 0;
	GetExitCodeThread(thread, &module);
	CloseHandle(thread);
	VirtualFreeEx(proc, remote, 0, MEM_RELEASE);
	CloseHandle(proc);
	if (module == 0) {
		fprintf(stderr, "inject: LoadLibraryA returned NULL in pid %lu\n", pid);
		return 1;
	}
	printf("inject: loaded into pid %lu at 0x%08lx\n", pid, module);
	return 0;
}
