#pragma once

#include <stdio.h>

#if defined(_MSC_VER)
#include <windows.h>
#endif

inline double get_time_secs() {

#if defined(_MSC_VER)
	LARGE_INTEGER counts_per_sec, count;
	QueryPerformanceFrequency(&counts_per_sec);
	QueryPerformanceCounter(&count);
	return (double)count.QuadPart / (double)counts_per_sec.QuadPart;
#else
	#error "Not implemented yet"
#endif
}
