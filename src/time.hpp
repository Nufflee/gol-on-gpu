#pragma once

#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#elif _POSIX_C_SOURCE >= 199309L
#include <time.h>
#else
#error "Timing not supported on this platform. Please report this to the author. If you do not need timing, feel free to comment this line out"
#endif

inline double get_time_secs() {
#if defined(_WIN32)
	LARGE_INTEGER counts_per_sec, count;
	QueryPerformanceFrequency(&counts_per_sec);
	QueryPerformanceCounter(&count);

	return (double)count.QuadPart / (double)counts_per_sec.QuadPart;
#elif _POSIX_C_SOURCE >= 199309L
	timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);

	return (double)time.tv_sec + (double)time.tv_nsec / 1e9;
#endif
}

inline double get_timer_resolution() {
#if defined(_WIN32)
	LARGE_INTEGER counts_per_sec;
	QueryPerformanceFrequency(&counts_per_sec);

	return 1.0 / (double)counts_per_sec.QuadPart;
#elif _POSIX_C_SOURCE >= 199309L
	timespec time;
	clock_getres(CLOCK_MONOTONIC, &time);

	return (double)time.tv_sec + (double)time.tv_nsec / 1e9;
#endif
}