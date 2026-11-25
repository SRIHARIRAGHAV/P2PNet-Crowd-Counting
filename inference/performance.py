import ctypes
import os
import sys
import time
from collections import deque


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, value):
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0


class RollingPerformance:
    def __init__(self, window_size=30):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        self._samples = deque(maxlen=window_size)

    def add(self, preprocessing_ms, inference_ms, postprocessing_ms, total_ms):
        self._samples.append((
            float(preprocessing_ms),
            float(inference_ms),
            float(postprocessing_ms),
            float(total_ms),
        ))

    def snapshot(self):
        if not self._samples:
            return {
                "frames": 0,
                "preprocessing_ms": 0.0,
                "inference_ms": 0.0,
                "postprocessing_ms": 0.0,
                "total_ms": 0.0,
                "inference_fps": 0.0,
                "processing_fps": 0.0,
            }

        frame_count = len(self._samples)
        totals = [sum(sample[index] for sample in self._samples) for index in range(4)]
        return {
            "frames": frame_count,
            "preprocessing_ms": totals[0] / frame_count,
            "inference_ms": totals[1] / frame_count,
            "postprocessing_ms": totals[2] / frame_count,
            "total_ms": totals[3] / frame_count,
            "inference_fps": (
                1000.0 * frame_count / totals[1] if totals[1] > 0 else 0.0
            ),
            "processing_fps": (
                1000.0 * frame_count / totals[3] if totals[3] > 0 else 0.0
            ),
        }


if os.name == "nt":
    from ctypes import wintypes

    class _FileTime(ctypes.Structure):
        _fields_ = [
            ("low", wintypes.DWORD),
            ("high", wintypes.DWORD),
        ]

    class _ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    ]
    _psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]


def _filetime_seconds(value):
    ticks = (value.high << 32) | value.low
    return ticks / 10_000_000.0


def _process_cpu_seconds():
    if os.name == "nt":
        creation = _FileTime()
        exit_time = _FileTime()
        kernel = _FileTime()
        user = _FileTime()
        succeeded = _kernel32.GetProcessTimes(
            _kernel32.GetCurrentProcess(),
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        if not succeeded:
            raise ctypes.WinError(ctypes.get_last_error())
        return _filetime_seconds(kernel) + _filetime_seconds(user)
    process_times = os.times()
    return process_times.user + process_times.system


def _resident_memory_bytes():
    if os.name == "nt":
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        succeeded = _psapi.GetProcessMemoryInfo(
            _kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.working_set_size, counters.peak_working_set_size

    statm_path = "/proc/self/statm"
    if os.path.isfile(statm_path):
        with open(statm_path, "r", encoding="ascii") as statm_file:
            resident_pages = int(statm_file.read().split()[1])
        current_bytes = resident_pages * os.sysconf("SC_PAGE_SIZE")
    else:
        current_bytes = None

    import resource

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    return current_bytes or peak_bytes, peak_bytes


class ProcessResourceMonitor:
    def __init__(self):
        self._logical_processors = max(1, os.cpu_count() or 1)
        self._last_cpu_seconds = _process_cpu_seconds()
        self._last_wall_seconds = time.perf_counter()
        self._peak_memory_mb = 0.0

    def sample(self):
        now = time.perf_counter()
        cpu_seconds = _process_cpu_seconds()
        elapsed = now - self._last_wall_seconds
        cpu_percent = 0.0
        if elapsed > 0:
            cpu_percent = (
                100.0
                * (cpu_seconds - self._last_cpu_seconds)
                / (elapsed * self._logical_processors)
            )
        self._last_cpu_seconds = cpu_seconds
        self._last_wall_seconds = now

        current_bytes, process_peak_bytes = _resident_memory_bytes()
        memory_mb = current_bytes / (1024.0 * 1024.0)
        self._peak_memory_mb = max(
            self._peak_memory_mb,
            memory_mb,
            process_peak_bytes / (1024.0 * 1024.0),
        )
        return {
            "cpu_percent": max(0.0, min(100.0, cpu_percent)),
            "memory_mb": memory_mb,
            "peak_memory_mb": self._peak_memory_mb,
        }
