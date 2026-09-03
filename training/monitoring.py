"""Shared monitoring helpers for training runs (Phase 30). Live,
per-step signal (loss, learning rate, throughput, memory) DURING a run,
distinct from Phase 15's evaluate_checkpoint(): monitoring is what lets
you catch a problem (divergence, memory pressure, a stalled run) while
it's happening, not just after."""
import ctypes


def get_available_memory_mb():
    """Windows-specific: queries available physical memory directly via
    ctypes (stdlib -- no psutil dependency). Returns None on non-Windows
    platforms or if the call fails, rather than guessing; callers must
    handle that, not assume a number. Moved here from scripts/benchmark.py
    (Phase 22) so training/train.py can reuse it too, without training/
    depending on scripts/ -- scripts are entry points, not a library
    other project code should import from."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return stat.ullAvailPhys / (1024 * 1024)
    except Exception:
        return None
