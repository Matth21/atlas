"""Hardware detection for Atlas.

Detects platform, chip, memory, GPU, and CPU core information so the
compressor can size its working set appropriately for the host machine.
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareSpec:
    platform: str
    chip: str
    ram_total_gb: float
    ram_available_gb: float
    gpu_vendor: str
    gpu_cores: int
    cpu_cores: int


class HardwareProfiler:
    """Detects host hardware characteristics, caching the result."""

    def __init__(self):
        self._spec: Optional[HardwareSpec] = None

    def detect(self) -> HardwareSpec:
        if self._spec is not None:
            return self._spec
        sys_platform = platform.system().lower()
        if sys_platform == "darwin":
            self._spec = self._detect_macos()
        elif sys_platform == "linux":
            self._spec = self._detect_linux()
        else:
            self._spec = self._detect_fallback(sys_platform)
        return self._spec

    def usable_memory_gb(self, overhead: float = 0.3) -> float:
        spec = self.detect()
        return spec.ram_total_gb * (1.0 - overhead)

    def _detect_macos(self) -> HardwareSpec:
        ram_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        ram_gb = ram_bytes / (1024 ** 3)

        cpu_cores = os.cpu_count() or 1

        chip = "Unknown"
        try:
            sp_output = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"], text=True
            )
            for line in sp_output.splitlines():
                line = line.strip()
                if line.startswith("Chip:"):
                    chip = line.split(":", 1)[1].strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        gpu_cores = 0
        try:
            disp_output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True
            )
            for line in disp_output.splitlines():
                line = line.strip()
                if line.startswith("Total Number of Cores:"):
                    parts = line.split(":", 1)
                    gpu_cores = int("".join(c for c in parts[1] if c.isdigit()) or "0")
                    break
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        avail_gb = ram_gb
        try:
            vm = subprocess.check_output(["vm_stat"], text=True)
            page_size = 16384
            free_pages = 0
            for line in vm.splitlines():
                if "page size" in line.lower():
                    page_size = int("".join(c for c in line if c.isdigit()) or "16384")
                elif "Pages free" in line:
                    free_pages += int("".join(c for c in line.split(":")[1] if c.isdigit()) or "0")
                elif "Pages inactive" in line:
                    free_pages += int("".join(c for c in line.split(":")[1] if c.isdigit()) or "0")
            avail_gb = (free_pages * page_size) / (1024 ** 3)
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
            pass

        return HardwareSpec(
            platform="darwin",
            chip=chip,
            ram_total_gb=round(ram_gb, 1),
            ram_available_gb=round(avail_gb, 1),
            gpu_vendor="apple",
            gpu_cores=gpu_cores,
            cpu_cores=cpu_cores,
        )

    def _detect_linux(self) -> HardwareSpec:
        ram_gb = 0.0
        avail_gb = 0.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        ram_gb = kb / (1024 ** 2)
                    elif line.startswith("MemAvailable"):
                        avail_kb = int(line.split()[1])
                        avail_gb = avail_kb / (1024 ** 2)
        except FileNotFoundError:
            ram_gb = 0.0
            avail_gb = 0.0

        gpu_vendor = "none"
        gpu_cores = 0
        try:
            nv = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            if nv.strip():
                gpu_vendor = "nvidia"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return HardwareSpec(
            platform="linux",
            chip=platform.processor() or "Unknown",
            ram_total_gb=round(ram_gb, 1),
            ram_available_gb=round(avail_gb, 1),
            gpu_vendor=gpu_vendor,
            gpu_cores=gpu_cores,
            cpu_cores=os.cpu_count() or 1,
        )

    def _detect_fallback(self, sys_platform: str) -> HardwareSpec:
        """Best-effort detection for platforms without a dedicated path.

        Avoids a hard dependency on psutil (not installed in this project).
        If psutil happens to be available, use it for memory info; otherwise
        report zeros rather than crashing.
        """
        ram_gb = 0.0
        try:
            import psutil  # type: ignore

            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            ram_gb = 0.0

        return HardwareSpec(
            platform=sys_platform,
            chip=platform.processor() or "Unknown",
            ram_total_gb=round(ram_gb, 1),
            ram_available_gb=round(ram_gb * 0.7, 1),
            gpu_vendor="unknown",
            gpu_cores=0,
            cpu_cores=os.cpu_count() or 1,
        )
