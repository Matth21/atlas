import platform

from atlas.profile.hardware import HardwareProfiler, HardwareSpec


def test_hardware_spec_fields():
    spec = HardwareSpec(
        platform="darwin",
        chip="Apple M1 Pro",
        ram_total_gb=32.0,
        ram_available_gb=24.0,
        gpu_vendor="apple",
        gpu_cores=16,
        cpu_cores=10,
    )
    assert spec.platform == "darwin"
    assert spec.ram_total_gb == 32.0
    assert spec.gpu_vendor == "apple"


def test_detect_returns_hardware_spec():
    profiler = HardwareProfiler()
    spec = profiler.detect()
    assert isinstance(spec, HardwareSpec)
    assert spec.platform == platform.system().lower()
    assert spec.ram_total_gb > 0
    assert spec.cpu_cores > 0


def test_usable_memory():
    profiler = HardwareProfiler()
    spec = profiler.detect()
    usable = profiler.usable_memory_gb(overhead=0.3)
    assert usable > 0
    assert usable < spec.ram_total_gb
    assert abs(usable - spec.ram_total_gb * 0.7) < 0.1


def test_detect_chip_name_not_unknown():
    profiler = HardwareProfiler()
    spec = profiler.detect()
    if spec.platform == "darwin":
        assert spec.chip != "Unknown"
