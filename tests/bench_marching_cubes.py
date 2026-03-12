#!/usr/bin/env python3
"""Benchmark marching_cubes to diagnose surface extraction slowness.

Run on both machines to compare:
    python tests/bench_marching_cubes.py

Checks:
  1. CPU info and architecture
  2. Whether skimage C extensions are compiled (vs pure Python fallback)
  3. NumPy/skimage build info
  4. Actual marching_cubes timing at different grid resolutions
"""

import platform
import sys
import time


def log(msg: str) -> None:
    print(msg, flush=True)


def cpu_info() -> None:
    log("=" * 60)
    log("SYSTEM INFO")
    log("=" * 60)
    log(f"Python:       {sys.version}")
    log(f"Platform:     {platform.platform()}")
    log(f"Machine:      {platform.machine()}")
    log(f"Processor:    {platform.processor() or 'unknown'}")

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    log(f"CPU:          {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        pass

    try:
        import os
        log(f"CPU count:    {os.cpu_count()}")
    except Exception:
        pass


def check_extensions() -> None:
    log("")
    log("=" * 60)
    log("LIBRARY CHECK")
    log("=" * 60)

    import numpy as np
    log(f"NumPy:        {np.__version__}")
    log(f"NumPy BLAS:   {np.show_config(mode='dicts').get('Build Dependencies', {}).get('blas', 'unknown')}")

    import skimage
    log(f"scikit-image: {skimage.__version__}")

    # Check if the C extension is compiled
    try:
        from skimage.measure import _marching_cubes_lewiner
        log(f"Lewiner C ext: LOADED ({_marching_cubes_lewiner.__file__})")
    except ImportError:
        log("Lewiner C ext: NOT FOUND (using pure Python fallback - THIS IS SLOW)")

    try:
        from skimage.measure import _marching_cubes_classic
        log(f"Classic C ext: LOADED ({_marching_cubes_classic.__file__})")
    except ImportError:
        log("Classic C ext: not found")

    # Check if Cython extensions are .so (compiled) or .py (interpreted)
    try:
        import skimage.measure._marching_cubes_lewiner as mcl
        ext_file = mcl.__file__
        if ext_file.endswith(".so") or ext_file.endswith(".pyd"):
            log("Extension type: COMPILED (native)")
        elif ext_file.endswith(".py"):
            log("Extension type: PURE PYTHON (expect 50-100x slower)")
        else:
            log(f"Extension type: {ext_file}")
    except Exception as e:
        log(f"Extension check failed: {e}")


def bench_marching_cubes() -> None:
    import numpy as np
    from skimage.measure import marching_cubes

    log("")
    log("=" * 60)
    log("MARCHING CUBES BENCHMARK")
    log("=" * 60)

    resolutions = [65, 129, 257, 385]  # octree_resolution + 1

    for grid_size in resolutions:
        # Create a sphere-like scalar field
        coords = np.linspace(-1, 1, grid_size)
        x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
        volume = x**2 + y**2 + z**2 - 0.5  # sphere at r=sqrt(0.5)
        volume = volume.astype(np.float32)

        total_points = grid_size**3
        mem_mb = volume.nbytes / 1024 / 1024

        # Warmup
        if grid_size <= 129:
            marching_cubes(volume, 0.0, method="lewiner")

        # Timed run
        t0 = time.monotonic()
        verts, faces, normals, values = marching_cubes(volume, 0.0, method="lewiner")
        elapsed = time.monotonic() - t0

        log(
            f"  grid={grid_size:>4}  points={total_points:>12,}  "
            f"mem={mem_mb:>7.1f}MB  verts={len(verts):>8,}  "
            f"time={elapsed:>7.3f}s"
        )

    log("")
    log("The 385 grid (octree_resolution=384) is what the API uses.")
    log("If this takes >10s, marching_cubes C extensions may not be compiled.")


def bench_numpy() -> None:
    """Quick NumPy benchmark to check general CPU/memory performance."""
    import numpy as np

    log("")
    log("=" * 60)
    log("NUMPY BENCHMARK (memory bandwidth indicator)")
    log("=" * 60)

    size = 385**3
    a = np.random.randn(size).astype(np.float32)

    t0 = time.monotonic()
    _ = a > 0  # simple comparison, memory-bound
    elapsed = time.monotonic() - t0
    log(f"  np.compare (57M floats): {elapsed:.3f}s")

    t0 = time.monotonic()
    _ = np.sort(a)
    elapsed = time.monotonic() - t0
    log(f"  np.sort    (57M floats): {elapsed:.3f}s")


if __name__ == "__main__":
    cpu_info()
    check_extensions()
    bench_numpy()
    bench_marching_cubes()
