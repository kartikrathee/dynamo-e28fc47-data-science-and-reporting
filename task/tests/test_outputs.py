#!/usr/bin/env python3
"""Verifier tests for the Lennard-Jones molecular dynamics task."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

APP = Path("/app")
N_PARTICLES = 1000
BOX_DIM = 10.0
TOTAL_STEPS = 100
TRAJECTORY_INTERVAL = 10
ENERGY_DRIFT_TOL = 0.10
MIN_PARTICLES_PER_SEC_PER_CORE = 100
MIN_PARALLEL_EFFICIENCY = 0.5
RDF_BIN_WIDTH = 0.1
RDF_MAX_R = 5.0


@pytest.fixture(scope="session", autouse=True)
def _ensure_app_workdir() -> None:
    os.chdir(APP)


def _read_energy_rows() -> list[dict[str, float | int]]:
    with (APP / "energy.log").open() as handle:
        lines = [line.strip() for line in handle if line.strip()]
    rows: list[dict[str, float | int]] = []
    for line in lines[1:]:
        step, ke, pe, total = line.split()
        rows.append({"step": int(step), "ke": float(ke), "pe": float(pe), "total": float(total)})
    return rows


def _temperature_from_ke(ke: float) -> float:
    return (2.0 * ke) / (3.0 * N_PARTICLES)


def _parse_temperature_stats() -> dict[str, float]:
    content = (APP / "temperature_stats.txt").read_text()
    pattern = r"^(Mean|Std Dev|Min|Max) Temperature:\s*([0-9.eE+-]+)"
    stats: dict[str, float] = {}
    for line in content.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            stats[match.group(1)] = float(match.group(2))
    return stats


def _parse_last_trajectory_frame() -> list[tuple[float, float, float]]:
    lines = (APP / "trajectory.xyz").read_text().splitlines()
    lines_per_frame = N_PARTICLES + 2
    expected_frames = (TOTAL_STEPS // TRAJECTORY_INTERVAL) + 1
    frame_start = (expected_frames - 1) * lines_per_frame
    frame = lines[frame_start : frame_start + lines_per_frame]
    return [
        (float(parts[1]), float(parts[2]), float(parts[3]))
        for parts in (line.split() for line in frame[2:])
    ]


def _minimum_image(delta: float) -> float:
    return delta - BOX_DIM * round(delta / BOX_DIM)


def _recompute_rdf(positions: list[tuple[float, float, float]]) -> dict[float, float]:
    num_bins = int(RDF_MAX_R / RDF_BIN_WIDTH) + 1
    counts = [0.0] * num_bins
    for i, pos_i in enumerate(positions):
        for j in range(i + 1, len(positions)):
            pos_j = positions[j]
            dx = _minimum_image(pos_i[0] - pos_j[0])
            dy = _minimum_image(pos_i[1] - pos_j[1])
            dz = _minimum_image(pos_i[2] - pos_j[2])
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r <= RDF_MAX_R:
                counts[int(r / RDF_BIN_WIDTH)] += 2.0

    density = len(positions) / (BOX_DIM ** 3)
    rdf: dict[float, float] = {}
    for idx, count in enumerate(counts):
        r = idx * RDF_BIN_WIDTH
        shell_volume = 4.0 * math.pi * r * r * RDF_BIN_WIDTH
        if shell_volume > 0:
            norm = len(positions) * density * shell_volume
            rdf[round(r, 1)] = count / norm if norm > 0 else 0.0
        else:
            rdf[0.0] = 0.0
    return rdf


def test_source_files_exist() -> None:
    """Required source artifacts must exist under /app and be non-empty."""
    required = ("simulation.cpp", "Makefile", "run_simulation.sh", "particles.dat")
    missing = [name for name in required if not (APP / name).exists()]
    assert missing == []


def test_simulation_uses_openmp() -> None:
    """The C++ simulator must include OpenMP pragmas and headers."""
    content = (APP / "simulation.cpp").read_text()
    assert "#pragma omp" in content and ("omp.h" in content or "_OPENMP" in content)


def test_makefile_supports_openmp() -> None:
    """The Makefile must compile with -fopenmp."""
    assert "fopenmp" in (APP / "Makefile").read_text()


def test_run_script_sets_thread_count() -> None:
    """The run script must set OMP_NUM_THREADS for parallel execution."""
    assert "OMP_NUM_THREADS" in (APP / "run_simulation.sh").read_text()


def test_output_files_exist() -> None:
    """All required simulation and analysis outputs must exist under /app."""
    required = (
        "trajectory.xyz",
        "energy.log",
        "timing.log",
        "temperature_stats.txt",
        "rdf.dat",
        "performance.txt",
        "summary.json",
    )
    missing = [name for name in required if not (APP / name).exists()]
    assert missing == []


def test_trajectory_format() -> None:
    """Trajectory must contain 11 VMD-compatible frames for 1000 particles."""
    lines = (APP / "trajectory.xyz").read_text().splitlines()
    expected_frames = (TOTAL_STEPS // TRAJECTORY_INTERVAL) + 1
    assert len(lines) == expected_frames * (N_PARTICLES + 2)


def test_energy_log_row_count() -> None:
    """Energy log must contain one row for each step from 0 through 100."""
    assert len(_read_energy_rows()) == TOTAL_STEPS + 1


def test_energy_log_total_matches_components() -> None:
    """Each energy row must satisfy total = kinetic + potential."""
    rows = _read_energy_rows()
    assert all(
        math.isclose(float(row["total"]), float(row["ke"]) + float(row["pe"]), rel_tol=1e-4, abs_tol=0.05)
        for row in rows
    )


def test_energy_conservation() -> None:
    """Relative total-energy drift from step 0 to 100 must stay below 10%."""
    rows = _read_energy_rows()
    drift = abs(float(rows[-1]["total"]) - float(rows[0]["total"])) / abs(float(rows[0]["total"]))
    assert drift < ENERGY_DRIFT_TOL


def test_temperature_stats_match_energy_log() -> None:
    """Temperature statistics must match values derived from /app/energy.log."""
    rows = _read_energy_rows()
    temps = [_temperature_from_ke(float(row["ke"])) for row in rows]
    stats = _parse_temperature_stats()
    assert math.isclose(stats["Mean"], sum(temps) / len(temps), rel_tol=1e-3)


def test_rdf_matches_trajectory() -> None:
    """RDF values must match recomputation from the last trajectory frame."""
    expected = _recompute_rdf(_parse_last_trajectory_frame())
    lines = [line.strip() for line in (APP / "rdf.dat").read_text().splitlines() if line.strip()]
    assert len(lines) == int(RDF_MAX_R / RDF_BIN_WIDTH) + 1
    for line in lines:
        r_val, gr_val = map(float, line.split())
        if r_val in expected:
            assert math.isclose(gr_val, expected[r_val], rel_tol=0.05, abs_tol=0.05)


def test_timing_log_has_positive_values() -> None:
    """Timing log must report positive measured durations."""
    numbers = [float(value) for value in re.findall(r"[0-9]+(?:\.[0-9]+)?", (APP / "timing.log").read_text())]
    assert any(number > 0 for number in numbers)


def test_performance_throughput() -> None:
    """Performance report must show throughput above the minimum threshold."""
    content = (APP / "performance.txt").read_text()
    pps = next(
        float(line.split(":")[1].strip().split()[0])
        for line in content.splitlines()
        if "Particles per second per core" in line
    )
    assert pps > MIN_PARTICLES_PER_SEC_PER_CORE


def test_performance_compute_fraction() -> None:
    """At least half of runtime must be spent in force calculation and integration."""
    timing = (APP / "timing.log").read_text()
    force_time = float(re.search(r"Force calculation time:\s*([0-9.]+)", timing).group(1))
    integ_time = float(re.search(r"Integration time:\s*([0-9.]+)", timing).group(1))
    total_time = float(re.search(r"Total runtime:\s*([0-9.]+)", timing).group(1))
    assert (force_time + integ_time) / total_time >= MIN_PARALLEL_EFFICIENCY


def test_summary_json_passes_validation() -> None:
    """Summary JSON must report PASS with drift consistent with energy.log."""
    summary = json.loads((APP / "summary.json").read_text())
    rows = _read_energy_rows()
    drift = abs(float(rows[-1]["total"]) - float(rows[0]["total"])) / abs(float(rows[0]["total"]))
    assert summary["validation_status"] == "PASS"
    assert math.isclose(summary["energy_conservation"], drift, rel_tol=1e-3)


def test_simulation_compiles() -> None:
    """Submitted artifacts must compile cleanly with make."""
    subprocess.run(["make", "clean"], cwd=APP, check=False, capture_output=True)
    result = subprocess.run(["make"], cwd=APP, capture_output=True, text=True)
    assert result.returncode == 0
