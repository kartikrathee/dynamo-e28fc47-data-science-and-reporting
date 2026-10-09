#!/usr/bin/env python3
"""Verifier tests for the Lennard-Jones molecular dynamics task."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import pytest

APP = Path("/app")
TESTS_DIR = Path("/tests")
REFERENCE_PATH = TESTS_DIR / "reference" / "expected.json"
N_PARTICLES = 1000
BOX_DIM = 10.0
TOTAL_STEPS = 500
DT = 0.0005
TRAJECTORY_INTERVAL = 10
OPENMP_THREADS = 4
ENERGY_DRIFT_TOL = 0.05
RDF_BIN_WIDTH = 0.1
RDF_MAX_R = 5.0
REF_STEP0_RTOL = 1e-3  # deterministic IC; tight band
REF_TRAJECTORY_RTOL = 2e-3  # mid-run energies; fp/OpenMP among valid NVE impls
RERUN_STEP0_RTOL = 1e-4
IC_VEL_ABS_TOL = 1e-5  # seed-42 velocities; primary difficulty gate
COM_VEL_ABS_TOL = 0.05
COM_VEL_REF_ABS_TOL = 1e-3
MIN_PAIR_ABS_TOL = 1e-4
RDF_RTOL = 0.02
RDF_ABS_TOL = 0.01
DERIVED_RTOL = 1e-3
PERFORMANCE_RTOL = 0.02
_MIN_FORCE_SPEEDUP = 1.10
_TIMING_OVERHEAD_MAX = 0.18
_SERIAL_TOTAL_PADDING_RATIO = 1.15


_REQUIRED_REFERENCE_KEYS = (
    "particle0",
    "particle999",
    "particle42",
    "step0",
    "step0_temperature",
    "step1",
    "step100",
    "step250",
    "step400",
    "step500",
    "particle0_step500",
    "particle42_step500",
    "particle999_step500",
    "com_velocity",
    "min_pair_distance",
    "rdf_g0",
)


@pytest.fixture(scope="session")
def reference() -> dict:
    assert REFERENCE_PATH.is_file(), (
        f"Missing verifier ground truth at {REFERENCE_PATH}. "
        "Ensure tests/reference/expected.json is shipped with the tests/ overlay."
    )
    data = json.loads(REFERENCE_PATH.read_text())
    missing = [key for key in _REQUIRED_REFERENCE_KEYS if key not in data]
    assert missing == [], f"expected.json missing keys: {missing}"
    return data


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


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _parse_timing_log() -> dict[str, float]:
    """Parse the three timing.log lines defined in instruction §2."""
    content = (APP / "timing.log").read_text()
    labels = {
        "force": r"^Force calculation time:\s*([0-9.]+)\s*s\s*$",
        "integration": r"^Integration time:\s*([0-9.]+)\s*s\s*$",
        "total": r"^Total runtime:\s*([0-9.]+)\s*s\s*$",
    }
    values: dict[str, float] = {}
    for key, pattern in labels.items():
        match = re.search(pattern, content, re.MULTILINE)
        assert match is not None, f"timing.log missing instruction §2 line for {key}"
        values[key] = float(match.group(1))
    return values


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


def _read_particle_rows() -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for line in (APP / "particles.dat").read_text().splitlines():
        pid, x, y, z, vx, vy, vz = line.split()
        rows.append(
            {
                "id": int(pid),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "vx": float(vx),
                "vy": float(vy),
                "vz": float(vz),
            }
        )
    return rows


def _particle_positions(rows: list[dict[str, float | int]]) -> list[tuple[float, float, float]]:
    return [(float(row["x"]), float(row["y"]), float(row["z"])) for row in rows]


def _com_velocity(rows: list[dict[str, float | int]]) -> dict[str, float]:
    n = len(rows)
    return {
        "vx": sum(float(row["vx"]) for row in rows) / n,
        "vy": sum(float(row["vy"]) for row in rows) / n,
        "vz": sum(float(row["vz"]) for row in rows) / n,
    }


def _min_pair_distance(positions: list[tuple[float, float, float]]) -> float:
    min_r = float("inf")
    for i, pos_i in enumerate(positions):
        for j in range(i + 1, len(positions)):
            pos_j = positions[j]
            dx = _minimum_image(pos_i[0] - pos_j[0])
            dy = _minimum_image(pos_i[1] - pos_j[1])
            dz = _minimum_image(pos_i[2] - pos_j[2])
            min_r = min(min_r, math.sqrt(dx * dx + dy * dy + dz * dz))
    return min_r


def _energy_row(step: int) -> dict[str, float | int]:
    for row in _read_energy_rows():
        if int(row["step"]) == step:
            return row
    raise AssertionError(f"energy.log missing step {step}")


def _missing_or_empty(names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for name in names:
        path = APP / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    return missing


def _particle_row(particle_id: int) -> dict[str, float | int]:
    for row in _read_particle_rows():
        if int(row["id"]) == particle_id:
            return row
    raise AssertionError(f"particles.dat missing particle {particle_id}")


def _parse_scaling_force_times() -> tuple[float, float, float, float]:
    content = (APP / "scaling.txt").read_text()
    parallel_runtime = float(re.search(r"Parallel runtime \(4 threads\):\s*([0-9.]+)", content).group(1))
    serial_runtime = float(re.search(r"Serial runtime \(1 thread\):\s*([0-9.]+)", content).group(1))
    parallel_force = float(re.search(r"Parallel force time \(4 threads\):\s*([0-9.]+)", content).group(1))
    serial_force = float(re.search(r"Serial force time \(1 thread\):\s*([0-9.]+)", content).group(1))
    return parallel_runtime, serial_runtime, parallel_force, serial_force


def _parse_trajectory_frames() -> list[list[tuple[float, float, float]]]:
    return [[(p[0], p[1], p[2]) for p in frame] for frame in _parse_trajectory_frames_extended()]


def _parse_trajectory_frames_extended() -> list[list[tuple[float, float, float, float, float, float]]]:
    lines = (APP / "trajectory.xyz").read_text().splitlines()
    frames: list[list[tuple[float, float, float, float, float, float]]] = []
    idx = 0
    while idx < len(lines):
        count = int(lines[idx])
        idx += 2
        frame: list[tuple[float, float, float, float, float, float]] = []
        for _ in range(count):
            parts = lines[idx].split()
            assert len(parts) == 7, f"Expected extended XYZ atom line with 7 fields, got {len(parts)}"
            frame.append(tuple(float(parts[i]) for i in range(1, 7)))
            idx += 1
        frames.append(frame)
    return frames


def _trajectory_frame_step_numbers() -> list[int]:
    lines = (APP / "trajectory.xyz").read_text().splitlines()
    steps: list[int] = []
    idx = 0
    while idx < len(lines):
        count = int(lines[idx])
        header = lines[idx + 1]
        match = re.fullmatch(r"Step: (\d+)", header)
        assert match is not None, f"Trajectory frame header must be 'Step: <n>', got {header!r}"
        steps.append(int(match.group(1)))
        idx += count + 2
    return steps


def _parse_frame_velocities(frame_index: int) -> list[tuple[float, float, float]]:
    frame = _parse_trajectory_frames_extended()[frame_index]
    return [(p[3], p[4], p[5]) for p in frame]


def _lj_force_mag(r: float) -> float:
    if r < 1e-10:
        return 0.0
    inv_r6 = (1.0 / r) ** 6
    inv_r12 = inv_r6 * inv_r6
    return 24.0 * (2.0 * inv_r12 - inv_r6) / r


def _recompute_vacf_lag1(frames: list[list[tuple[float, float, float, float, float, float]]]) -> float:
    frame_count = len(frames)
    n_particles = len(frames[0])
    numerator = 0.0
    denominator = 0.0
    for frame_idx in range(frame_count - 1):
        for particle in range(n_particles):
            _, _, _, vx0, vy0, vz0 = frames[frame_idx][particle]
            _, _, _, vx1, vy1, vz1 = frames[frame_idx + 1][particle]
            numerator += vx0 * vx1 + vy0 * vy1 + vz0 * vz1
    for frame_idx in range(frame_count):
        for particle in range(n_particles):
            _, _, _, vx, vy, vz = frames[frame_idx][particle]
            denominator += vx * vx + vy * vy + vz * vz
    return numerator / denominator


def _recompute_pressure_virial(positions: list[tuple[float, float, float]], t500: float) -> tuple[float, float, float]:
    virial = 0.0
    rc2 = 2.5 * 2.5
    for i, pos_i in enumerate(positions):
        for j in range(i + 1, len(positions)):
            pos_j = positions[j]
            dx = _minimum_image(pos_j[0] - pos_i[0])
            dy = _minimum_image(pos_j[1] - pos_i[1])
            dz = _minimum_image(pos_j[2] - pos_i[2])
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= rc2:
                continue
            r = math.sqrt(r2)
            force_mag = _lj_force_mag(r)
            fx = -force_mag * dx / r
            fy = -force_mag * dy / r
            fz = -force_mag * dz / r
            virial += dx * fx + dy * fy + dz * fz
    volume = BOX_DIM ** 3
    thermal = N_PARTICLES * t500 / volume
    virial_corr = virial / (3.0 * volume)
    return thermal, virial_corr, thermal + virial_corr


def _parse_vacf_lag1() -> float:
    content = (APP / "vacf_lag1.txt").read_text()
    assert re.search(r"^Lag frames:\s*1\s*$", content, re.MULTILINE)
    return float(re.search(r"^VACF:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1))


def _parse_pressure_virial() -> dict[str, float]:
    content = (APP / "pressure_virial.txt").read_text()
    return {
        "thermal": float(re.search(r"^Thermal pressure:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1)),
        "virial": float(re.search(r"^Virial correction:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1)),
        "total": float(re.search(r"^Total pressure:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1)),
    }


def _recompute_msd_lag1(frames: list[list[tuple[float, float, float]]]) -> float:
    """MSD with cumulative MIC unwrap; lag = 1 saved frame (instruction §3)."""
    n_particles = len(frames[0])
    frame_count = len(frames)
    unwrapped = [[[0.0, 0.0, 0.0] for _ in range(n_particles)] for _ in range(frame_count)]
    for particle in range(n_particles):
        unwrapped[0][particle] = list(frames[0][particle])
    for frame_idx in range(1, frame_count):
        for particle in range(n_particles):
            for axis in range(3):
                delta = frames[frame_idx][particle][axis] - frames[frame_idx - 1][particle][axis]
                unwrapped[frame_idx][particle][axis] = (
                    unwrapped[frame_idx - 1][particle][axis] + _minimum_image(delta)
                )
    acc = 0.0
    pairs = 0
    for frame_idx in range(frame_count - 1):
        for particle in range(n_particles):
            dx = unwrapped[frame_idx + 1][particle][0] - unwrapped[frame_idx][particle][0]
            dy = unwrapped[frame_idx + 1][particle][1] - unwrapped[frame_idx][particle][1]
            dz = unwrapped[frame_idx + 1][particle][2] - unwrapped[frame_idx][particle][2]
            acc += dx * dx + dy * dy + dz * dz
            pairs += 1
    return acc / pairs


def _temperatures_by_step() -> dict[int, float]:
    return {int(row["step"]): _temperature_from_ke(float(row["ke"])) for row in _read_energy_rows()}


def _recompute_block_temperature_stats() -> tuple[float, float]:
    """Block stderr and grand mean over steps 1-500 excluding step 0 (instruction §3)."""
    temps = _temperatures_by_step()
    block_means = []
    for block in range(10):
        start = block * 50 + 1
        end = start + 49
        block_temps = [temps[step] for step in range(start, end + 1)]
        block_means.append(sum(block_temps) / len(block_temps))
    grand_mean = sum(block_means) / len(block_means)
    block_stderr = statistics.stdev(block_means) / math.sqrt(len(block_means))
    return block_stderr, grand_mean


def _parse_temperature_blocks() -> dict[str, float]:
    content = (APP / "temperature_blocks.txt").read_text()
    return {
        "block_stderr": float(re.search(r"^Block mean stderr:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1)),
        "grand_mean": float(re.search(r"^Grand mean:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1)),
    }


def _recompute_heat_capacity_window() -> tuple[float, float, float]:
    """Sample variance of total energy and reduced Cv for steps 250-500 (instruction §3)."""
    rows = _read_energy_rows()
    window = [row for row in rows if 250 <= int(row["step"]) <= 500]
    totals = [float(row["total"]) for row in window]
    temps = [_temperature_from_ke(float(row["ke"])) for row in window]
    energy_var = statistics.variance(totals)
    t_win = sum(temps) / len(temps)
    reduced_cv = energy_var / (N_PARTICLES * t_win * t_win)
    return energy_var, t_win, reduced_cv


def _parse_heat_capacity() -> dict[str, float]:
    content = (APP / "heat_capacity.txt").read_text()
    return {
        "energy_var": float(
            re.search(r"^Energy sample variance:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1)
        ),
        "t_win": float(
            re.search(r"^Window mean temperature:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1)
        ),
        "reduced_cv": float(re.search(r"^Reduced Cv:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1)),
    }


def _parse_msd_lag10() -> float:
    content = (APP / "msd_lag10.txt").read_text()
    assert re.search(r"^Lag frames:\s*1\s*$", content, re.MULTILINE)
    assert re.search(r"^Lag steps:\s*10\s*$", content, re.MULTILINE)
    return float(re.search(r"^MSD:\s*([0-9.eE+-]+)", content, re.MULTILINE).group(1))


def test_source_files_exist() -> None:
    """Required source artifacts must exist under /app and be non-empty."""
    required = ("simulation.cpp", "Makefile", "run_simulation.sh", "particles.dat")
    assert _missing_or_empty(required) == []


def test_particles_lattice_seed(reference: dict) -> None:
    """Particle 0 must sit on the seeded lattice site specified in the instruction."""
    first_line = (APP / "particles.dat").read_text().splitlines()[0].split()
    particle_id, x, y, z = int(first_line[0]), float(first_line[1]), float(first_line[2]), float(first_line[3])
    assert particle_id == 0
    ref = reference["particle0"]
    assert math.isclose(x, ref["x"], abs_tol=1e-5)
    assert math.isclose(y, ref["y"], abs_tol=1e-5)
    assert math.isclose(z, ref["z"], abs_tol=1e-5)


def test_particle0_velocity_matches_reference(reference: dict) -> None:
    """Particle 0 Maxwell-Boltzmann draws must match the seed-42 reference velocities."""
    row = _particle_row(0)
    ref = reference["particle0"]
    for axis in ("vx", "vy", "vz"):
        assert math.isclose(float(row[axis]), ref[axis], abs_tol=IC_VEL_ABS_TOL)


def test_particle42_lattice_position_matches_reference(reference: dict) -> None:
    """Particle 42 must sit on the x-fastest mid-lattice site from instruction §1."""
    row = _particle_row(42)
    ref = reference["particle42"]
    for axis in ("x", "y", "z"):
        assert math.isclose(float(row[axis]), ref[axis], abs_tol=1e-5)


def test_particle42_velocity_matches_reference(reference: dict) -> None:
    """Particle 42 velocities must match the seed-42 reference (mid-lattice IC trap)."""
    row = _particle_row(42)
    ref = reference["particle42"]
    for axis in ("vx", "vy", "vz"):
        assert math.isclose(float(row[axis]), ref[axis], abs_tol=IC_VEL_ABS_TOL)


def test_particle999_lattice_and_velocity_match_reference(reference: dict) -> None:
    """Corner particle 999 must sit on the far lattice site with reference seed-42 velocities."""
    row = _particle_row(999)
    ref = reference["particle999"]
    for axis in ("x", "y", "z", "vx", "vy", "vz"):
        assert math.isclose(float(row[axis]), ref[axis], abs_tol=IC_VEL_ABS_TOL)


def _simulation_binary() -> Path:
    binary = APP / "simulation"
    if not binary.is_file():
        subprocess.run(["make"], cwd=APP, check=True, capture_output=True, text=True)
    assert binary.is_file() and binary.stat().st_size > 0
    return binary


def test_simulation_binary_links_openmp() -> None:
    """The compiled simulator must be linked against libgomp (OpenMP runtime)."""
    binary = _simulation_binary()
    ldd = subprocess.run(
        ["ldd", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "libgomp" in ldd.stdout.lower(), "simulation binary is not linked with OpenMP"


def test_run_script_is_executable_bash() -> None:
    """run_simulation.sh must be a valid, executable bash entrypoint."""
    script = APP / "run_simulation.sh"
    assert script.is_file() and os.access(script, os.X_OK)
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True)


def test_output_files_exist() -> None:
    """All required simulation and analysis outputs must exist under /app."""
    required = (
        "trajectory.xyz",
        "energy.log",
        "timing.log",
        "temperature_stats.txt",
        "rdf.dat",
        "rdf_step250.dat",
        "msd_lag10.txt",
        "vacf_lag1.txt",
        "pressure_virial.txt",
        "temperature_blocks.txt",
        "heat_capacity.txt",
        "performance.txt",
        "scaling.txt",
        "summary.json",
    )
    assert _missing_or_empty(required) == []


def test_trajectory_format() -> None:
    """Trajectory must contain 51 extended XYZ frames with Step: headers and velocities."""
    lines = (APP / "trajectory.xyz").read_text().splitlines()
    expected_frames = (TOTAL_STEPS // TRAJECTORY_INTERVAL) + 1
    assert len(lines) == expected_frames * (N_PARTICLES + 2)
    steps = _trajectory_frame_step_numbers()
    assert steps == list(range(0, TOTAL_STEPS + 1, TRAJECTORY_INTERVAL))
    for frame in _parse_trajectory_frames_extended():
        assert len(frame) == N_PARTICLES
        for particle in frame:
            assert len(particle) == 6


def test_energy_log_row_count() -> None:
    """Energy log must contain one row for each step from 0 through 500."""
    assert len(_read_energy_rows()) == TOTAL_STEPS + 1


def test_energy_log_total_matches_components() -> None:
    """Each energy row must satisfy total = kinetic + potential."""
    rows = _read_energy_rows()
    assert all(
        math.isclose(float(row["total"]), float(row["ke"]) + float(row["pe"]), rel_tol=1e-6, abs_tol=1e-4)
        for row in rows
    )


def test_step0_energy_matches_reference(reference: dict) -> None:
    """Step-0 energies must match independently calibrated reference physics."""
    row = _read_energy_rows()[0]
    ref = reference["step0"]
    assert math.isclose(float(row["ke"]), ref["kinetic"], rel_tol=REF_STEP0_RTOL)
    assert math.isclose(float(row["pe"]), ref["potential"], rel_tol=REF_STEP0_RTOL)
    assert math.isclose(float(row["total"]), ref["total"], rel_tol=REF_STEP0_RTOL)


def test_final_energy_matches_reference(reference: dict) -> None:
    """Step-500 total energy must match the reference integrator trajectory."""
    row = _read_energy_rows()[-1]
    ref_total = reference["step500"]["total"]
    assert math.isclose(float(row["total"]), ref_total, rel_tol=REF_TRAJECTORY_RTOL)


def test_energy_conservation() -> None:
    """Relative total-energy drift from step 0 to 500 must stay below 5%."""
    rows = _read_energy_rows()
    drift = abs(float(rows[-1]["total"]) - float(rows[0]["total"])) / abs(float(rows[0]["total"]))
    assert drift < ENERGY_DRIFT_TOL


def test_step0_temperature_matches_reference(reference: dict) -> None:
    """Step-0 temperature T = 2·KE/(3N) must match the seeded Maxwell-Boltzmann reference."""
    t0 = _temperature_from_ke(float(_energy_row(0)["ke"]))
    assert math.isclose(t0, reference["step0_temperature"], rel_tol=REF_STEP0_RTOL)


def test_rdf_origin_bin_is_zero(reference: dict) -> None:
    """g(r) at r = 0 must be exactly zero under the instruction RDF convention."""
    first = (APP / "rdf.dat").read_text().splitlines()[0].split()
    assert float(first[0]) == 0.0
    assert float(first[1]) == reference["rdf_g0"]


def test_step1_total_energy_matches_reference(reference: dict) -> None:
    """Step-1 total energy must match the reference after the first Velocity-Verlet advance."""
    total = float(_energy_row(1)["total"])
    assert math.isclose(total, reference["step1"]["total"], rel_tol=REF_TRAJECTORY_RTOL)


def test_step100_total_energy_matches_reference(reference: dict) -> None:
    """Step-100 total energy must match the reference integrator trajectory."""
    total = float(_energy_row(100)["total"])
    assert math.isclose(total, reference["step100"]["total"], rel_tol=REF_TRAJECTORY_RTOL)


def test_step250_total_energy_matches_reference(reference: dict) -> None:
    """Step-250 total energy must match the reference mid-trajectory integrator state."""
    total = float(_energy_row(250)["total"])
    assert math.isclose(total, reference["step250"]["total"], rel_tol=REF_TRAJECTORY_RTOL)


def test_step400_total_energy_matches_reference(reference: dict) -> None:
    """Step-400 total energy must match the reference integrator trajectory."""
    total = float(_energy_row(400)["total"])
    assert math.isclose(total, reference["step400"]["total"], rel_tol=REF_TRAJECTORY_RTOL)


def test_initial_com_velocity_near_zero(reference: dict) -> None:
    """Seed-42 Maxwell-Boltzmann draws must leave net center-of-mass velocity near zero."""
    com = _com_velocity(_read_particle_rows())
    for axis in ("vx", "vy", "vz"):
        assert abs(com[axis]) < COM_VEL_ABS_TOL
        assert math.isclose(com[axis], reference["com_velocity"][axis], abs_tol=COM_VEL_REF_ABS_TOL)


def test_initial_minimum_image_pair_distance(reference: dict) -> None:
    """Minimum-image nearest-neighbor spacing on the seed lattice at t = 0 must be 1.0."""
    min_r = _min_pair_distance(_particle_positions(_read_particle_rows()))
    assert math.isclose(min_r, reference["min_pair_distance"], abs_tol=MIN_PAIR_ABS_TOL)


def test_rerun_simulation_reproduces_step0_and_step250_energy() -> None:
    """Recompiling and re-running the submitted binary must reproduce step-0 and step-250 totals."""
    submitted_step0 = float(_read_energy_rows()[0]["total"])
    submitted_step250 = float(_energy_row(250)["total"])
    backups = {
        name: (APP / name).read_bytes()
        for name in ("energy.log", "timing.log", "trajectory.xyz")
        if (APP / name).exists()
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for name in ("simulation.cpp", "Makefile"):
                shutil.copy2(APP / name, work / name)
            subprocess.run(["make"], cwd=work, check=True, capture_output=True, text=True)
            subprocess.run(
                [str(work / "simulation"), "--steps", str(TOTAL_STEPS)],
                cwd=APP,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "OMP_NUM_THREADS": str(OPENMP_THREADS)},
            )
        rows = _read_energy_rows()
        assert math.isclose(float(rows[0]["total"]), submitted_step0, rel_tol=RERUN_STEP0_RTOL)
        assert math.isclose(float(_energy_row(250)["total"]), submitted_step250, rel_tol=REF_TRAJECTORY_RTOL)
    finally:
        for name, data in backups.items():
            (APP / name).write_bytes(data)


def test_temperature_stats_match_energy_log() -> None:
    """All temperature statistics must match values derived from /app/energy.log."""
    rows = _read_energy_rows()
    temps = [_temperature_from_ke(float(row["ke"])) for row in rows]
    stats = _parse_temperature_stats()
    assert math.isclose(stats["Mean"], sum(temps) / len(temps), rel_tol=DERIVED_RTOL)
    assert math.isclose(stats["Std Dev"], statistics.stdev(temps), rel_tol=DERIVED_RTOL)
    assert math.isclose(stats["Min"], min(temps), rel_tol=DERIVED_RTOL)
    assert math.isclose(stats["Max"], max(temps), rel_tol=DERIVED_RTOL)


def test_rdf_matches_trajectory() -> None:
    """rdf.dat must be two-column g(r) from the last trajectory frame with 0.1 bins to r=5.0."""
    expected = _recompute_rdf(_parse_last_trajectory_frame())
    lines = [line.strip() for line in (APP / "rdf.dat").read_text().splitlines() if line.strip()]
    assert len(lines) == int(RDF_MAX_R / RDF_BIN_WIDTH) + 1
    for line in lines:
        parts = line.split()
        assert len(parts) == 2
        r_val, gr_val = map(float, parts)
        if r_val in expected:
            assert math.isclose(gr_val, expected[r_val], rel_tol=RDF_RTOL, abs_tol=RDF_ABS_TOL)


def test_rdf_first_shell_peak_exceeds_unity() -> None:
    """The first coordination shell in g(r) must exceed unity for this dense LJ liquid."""
    lines = [line.split() for line in (APP / "rdf.dat").read_text().splitlines() if line.strip()]
    peak = max(
        float(gr)
        for r_str, gr in ((parts[0], parts[1]) for parts in lines)
        if 0.8 <= float(r_str) <= 2.5
    )
    assert peak > 1.0


def test_performance_formulas() -> None:
    """performance.txt must use the instruction labels and formulas derived from timing.log."""
    timing = _parse_timing_log()
    force_time = timing["force"]
    integ_time = timing["integration"]
    total_time = timing["total"]
    content = (APP / "performance.txt").read_text()
    assert re.search(r"^Total Simulation Time:\s*[0-9.]+\s*s\s*$", content, re.MULTILINE)
    assert re.search(r"^Force Calculation Time:\s*[0-9.]+\s*s\s*$", content, re.MULTILINE)
    assert re.search(r"^Integration Time:\s*[0-9.]+\s*s\s*$", content, re.MULTILINE)
    assert re.search(r"^Particles per second per core:\s*[0-9.]+\s*$", content, re.MULTILINE)
    assert re.search(r"^Parallel Efficiency:\s*[0-9.]+\s*$", content, re.MULTILINE)
    reported_total = float(re.search(r"Total Simulation Time:\s*([0-9.]+)", content).group(1))
    reported_force = float(re.search(r"Force Calculation Time:\s*([0-9.]+)", content).group(1))
    reported_integ = float(re.search(r"Integration Time:\s*([0-9.]+)", content).group(1))
    reported_pps = float(re.search(r"Particles per second per core:\s*([0-9.]+)", content).group(1))
    reported_eff = float(re.search(r"Parallel Efficiency:\s*([0-9.]+)", content).group(1))
    expected_pps = (N_PARTICLES * TOTAL_STEPS) / total_time / OPENMP_THREADS
    expected_eff = min((force_time + integ_time) / total_time, 1.0)
    assert reported_total > 0 and reported_force > 0 and reported_integ > 0
    assert reported_pps > 0 and reported_eff > 0
    assert math.isclose(reported_total, total_time, rel_tol=DERIVED_RTOL)
    assert math.isclose(reported_force, force_time, rel_tol=DERIVED_RTOL)
    assert math.isclose(reported_integ, integ_time, rel_tol=DERIVED_RTOL)
    assert math.isclose(reported_pps, expected_pps, rel_tol=PERFORMANCE_RTOL)
    assert math.isclose(reported_eff, expected_eff, rel_tol=PERFORMANCE_RTOL)


def test_scaling_report() -> None:
    """scaling.txt must use the instruction labels and report speedup as serial / parallel."""
    content = (APP / "scaling.txt").read_text()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    assert len(lines) == 5
    assert lines[0].startswith("Parallel runtime (4 threads):")
    assert lines[1].startswith("Serial runtime (1 thread):")
    assert lines[2].startswith("Parallel force time (4 threads):")
    assert lines[3].startswith("Serial force time (1 thread):")
    assert lines[4].startswith("Speedup:")
    parallel = float(re.search(r"Parallel runtime \(4 threads\):\s*([0-9.]+)", content).group(1))
    serial = float(re.search(r"Serial runtime \(1 thread\):\s*([0-9.]+)", content).group(1))
    speedup = float(re.search(r"Speedup:\s*([0-9.]+)", content).group(1))
    assert parallel > 0 and serial > 0 and speedup > 0
    assert math.isclose(speedup, serial / parallel, rel_tol=PERFORMANCE_RTOL)


def test_summary_json_fields() -> None:
    """summary.json must contain the five instruction fields with consistent values."""
    summary = json.loads((APP / "summary.json").read_text())
    assert set(summary.keys()) == {
        "validation_status",
        "energy_conservation",
        "mean_temperature",
        "parallel_efficiency",
        "final_total_energy",
    }
    rows = _read_energy_rows()
    drift = abs(float(rows[-1]["total"]) - float(rows[0]["total"])) / abs(float(rows[0]["total"]))
    stats = _parse_temperature_stats()
    perf_eff = float(re.search(r"Parallel Efficiency:\s*([0-9.]+)", (APP / "performance.txt").read_text()).group(1))
    assert summary["validation_status"] == ("PASS" if drift < ENERGY_DRIFT_TOL else "FAIL")
    assert math.isclose(summary["energy_conservation"], drift, rel_tol=DERIVED_RTOL)
    assert math.isclose(summary["mean_temperature"], stats["Mean"], rel_tol=DERIVED_RTOL)
    assert math.isclose(summary["parallel_efficiency"], perf_eff, rel_tol=DERIVED_RTOL)
    assert math.isclose(summary["final_total_energy"], float(rows[-1]["total"]), rel_tol=DERIVED_RTOL)


def test_simulation_compiles() -> None:
    """Submitted Makefile and simulation.cpp must compile cleanly in isolation."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy2(APP / "simulation.cpp", work / "simulation.cpp")
        shutil.copy2(APP / "Makefile", work / "Makefile")
        subprocess.run(["make", "clean"], cwd=work, check=False, capture_output=True)
        result = subprocess.run(["make"], cwd=work, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr or result.stdout


def test_simulation_source_has_no_artificial_delays() -> None:
    """simulation.cpp must not inject sleeps or env-based runtime padding."""
    source = (APP / "simulation.cpp").read_text()
    forbidden = (
        r"sleep_for",
        r"std::this_thread",
        r"\busleep\s*\(",
        r"\bnanosleep\s*\(",
        r"SIM_MIN_TOTAL_RUNTIME",
        r"SIM_EXTRA_DELAY",
        r"EXTRA_DELAY",
        r"MIN_TOTAL_RUNTIME",
        r'timing_parallel\.log',
        r'getenv\s*\(\s*["\']SIM_',
    )
    for pattern in forbidden:
        assert re.search(pattern, source) is None, f"Forbidden timing manipulation: {pattern}"


def test_energy_log_tsv_header() -> None:
    """energy.log must begin with the tab-separated header specified in the instruction."""
    header = (APP / "energy.log").read_text().splitlines()[0]
    assert header == "Step\tKinetic\tPotential\tTotal"


def test_particles_dat_count_and_format() -> None:
    """particles.dat must contain 1000 rows in the id x y z vx vy vz format."""
    lines = (APP / "particles.dat").read_text().splitlines()
    assert len(lines) == N_PARTICLES
    for line in lines:
        fields = line.split()
        assert len(fields) == 7
        assert fields[0].isdigit()
        assert all(_is_float(field) for field in fields[1:])


def test_temperature_stats_output_format() -> None:
    """temperature_stats.txt must use the four labeled fields from the instruction."""
    lines = [line.strip() for line in (APP / "temperature_stats.txt").read_text().splitlines() if line.strip()]
    assert len(lines) == 4
    assert lines[0].startswith("Mean Temperature:")
    assert lines[1].startswith("Std Dev Temperature:")
    assert lines[2].startswith("Min Temperature:")
    assert lines[3].startswith("Max Temperature:")
    for line in lines:
        assert float(line.split(":", 1)[1].strip()) > 0


def test_timing_log_reports_force_integration_and_total() -> None:
    """timing.log must use the three labeled lines specified in instruction §2."""
    timing = _parse_timing_log()
    assert timing["force"] > 0 and timing["integration"] > 0 and timing["total"] > 0


def test_timing_log_total_overhead_is_bounded() -> None:
    """timing.log overhead (total-force-integration)/total must be <= 0.18 (instruction §2)."""
    timing = _parse_timing_log()
    work = timing["force"] + timing["integration"]
    assert timing["total"] >= work * (1.0 - DERIVED_RTOL)
    overhead = timing["total"] - work
    assert overhead / timing["total"] <= _TIMING_OVERHEAD_MAX


def test_parallel_force_time_faster_than_serial() -> None:
    """Parallel force time must beat serial by at least 10% (instruction §3 scaling.txt)."""
    _, _, parallel_force, serial_force = _parse_scaling_force_times()
    assert parallel_force > 0 and serial_force > 0
    assert parallel_force * _MIN_FORCE_SPEEDUP <= serial_force


def test_scaling_parallel_force_matches_timing_log() -> None:
    """Reported parallel force time must match the production timing.log measurement."""
    _, _, parallel_force, _ = _parse_scaling_force_times()
    logged_force = _parse_timing_log()["force"]
    assert math.isclose(parallel_force, logged_force, rel_tol=DERIVED_RTOL)


def test_scaling_serial_total_not_disproportionately_padded() -> None:
    """Serial total/force ratio must be <= 1.15x parallel (instruction §3 scaling.txt)."""
    parallel_total, serial_total, parallel_force, serial_force = _parse_scaling_force_times()
    parallel_ratio = parallel_total / parallel_force
    serial_ratio = serial_total / serial_force
    assert serial_ratio <= parallel_ratio * _SERIAL_TOTAL_PADDING_RATIO


def test_scaling_parallel_runtime_matches_parallel_timing() -> None:
    """scaling.txt must record the parallel-run total runtime from timing.log."""
    timing_total = _parse_timing_log()["total"]
    scaling_parallel = float(
        re.search(
            r"Parallel runtime \(4 threads\):\s*([0-9.]+)",
            (APP / "scaling.txt").read_text(),
        ).group(1)
    )
    assert math.isclose(scaling_parallel, timing_total, rel_tol=DERIVED_RTOL)


def test_msd_lag10_matches_unwrapped_trajectory() -> None:
    """MSD must use cumulative MIC unwrap over all trajectory frames at lag 1 frame (10 steps)."""
    expected = _recompute_msd_lag1(_parse_trajectory_frames())
    reported = _parse_msd_lag10()
    assert reported > 0
    assert math.isclose(reported, expected, rel_tol=DERIVED_RTOL)


def test_temperature_block_stderr_excludes_step0() -> None:
    """Block stderr must use sample std of ten block means over steps 1-500 only (exclude step 0)."""
    expected_stderr, expected_grand = _recompute_block_temperature_stats()
    reported = _parse_temperature_blocks()
    content = (APP / "temperature_blocks.txt").read_text()
    assert "Block size: 50" in content
    assert "Block count: 10" in content
    assert math.isclose(reported["block_stderr"], expected_stderr, rel_tol=DERIVED_RTOL)
    assert math.isclose(reported["grand_mean"], expected_grand, rel_tol=DERIVED_RTOL)


def test_heat_capacity_second_half_sample_variance() -> None:
    """Reduced Cv must use sample variance (ddof=1) of total energy for steps 250-500 inclusive."""
    expected_var, expected_t, expected_cv = _recompute_heat_capacity_window()
    reported = _parse_heat_capacity()
    assert (APP / "heat_capacity.txt").read_text().splitlines()[0] == "Window: steps 250-500"
    assert math.isclose(reported["energy_var"], expected_var, rel_tol=DERIVED_RTOL)
    assert math.isclose(reported["t_win"], expected_t, rel_tol=DERIVED_RTOL)
    assert math.isclose(reported["reduced_cv"], expected_cv, rel_tol=DERIVED_RTOL)


def test_block_grand_mean_differs_from_global_temperature_mean() -> None:
    """Grand mean in temperature_blocks.txt must exclude step 0 and differ from temperature_stats mean."""
    global_mean = _parse_temperature_stats()["Mean"]
    block_grand = _parse_temperature_blocks()["grand_mean"]
    steps_1_500_mean = sum(_temperatures_by_step()[s] for s in range(1, TOTAL_STEPS + 1)) / TOTAL_STEPS
    assert math.isclose(block_grand, steps_1_500_mean, rel_tol=DERIVED_RTOL)
    assert not math.isclose(block_grand, global_mean, rel_tol=1e-4)


def test_vacf_lag1_from_trajectory_velocities() -> None:
    """VACF must use velocity columns from consecutive trajectory frames (instruction §3)."""
    expected = _recompute_vacf_lag1(_parse_trajectory_frames_extended())
    reported = _parse_vacf_lag1()
    assert -1.0 <= reported <= 1.0
    assert math.isclose(reported, expected, rel_tol=DERIVED_RTOL)


def test_pressure_virial_last_frame() -> None:
    """Pressure virial must recompute LJ pair forces from the step-500 trajectory frame."""
    t500 = _temperature_from_ke(float(_energy_row(500)["ke"]))
    positions = _parse_last_trajectory_frame()
    thermal, virial_corr, total_p = _recompute_pressure_virial(positions, t500)
    expected = {"thermal": thermal, "virial": virial_corr, "total": total_p}
    reported = _parse_pressure_virial()
    for key in ("thermal", "virial", "total"):
        assert math.isclose(reported[key], expected[key], rel_tol=DERIVED_RTOL)
    assert math.isclose(reported["total"], reported["thermal"] + reported["virial"], rel_tol=DERIVED_RTOL)


def test_rdf_step250_matches_mid_trajectory_frame() -> None:
    """rdf_step250.dat must be g(r) from the step-250 trajectory frame, not the last frame."""
    mid_positions = _parse_trajectory_frames()[250 // TRAJECTORY_INTERVAL]
    expected = _recompute_rdf(mid_positions)
    lines = [line.strip() for line in (APP / "rdf_step250.dat").read_text().splitlines() if line.strip()]
    assert len(lines) == int(RDF_MAX_R / RDF_BIN_WIDTH) + 1
    for line in lines:
        r_val, gr_val = map(float, line.split())
        if r_val in expected:
            assert math.isclose(gr_val, expected[r_val], rel_tol=RDF_RTOL, abs_tol=RDF_ABS_TOL)


def test_step500_velocities_match_reference(reference: dict) -> None:
    """Step-500 velocities for particles 0/42/999 must match the reference integrator."""
    velocities = _parse_frame_velocities(TOTAL_STEPS // TRAJECTORY_INTERVAL)
    for particle_id, ref_key in ((0, "particle0_step500"), (42, "particle42_step500"), (999, "particle999_step500")):
        ref = reference[ref_key]
        vx, vy, vz = velocities[particle_id]
        for axis, val in zip(("vx", "vy", "vz"), (vx, vy, vz)):
            assert math.isclose(val, ref[axis], rel_tol=REF_TRAJECTORY_RTOL)


def test_rdf_step250_differs_from_final_rdf() -> None:
    """Mid-trajectory g(r) must differ from the step-500 RDF (liquid has drifted)."""
    final_lines = (APP / "rdf.dat").read_text().splitlines()
    mid_lines = (APP / "rdf_step250.dat").read_text().splitlines()
    diffs = [
        abs(float(a.split()[1]) - float(b.split()[1]))
        for a, b in zip(final_lines, mid_lines)
        if float(a.split()[0]) >= 0.8
    ]
    assert max(diffs) > 0.01
