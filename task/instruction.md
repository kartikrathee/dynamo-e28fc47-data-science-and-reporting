You are inside a Linux container. Your working directory is `/app`.

Build a multi-threaded Lennard-Jones (LJ) molecular dynamics pipeline in C++ with OpenMP, then post-process the outputs with standard Unix tools. The initial configuration is synthetic (a seeded lattice with Maxwell-Boltzmann velocities); there is no external dataset file.

## 1. Simulation Parameters

- Particles: N = 1,000
- Box: cubic, Lx = Ly = Lz = 10.0 (reduced units)
- Cutoff: rc = 2.5σ
- Time step: dt = 0.0005
- Steps: 500 (log energies for steps 0 through 500 inclusive)
- Trajectory: write `/app/trajectory.xyz` every 10 steps (VMD-compatible XYZ format)
- OpenMP: 4 threads (`OMP_NUM_THREADS=4`) for integration and energy-reduction loops; the pairwise force loop must be race-free (incorrect parallel force updates fail verification)
- Initial conditions: generate `/app/particles.dat` with format `id x y z vx vy vz`. Place particles on a 10×10×10 cubic lattice (spacing 1.0) so particle 0 sits at (0.5, 0.5, 0.5). Assign Maxwell-Boltzmann velocities at T = 1.0 in reduced units (kB = m = 1): each velocity component is an independent standard normal N(0, 1) with variance kT/m = 1. Use `random.seed(42)` and Box-Muller with the standard library `random` module only; for each particle call Box-Muller twice. Each call returns `(z_cos, z_sin) = (r·cos θ, r·sin θ)` in that order; assign `vx` and `vy` from the first call's cos and sin, `vz` from the second call's cos, and discard the second sin (the fourth draw).

## 2. C++ Simulation

Implement a periodic-boundary NVE simulation in `/app/simulation.cpp` with:

- LJ potential: U(r) = 4ε[(σ/r)¹² - (σ/r)⁶] (ε = σ = 1 in reduced units)
- Forces from F = -∇U
- Velocity Verlet integration
- OpenMP parallelization on force calculation, integration, and energy reduction loops

Also create:

- `/app/Makefile` (must compile with `-fopenmp`)
- `/app/run_simulation.sh` (sets `OMP_NUM_THREADS=4`, builds and runs `/app/simulation --steps 500`)

### Simulation outputs

- `/app/energy.log`: tab-separated header `Step\tKinetic\tPotential\tTotal`, then one row per step with computed energies
- `/app/timing.log`: measured wall-clock seconds (do not hardcode), one field per line:

```
Force calculation time: <seconds> s
Integration time: <seconds> s
Total runtime: <seconds> s
```

## 3. Analysis Pipeline

Derive all analysis files from the simulation outputs:

### `/app/temperature_stats.txt`

Compute temperature per step from kinetic energy using reduced units: T = 2·KE / (3·N). Write:

```
Mean Temperature: <mean>
Std Dev Temperature: <stddev>
Min Temperature: <min>
Max Temperature: <max>
```

### `/app/rdf.dat`

Compute g(r) from the last frame of `/app/trajectory.xyz` using minimum-image convention. Use bins of width Δr = 0.1 from r = 0.0 to 5.0 inclusive (51 rows). Two columns: `r g(r)`.

**Binning and normalization (normative):**

1. For each unordered pair (i, j), compute minimum-image distance r; if r ≤ 5.0, increment bin `b = floor(r / Δr)` by 2 (count both i→j and j→i).
2. Report **r as the bin left edge**: r = b · Δr (so rows are r = 0.0, 0.1, …, 5.0 — not bin centers).
3. Number density ρ = N / L³ with N = 1000 and L = 10.0.
4. Shell volume at the left edge: V_shell = 4π r² · Δr (thin-shell approximation evaluated at the left edge, not the exact spherical shell 4π/3 · [(r+Δr)³ − r³]).
5. Normalize each bin: g(r) = count[b] / (N · ρ · V_shell); set g(0) = 0 when r = 0.

Example first rows (illustrative format only):

```
0.0 0.000000
0.1 <g(r)>
1.0 <g(r)>
```

### `/app/performance.txt`

Derive from the three `/app/timing.log` lines defined in §2 (`Force calculation time`, `Integration time`, `Total runtime`). Write:

```
Total Simulation Time: <total> s
Force Calculation Time: <force> s
Integration Time: <integration> s
Particles per second per core: <(N * steps) / total / 4>
Parallel Efficiency: <min((force_time + integration_time) / total_time, 1.0)>
```

All five numeric fields must be strictly positive after a real run.

### `/app/scaling.txt`

After the parallel run, execute one additional serial timing run with `OMP_NUM_THREADS=1` (same `particles.dat`, 500 steps). Do not overwrite the parallel simulation outputs. Write:

```
Parallel runtime (4 threads): <total runtime from the parallel run> s
Serial runtime (1 thread): <total runtime from the serial run> s
Speedup: <serial_runtime / parallel_runtime>
```

### `/app/summary.json`

Valid JSON with:

```json
{
  "validation_status": "PASS",
  "energy_conservation": <relative drift of total energy>,
  "mean_temperature": <mean T from temperature_stats.txt>,
  "parallel_efficiency": <from performance.txt>,
  "final_total_energy": <total energy at step 500>
}
```

Set `"validation_status": "PASS"` only if relative total-energy drift from step 0 to 500 is below 0.05.

## 4. Constraints

- No external libraries beyond the C++ standard library, OpenMP, and Python standard library for initial-condition generation
- Use seed 42 for reproducible initial conditions
- All analysis values must be computed from simulation outputs (do not hardcode metrics)
- The compiled `/app/simulation` binary must be executed to produce `/app/energy.log` and `/app/trajectory.xyz`
- Verification compares step-0/step-500 energies and RDF peaks against oracle-calibrated reference values for this seeded initial condition (stored under `/tests/reference/` at verify time, not visible during the agent run). Reference numbers were generated by running the oracle on the pinned environment image in `environment/Dockerfile`. Gates: step-0 energy rtol 0.1%; step-500 total energy rtol 0.2%; RDF peak values ±5% per bin; re-executing the submitted binary on `/app/particles.dat` must reproduce step-0 total energy within 0.01%
