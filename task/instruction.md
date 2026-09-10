You are inside a Linux container. Your working directory is `/app`.

Build a multi-threaded Lennard-Jones NVE pipeline in C++ with OpenMP and post-process with Unix tools. Synthetic ICs: seeded lattice plus Maxwell-Boltzmann velocities.

## 1. Simulation Parameters

- Particles: N = 1,000
- Box: cubic, Lx = Ly = Lz = 10.0 (reduced units)
- Cutoff: rc = 2.5σ
- Time step: dt = 0.0005
- Steps: 500 (log energies for steps 0 through 500 inclusive)
- Trajectory: write `/app/trajectory.xyz` every 10 steps (VMD-compatible XYZ format)
- OpenMP: 4 threads (`OMP_NUM_THREADS=4`) for integration and energy-reduction loops; the pairwise force loop must be race-free
- Initial conditions: `/app/particles.dat` as `id x y z vx vy vz`. 10×10×10 lattice (spacing 1.0), particle 0 at (0.5, 0.5, 0.5). Index ids with **x fastest**: `ix=id%10`, `iy=(id//10)%10`, `iz=id//100`. T = 1.0 means unit-variance Maxwell-Boltzmann draws: `random.seed(42)`; per particle draw `u1=max(random(),1e-12)`, `u2=random()`, `radius=sqrt(-2*ln(u1))`, `angle=2π*u2`; first Box-Muller pair → vx,vy; second pair's **cos** → vz. No rescale-to-exact-T and no COM-velocity subtraction.

## 2. C++ Simulation

Implement a periodic-boundary NVE simulation in `/app/simulation.cpp` with:

- LJ potential: U(r) = 4ε[(σ/r)¹² - (σ/r)⁶] (ε = σ = 1 in reduced units)
- Forces from F = -∇U
- Velocity Verlet NVE integration
- OpenMP on force, integration, and energy-reduction loops (race-free force accumulation required)

Also create:

- `/app/Makefile` (`-fopenmp`; `clean` removes only `simulation` and `*.o`)
- `/app/run_simulation.sh` (sets `OMP_NUM_THREADS=4`, builds and runs `/app/simulation --steps 500`)

### Simulation outputs

- `/app/energy.log`: header `Step\tKinetic\tPotential\tTotal`, then one row per step.
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

From the **last** XYZ frame in `/app/trajectory.xyz` (step-500 configuration). Write 51 rows, two columns `r g(r)`, with Δr = 0.1 and r at bin **left edges** 0.0, 0.1, …, 5.0.

**Normative g(r) procedure:**

1. For each pair with i < j, compute separations Δx, Δy, Δz; apply minimum-image per component: Δ' = Δ − L·round(Δ/L) with L = 10.0; then r = √(Δx'² + Δy'² + Δz'²).
2. If r ≤ 5.0, bin b = ⌊r/Δr⌋ and add 2 to count[b] (count both i→j and j→i).
3. ρ = N/L³; shell volume V_shell = 4π r²·Δr at left edge r = b·Δr.
4. g(r) = count[b] / (N·ρ·V_shell); g(0) = 0.

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
Parallel force time (4 threads): <from parallel timing.log> s
Serial force time (1 thread): <from serial timing run> s
Speedup: <serial_runtime / parallel_runtime>
```

Parallel force time must beat serial by at least 10% (`serial_force ≥ 1.10 × parallel_force`).

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
- Verification gates (reference under `/tests/reference/`, not visible): seed-42 IC on particles 0/42/999 (x-fastest lattice, u1 Box-Muller radius); mid-trajectory total-energy rtol 0.2%; NVE drift <5%; RDF per §3; parallel force ≥10% below serial; no timing padding in `simulation.cpp`
