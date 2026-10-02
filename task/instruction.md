You are inside a Linux container. Your working directory is `/app`.

Build a Lennard-Jones NVE pipeline in C++ with OpenMP and Unix post-processing. Synthetic seed-42 ICs.

## 1. Simulation Parameters

- Particles: N = 1,000
- Box: cubic, Lx = Ly = Lz = 10.0 (reduced units)
- Cutoff: rc = 2.5σ
- Time step: dt = 0.0005
- Steps: 500 (log energies for steps 0 through 500 inclusive)
- Trajectory: write `/app/trajectory.xyz` every 10 steps (VMD-compatible XYZ format)
- OpenMP: 4 threads (`OMP_NUM_THREADS=4`) for integration and energy-reduction loops; the pairwise force loop must be race-free
- Initial conditions: `/app/particles.dat` (`id x y z vx vy vz`). 10×10×10 lattice (spacing 1.0); x-fastest ids `ix=id%10`, `iy=(id//10)%10`, `iz=id//100`; particle 0 at (0.5,0.5,0.5). `random.seed(42)` Box-Muller per particle: `u1=max(random(),1e-12)`, `radius=sqrt(-2 ln u1)`, `angle=2π·random()`; first pair→vx,vy; second **cos**→vz; no rescale-to-exact-T, no COM subtraction.

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

**Timing partition:** Force = all force calls; Integration = energy reduction plus `energy.log` / `trajectory.xyz` I/O plus VV updates (exclude force); Total = wall clock from first force through step 500. Require `total ≥ force + integration` and `(total − force − integration)/total ≤ 0.18`.

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

1. For i<j pairs, minimum-image Δx,Δy,Δz with L=10; r=√(Δx²+Δy²+Δz²).
2. If r ≤ 5.0, bin b = ⌊r/Δr⌋ and add 2 to count[b] (count both i→j and j→i).
3. ρ = N/L³; shell volume V_shell = 4π r²·Δr at left edge r = b·Δr.
4. g(r) = count[b] / (N·ρ·V_shell); g(0) = 0.

### `/app/msd_lag10.txt`

From **all** frames in `/app/trajectory.xyz` (51 frames at steps 0, 10, …, 500). Positions only — do not read velocities from elsewhere.

**Cumulative MIC unwrap** (L = 10.0): for each particle `p`, frame `f = 0` uses wrapped `(x, y, z)` from the XYZ frame. For `f ≥ 1`, incremental displacement from consecutive wrapped coordinates uses `Δ = coord(f) − coord(f−1)` then `Δ_mic = Δ − L·round(Δ/L)` (same convention as RDF §3), and unwrapped coordinates accumulate: `coord_u(f) = coord_u(f−1) + Δ_mic`.

**MSD** for lag **1 saved frame** (= 10 simulation steps):

```
MSD = (1 / (N · (F − 1))) · Σ_p Σ_{f=0}^{F−2} |r_u(p,f+1) − r_u(p,f)|²
```

with N = 1000, F = 51. Write exactly:

```
Lag frames: 1
Lag steps: 10
MSD: <value with at least 9 digits after the decimal>
```

### `/app/temperature_blocks.txt`

From `/app/energy.log` only. **Exclude step 0** from this entire analysis. Partition steps **1 through 500** into **10** consecutive blocks of **50** steps: block `b` (0-indexed) covers steps `50·b + 1` … `50·b + 50`. For each block, `T_b = mean of T(step)` with `T = 2·KE/(3·N)`.

Let `μ_B = (1/10) Σ_b T_b` (mean of the ten block means). Sample standard deviation of block means: `s_B = sqrt( (1/9) Σ_b (T_b − μ_B)² )`. **Block standard error:** `SE = s_B / sqrt(10)`.

Write exactly:

```
Block size: 50
Block count: 10
Block mean stderr: <SE>
Grand mean: <μ_B>
```

(`Grand mean` is the mean of block means over steps 1–500; it is **not** the same as `Mean Temperature` in `temperature_stats.txt`, which includes step 0.)

### `/app/heat_capacity.txt`

From `/app/energy.log` **total-energy column** for steps **250 through 500 inclusive** (251 samples). Let `μ_E` be the sample mean and `Var_E` the **sample variance with ddof = 1** (divide by 250, not 251). Let `T_win` be the mean of `T = 2·KE/(3·N)` over the same step window. **Reduced heat capacity estimate:** `Cv = Var_E / (N · T_win²)`.

Write exactly:

```
Window: steps 250-500
Energy sample variance: <Var_E>
Window mean temperature: <T_win>
Reduced Cv: <Cv>
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
Parallel force time (4 threads): <from parallel timing.log> s
Serial force time (1 thread): <from serial timing run> s
Speedup: <serial_runtime / parallel_runtime>
```

Parallel force: `serial_force ≥ 1.10 × parallel_force`. Total/force overhead: `(serial_total/serial_force) ≤ 1.15 × (parallel_total/parallel_force)`.

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
- Verification gates (not visible): seed-42 IC particles 0/42/999; mid-trajectory energy rtol 0.2%; NVE drift <5%; RDF §3; timing §2; scaling §3; no sleep padding in `simulation.cpp`
