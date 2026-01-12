You are inside a Linux container. Your working directory is `/app`.

Build a multi-threaded Lennard-Jones (LJ) molecular dynamics pipeline in C++ with OpenMP, then post-process the outputs with standard Unix tools.

## 1. Simulation Parameters

- Particles: N = 1,000
- Box: cubic, Lx = Ly = Lz = 10.0 (reduced units)
- Cutoff: rc = 2.5σ
- Time step: dt = 0.00001
- Steps: 100 (log energies for steps 0 through 100 inclusive)
- Trajectory: write `/app/trajectory.xyz` every 10 steps (VMD-compatible XYZ format)
- OpenMP: 4 threads (`OMP_NUM_THREADS=4`)
- Initial conditions: generate `/app/particles.dat` with format `id x y z vx vy vz`. Place particles on a reproducible 10×10×10 cubic lattice (spacing 1.0) inside the box and assign Maxwell-Boltzmann velocities at T = 1.0 using random seed 42.

## 2. C++ Simulation

Implement a periodic-boundary NVE simulation in `/app/simulation.cpp` with:

- LJ potential: U(r) = 4ε[(σ/r)¹² - (σ/r)⁶] (ε = σ = 1 in reduced units)
- Forces from F = -∇U
- Velocity Verlet integration
- OpenMP parallelization on force calculation, integration, and energy reduction loops

Also create:

- `/app/Makefile` (must compile with `-fopenmp`)
- `/app/run_simulation.sh` (sets `OMP_NUM_THREADS=4`, builds and runs the simulation)

### Simulation outputs

- `/app/energy.log`: tab-separated header `Step\tKinetic\tPotential\tTotal`, then one row per step with computed energies
- `/app/timing.log`: wall-clock seconds for force calculation, integration, and total runtime (measured, not hardcoded)

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

Compute g(r) from the last frame of `/app/trajectory.xyz` using minimum-image convention and bins of width 0.1 from r = 0.0 to 5.0. Two columns: `r g(r)`.

### `/app/performance.txt`

Derive from `/app/timing.log`:

```
Total Simulation Time: <total> s
Force Calculation Time: <force> s
Integration Time: <integration> s
Particles per second per core: <(N * steps) / total / 4>
Parallel Efficiency: <(force_time + integration_time) / total_time, capped at 1.0>
```

### `/app/summary.json`

Valid JSON with:

```json
{
  "validation_status": "PASS",
  "energy_conservation": <relative drift of total energy>,
  "mean_temperature": <mean T from analysis>,
  "parallel_efficiency": <from performance.txt>,
  "final_total_energy": <total energy at step 100>
}
```

Set `"validation_status": "PASS"` only if relative total-energy drift from step 0 to 100 is below 0.10.

## 4. Constraints

- No external libraries beyond the C++ standard library and OpenMP
- Use seed 42 for reproducible initial conditions
- All analysis values must be computed from simulation outputs (do not hardcode metrics)

You have 900 seconds to complete this task.
