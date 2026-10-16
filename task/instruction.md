You are inside a Linux container. Working directory: `/app`.

Build a Lennard-Jones NVE pipeline in C++ with OpenMP and post-processing. Seed-42 ICs.

## 1. Simulation

- N = 1000; cubic box L = 10.0; rc = 2.5; dt = 0.0005; steps 0–500 inclusive.
- Trajectory `/app/trajectory.xyz` every 10 steps: line 1 = N, line 2 = `Step: <step>`, then N lines `Ar x y z vx vy vz` (six decimals each; extended XYZ with integrator velocities).
- OpenMP: `OMP_NUM_THREADS=4` on force, integration, and energy-reduction loops; pairwise force loop must be race-free.
- IC `/app/particles.dat` (`id x y z vx vy vz`): 10×10×10 lattice spacing 1.0; x-fastest `ix=id%10`, `iy=(id//10)%10`, `iz=id//100`; id 0 at (0.5,0.5,0.5). `random.seed(42)` Box-Muller: `u1=max(random(),1e-12)`, `radius=sqrt(-2 ln u1)`, `angle=2π·random()`; first pair→vx,vy; second **cos**→vz; no T rescale, no COM subtraction.

Implement periodic NVE in `/app/simulation.cpp`: LJ U(r)=4ε[(σ/r)¹²−(σ/r)⁶] (ε=σ=1), F=−∇U, velocity Verlet, MIC PBC. Also `/app/Makefile` (`-fopenmp`; `clean` removes only `simulation` and `*.o`) and `/app/run_simulation.sh` (builds and runs `/app/simulation --steps 500` at 4 threads).

Outputs:
- `/app/energy.log`: header `Step\tKinetic\tPotential\tTotal`, one row per step.
- `/app/timing.log` (measured, not hardcoded):
  `Force calculation time: <s> s` / `Integration time: <s> s` / `Total runtime: <s> s`
  Force = all force calls; Integration = energy reduction + `energy.log`/`trajectory.xyz` I/O + VV kicks (exclude force); Total = wall clock through step 500. Require `total ≥ force + integration` and overhead `(total−force−integration)/total ≤ 0.18`.

## 2. Analysis (derive from simulation outputs)

**Shared g(r)** (left-edge bins Δr=0.1, r=0.0…5.0, 51 rows `r g(r)`): MIC pairs i<j, L=10; bin b=⌊r/Δr⌋ if r≤5; count both ends; ρ=N/L³; V_shell=4πr²Δr at left edge; g=count/(N·ρ·V_shell); g(0)=0.

- `/app/temperature_stats.txt`: T=2·KE/(3N) per step; **sample** std (ddof=1). Labels: `Mean Temperature`, `Std Dev Temperature`, `Min Temperature`, `Max Temperature`.
- `/app/rdf.dat`: g(r) from **last** trajectory frame (step 500).
- `/app/rdf_step250.dat`: same procedure from **step 250** frame.
- `/app/msd_lag10.txt`: positions only from all 51 frames. Cumulative MIC unwrap (L=10): f=0 uses wrapped coords; f≥1 adds `Δ−L·round(Δ/L)` per axis. MSD lag 1 frame (=10 steps): `(1/(N(F−1))) Σ_p Σ_f |r_u(f+1)−r_u(f)|²`, N=1000, F=51. Write `Lag frames: 1`, `Lag steps: 10`, `MSD: <≥9 decimals>`.
- `/app/temperature_blocks.txt`: from `energy.log`, **exclude step 0**. Ten blocks of 50 steps (1–500); block mean T_b; μ_B=mean(T_b); s_B=sample std of block means (ddof=1); SE=s_B/√10. Write `Block size: 50`, `Block count: 10`, `Block mean stderr: <SE>`, `Grand mean: <μ_B>` (steps 1–500 only, ≠ global mean in temperature_stats).
- `/app/heat_capacity.txt`: total energy steps 250–500; Var_E (ddof=1, ÷250); T_win=mean T; Cv=Var_E/(N·T_win²). Labels: `Window: steps 250-500`, `Energy sample variance`, `Window mean temperature`, `Reduced Cv`.
- `/app/vacf_lag1.txt`: velocity columns in trajectory (no finite-difference). Lag 1 frame: VACF = Σ(v·v_next)/Σ(v²) over particles and frames. Write `Lag frames: 1`, `VACF: <≥9 decimals>`.
- `/app/pressure_virial.txt`: step-500 positions; LJ pairs (ε=σ=rc=2.5, `f_mag=24(2/r¹²−1/r⁶)/r`, MIC i→j). vir=Σ(dx·fx+dy·fy+dz·fz) for r<rc; T500 from step-500 KE. Lines: `Thermal pressure`, `Virial correction`, `Total pressure` (six decimals).
- `/app/performance.txt` from timing.log: `Total Simulation Time`, `Force Calculation Time`, `Integration Time` (append ` s`), `Particles per second per core` = N·steps/total/4, `Parallel Efficiency` = min((force+integration)/total, 1.0). All five values > 0.
- `/app/scaling.txt`: after parallel run, one serial run (`OMP_NUM_THREADS=1`, same IC, 500 steps) without overwriting parallel outputs. Lines: parallel/serial total runtime, parallel/serial force time, `Speedup` = serial/parallel. Require serial_force ≥ 1.10×parallel_force and `(serial_total/serial_force) ≤ 1.15×(parallel_total/parallel_force)`.
- `/app/summary.json`: `validation_status`, `energy_conservation` (relative drift step 0→500), `mean_temperature`, `parallel_efficiency`, `final_total_energy`. Set `"validation_status":"PASS"` only if drift < 0.05.

## 3. Constraints

- C++ stdlib + OpenMP + Python stdlib for ICs only; seed 42; no hardcoded metrics.
- Run `/app/simulation` to produce `/app/energy.log` and `/app/trajectory.xyz`.
