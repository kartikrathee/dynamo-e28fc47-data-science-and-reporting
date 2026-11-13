You are inside a Linux container. Working directory: `/app`.

Build a Lennard-Jones NVE pipeline in C++ with OpenMP and post-processing. Seed-42 ICs.

## 1. Simulation

- N = 1000; cubic box L = 10.0; rc = 2.5; dt = 0.0005; steps 0–500 inclusive.
- Trajectory `/app/trajectory.xyz` every 10 steps: N, `Step: <step>`, N lines `Ar x y z vx vy vz` (six decimals; extended XYZ).
- OpenMP: `OMP_NUM_THREADS=4` on force, integration, and energy-reduction loops; pairwise force loop must be race-free.
- IC `/app/particles.dat` (`id x y z vx vy vz`): 10×10×10 lattice spacing 1.0; x-fastest indexing; id 0 at (0.5,0.5,0.5). `random.seed(42)` Box-Muller: u1=max(random(),1e-12), r=√(-2 ln u1), θ=2π·random(); vx,vy=r cos/sin θ; second draw cos→vz; no T rescale/COM fix.

Implement NVE in `/app/simulation.cpp`: LJ U=4ε[(σ/r)¹²−(σ/r)⁶], VV, MIC PBC. `/app/Makefile` (-fopenmp; `clean`→simulation,*.o); `/app/run_simulation.sh` runs `simulation --steps 500` at 4 threads.

Outputs:
- `/app/energy.log`: header `Step\tKinetic\tPotential\tTotal`, one row per step.
- `/app/timing.log` (measured, not hardcoded):
  `Force calculation time: <s> s` / `Integration time: <s> s` / `Total runtime: <s> s`
  Force=all force calls; Integration=energy reduction+energy.log/trajectory I/O+VV kicks; Total=wall clock through step 500. Require total≥force+integration; overhead (total−force−integration)/total≤0.18.

## 2. Analysis (derive from simulation outputs)

**g(r)** left-edge Δr=0.1, r=0…5.0, 51 rows: MIC i<j; bin ⌊r/Δr⌋; count both ends; ρ=N/L³; V_shell=4πr²Δr; g=count/(NρV_shell); g(0)=0.

- `/app/temperature_stats.txt`: T=2·KE/(3N); sample std (ddof=1). Labels `Mean Temperature`, `Std Dev Temperature`, `Min Temperature`, `Max Temperature`.
- `/app/rdf.dat`: g(r) from **last** trajectory frame (step 500).
- `/app/rdf_step250.dat`: same procedure from **step 250** frame.
- `/app/msd_lag10.txt`: 51 frames; cumulative MIC unwrap (f=0 wrapped; f≥1 add Δ−L·round(Δ/L)). MSD lag 1 frame: mean |r_u(f+1)−r_u(f)|² over particles and frames. Labels `Lag frames: 1`, `Lag steps: 10`, `MSD: <≥9 decimals>`.
- `/app/temperature_blocks.txt`: from `energy.log`, exclude step 0; ten 50-step blocks (1–500); SE=sample_std(block means,ddof=1)/√10. Labels `Block size: 50`, `Block count: 10`, `Block mean stderr`, `Grand mean` (≠ temperature_stats mean).
- `/app/heat_capacity.txt`: total energy steps 250–500; Var_E (ddof=1, ÷250); T_win=mean T; Cv=Var_E/(N·T_win²). Labels: `Window: steps 250-500`, `Energy sample variance`, `Window mean temperature`, `Reduced Cv`.
- `/app/vacf_lag1.txt`: trajectory velocity columns (no finite-difference). Lag 1 frame with F=51 trajectory frames: **numerator** = Σ(v·v_next) over all N particles and the 50 consecutive frame pairs (f=0 with f+1=1, …, f=49 with f+1=50); **denominator** = Σ(v²) over all N particles and **every** trajectory frame f=0…50 (51 frames total)—computed in a separate loop, not restricted to the 50 paired frames used in the numerator; VACF = numerator/denominator. Write `Lag frames: 1`, `VACF: <≥9 decimals>`.
- `/app/pressure_virial.txt`: step-500 positions; LJ pairs i<j MIC (ε=σ=rc=2.5). Force on **i**: `f_mag=24(2/r¹²−1/r⁶)/r`, `fx=−f_mag·dx/r` (dx MIC j−i); vir=Σ(dx·fx+dy·fy+dz·fz); T500 from step-500 KE. Lines: `Thermal pressure`, `Virial correction`, `Total pressure` (six decimals).
- `/app/performance.txt` from timing.log: `Total Simulation Time`, `Force Calculation Time`, `Integration Time` (append ` s`), `Particles per second per core` = N·steps/total/4, `Parallel Efficiency` = min((force+integration)/total, 1.0). All five values > 0.
- `/app/scaling.txt`: after parallel run, one serial run (`OMP_NUM_THREADS=1`, same IC, 500 steps) without overwriting parallel outputs. Exact five lines: `Parallel runtime (4 threads): <s> s`, `Serial runtime (1 thread): <s> s`, `Parallel force time (4 threads): <s> s`, `Serial force time (1 thread): <s> s`, `Speedup: <serial/parallel>`. Require serial_force ≥ 1.10×parallel_force and `(serial_total/serial_force) ≤ 1.15×(parallel_total/parallel_force)`.
- `/app/summary.json`: `validation_status`, `energy_conservation`, `mean_temperature`, `parallel_efficiency`, `final_total_energy`; PASS only if drift<0.05.

## 3. Constraints

- C++ stdlib + OpenMP + Python stdlib for ICs only; seed 42; no hardcoded metrics.
- Run `/app/simulation` to produce `/app/energy.log` and `/app/trajectory.xyz`.
