#!/usr/bin/env python3
"""Generate seeded lattice positions with Maxwell-Boltzmann velocities at T=1."""

import math
import random

random.seed(42)
spacing = 1.0


def box_muller() -> tuple[float, float]:
    u1 = max(random.random(), 1e-12)
    u2 = random.random()
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = 2.0 * math.pi * u2
    return radius * math.cos(angle), radius * math.sin(angle)  # (z_cos, z_sin) per instruction


for particle_id in range(1000):
    ix = particle_id % 10
    iy = (particle_id // 10) % 10
    iz = particle_id // 100
    x = (ix + 0.5) * spacing
    y = (iy + 0.5) * spacing
    z = (iz + 0.5) * spacing
    z1, z2 = box_muller()
    z3, _ = box_muller()
    vx, vy, vz = z1, z2, z3
    print(f"{particle_id} {x:.6f} {y:.6f} {z:.6f} {vx:.6f} {vy:.6f} {vz:.6f}")
