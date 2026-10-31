# Benchmark Result

**Timestamp:** 2026-10-31T18:59:59.183383+00:00  
**Laws:** 23  
**Noise:** 2%  
**Wall time:** 790.2s

## Summary

| Method | Conv. | Skipped | Median iter | Median pts | Median time |
|---|---|---|---|---|---|
| single_shot | 16/23 (70%) | 0 | 0.0 | 15.0 | 13.5s |
| random_growth | 21/23 (91%) | 0 | 0.0 | 5.0 | 8.8s |
| closed_loop_boed | 8/8 (100%) | 15 | 0.0 | 5.0 | 4.5s |

## Per-law

| Law | Method | Conv. | Iter | Pts | Time |
|---|---|---|---|---|---|
| stefan_boltzmann_T4 | single_shot | ✓ | 0 | 15 | 5.1s |
| stefan_boltzmann_T4 | random_growth | ✓ | 0 | 5 | 2.5s |
| stefan_boltzmann_T4 | closed_loop_boed | ✓ | 0 | 5 | 2.2s |
| kepler_third | single_shot | ✓ | 0 | 15 | 8.5s |
| kepler_third | random_growth | ✓ | 0 | 5 | 3.3s |
| kepler_third | closed_loop_boed | ✓ | 0 | 5 | 1.0s |
| inverse_square | single_shot | ✓ | 0 | 15 | 10.4s |
| inverse_square | random_growth | ✓ | 0 | 5 | 3.2s |
| inverse_square | closed_loop_boed | ✓ | 0 | 5 | 5.4s |
| linear | single_shot | ✓ | 0 | 15 | 1.4s |
| linear | random_growth | ✓ | 0 | 5 | 1.5s |
| linear | closed_loop_boed | ✓ | 0 | 5 | 1.1s |
| ohm | single_shot | ✓ | 0 | 15 | 7.7s |
| ohm | random_growth | ✓ | 0 | 5 | 2.7s |
| ohm | closed_loop_boed | skip | 0 | 0 | 0.0s |
| newton_2 | single_shot | ✓ | 0 | 15 | 8.8s |
| newton_2 | random_growth | ✓ | 0 | 5 | 4.8s |
| newton_2 | closed_loop_boed | skip | 0 | 0 | 0.0s |
| kinetic_energy | single_shot | ✓ | 0 | 15 | 7.8s |
| kinetic_energy | random_growth | ✓ | 0 | 5 | 3.1s |
| kinetic_energy | closed_loop_boed | skip | 0 | 0 | 0.0s |
| arrhenius | single_shot | ✓ | 0 | 15 | 5.3s |
| arrhenius | random_growth | ✓ | 0 | 5 | 5.4s |
| arrhenius | closed_loop_boed | ✓ | 0 | 5 | 5.2s |
| offset_inverse | single_shot | ✓ | 0 | 15 | 5.4s |
| offset_inverse | random_growth | ✓ | 0 | 5 | 4.1s |
| offset_inverse | closed_loop_boed | ✓ | 0 | 5 | 9.8s |
| gravitational | single_shot | ✗ | 0 | 15 | 21.2s |
| gravitational | random_growth | ✓ | 0 | 5 | 9.2s |
| gravitational | closed_loop_boed | skip | 0 | 0 | 0.0s |
| coulomb | single_shot | ✗ | 0 | 15 | 21.5s |
| coulomb | random_growth | ✓ | 0 | 5 | 8.8s |
| coulomb | closed_loop_boed | skip | 0 | 0 | 0.0s |
| yukawa_potential | single_shot | ✓ | 0 | 15 | 13.5s |
| yukawa_potential | random_growth | ✓ | 0 | 5 | 9.2s |
| yukawa_potential | closed_loop_boed | skip | 0 | 0 | 0.0s |
| newton_1pn | single_shot | ✗ | 0 | 15 | 6.0s |
| newton_1pn | random_growth | ✓ | 1 | 6 | 8.4s |
| newton_1pn | closed_loop_boed | ✓ | 0 | 5 | 3.9s |
| newton_short_distance | single_shot | ✓ | 0 | 15 | 6.2s |
| newton_short_distance | random_growth | ✓ | 0 | 5 | 5.6s |
| newton_short_distance | closed_loop_boed | ✓ | 0 | 5 | 9.0s |
| cow_phase | single_shot | ✗ | 0 | 15 | 16.6s |
| cow_phase | random_growth | ✗ | 10 | 15 | 128.8s |
| cow_phase | closed_loop_boed | skip | 0 | 0 | 0.0s |
| sagnac_phase | single_shot | ✓ | 0 | 15 | 19.3s |
| sagnac_phase | random_growth | ✓ | 0 | 5 | 9.7s |
| sagnac_phase | closed_loop_boed | skip | 0 | 0 | 0.0s |
| magnetic_ab | single_shot | ✓ | 0 | 15 | 19.2s |
| magnetic_ab | random_growth | ✓ | 0 | 5 | 9.8s |
| magnetic_ab | closed_loop_boed | skip | 0 | 0 | 0.0s |
| electric_ab | single_shot | ✓ | 0 | 15 | 19.0s |
| electric_ab | random_growth | ✓ | 0 | 5 | 9.5s |
| electric_ab | closed_loop_boed | skip | 0 | 0 | 0.0s |
| grav_ab_photon | single_shot | ✗ | 0 | 15 | 24.2s |
| grav_ab_photon | random_growth | ✓ | 1 | 6 | 33.6s |
| grav_ab_photon | closed_loop_boed | skip | 0 | 0 | 0.0s |
| compton_clock | single_shot | ✗ | 0 | 15 | 24.3s |
| compton_clock | random_growth | ✓ | 1 | 6 | 33.2s |
| compton_clock | closed_loop_boed | skip | 0 | 0 | 0.0s |
| aharonov_casher | single_shot | ✓ | 0 | 15 | 19.4s |
| aharonov_casher | random_growth | ✓ | 0 | 5 | 10.1s |
| aharonov_casher | closed_loop_boed | skip | 0 | 0 | 0.0s |
| scalar_ab | single_shot | ✓ | 0 | 15 | 19.7s |
| scalar_ab | random_growth | ✓ | 0 | 5 | 9.4s |
| scalar_ab | closed_loop_boed | skip | 0 | 0 | 0.0s |
| atom_grav_gradient | single_shot | ✗ | 0 | 15 | 16.6s |
| atom_grav_gradient | random_growth | ✗ | 10 | 15 | 129.5s |
| atom_grav_gradient | closed_loop_boed | skip | 0 | 0 | 0.0s |
