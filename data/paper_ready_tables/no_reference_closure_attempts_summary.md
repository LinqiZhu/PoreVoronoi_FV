# No-reference/local closure attempts summary, 2026-06-06

This is a data-only screening summary. It gathers deterministic local closures that were tested while trying to avoid reference-tuned parameters. These rows are not the retained production method unless explicitly stated.

| Family | Closure | Cases | eK min/median/max (%) | Best | Worst | Decision |
| --- | --- | --- | ---: | --- | --- | --- |
| capacity-wall local Dirichlet closure | `capacity_wall_dirichlet_interface` | maze; Bentheimer; fibrous; orthogonal | 1.87/34.16/116.39 | Bentheimer | fibrous | rejected for main method |
| capacity-wall local Dirichlet closure | `original_geometry` | maze; Bentheimer; fibrous; orthogonal | 10.29/17.14/124.92 | orthogonal | fibrous | diagnostic baseline |
| capacity-wall local Dirichlet closure | `p0_face_capacity_wall` | maze; Bentheimer; fibrous; orthogonal | 13.25/55.27/76.02 | Bentheimer | maze | rejected for main method |
| capacity-wall local Dirichlet closure | `p0_galerkin_face_only` | maze; Bentheimer; fibrous; orthogonal | 3.03/49.90/81.95 | Bentheimer | fibrous | rejected for main method |
| capacity-wall local Dirichlet closure | `p0_galerkin_face_wall` | maze; Bentheimer; fibrous; orthogonal | 35.68/75.44/115.07 | Bentheimer | fibrous | rejected for main method |
| capacity-wall local Dirichlet closure | `p0_galerkin_wall_only` | maze; Bentheimer; fibrous; orthogonal | 30.22/43.47/458.18 | Bentheimer | fibrous | rejected for main method |
| torsion/shape-factor local closure | `hydraulic_size_factor_network_centroid` | Bentheimer; fibrous | 73.01/77.26/81.51 | Bentheimer | fibrous | rejected for main method |
| torsion/shape-factor local closure | `hydraulic_size_factor_network_voxel_throat` | Bentheimer; fibrous | 7.54/7.68/7.83 | fibrous | Bentheimer | prior-tech screening only |
| torsion/shape-factor local closure | `shape_face_original_wall` | Bentheimer; fibrous | 89.81/106.83/123.84 | Bentheimer | fibrous | rejected for main method |
| torsion/shape-factor local closure | `torsion_wall` | Bentheimer; fibrous | 6.39/115.15/223.91 | Bentheimer | fibrous | rejected for main method |
| torsion/shape-factor local closure | `torsion_wall_shape_face` | Bentheimer; fibrous | 50.41/63.29/76.18 | Bentheimer | fibrous | rejected for main method |
| star-Schur local closure | `original_geometry` | Bentheimer; fibrous; orthogonal | 10.29/11.85/124.92 | orthogonal | fibrous | diagnostic baseline |
| star-Schur local closure | `star_cell_average_schur_momentum` | Bentheimer; fibrous; orthogonal | 47.33/864.80/1206.55 | Bentheimer | fibrous | rejected for main method |
| star-Schur local closure | `star_schur_faces_capacity_wall` | Bentheimer; fibrous; orthogonal | 7.26/58.32/96.42 | Bentheimer | fibrous | rejected for main method |
| star-Schur local closure | `star_schur_faces_original_wall` | Bentheimer; fibrous; orthogonal | 4.35/20.47/142.09 | Bentheimer | fibrous | rejected for main method |

Use in manuscript:

- These rows support a guarded statement that several deterministic no-reference closures were screened but were not adopted as the retained operator.
- A closure that performs well on Bentheimer alone is not enough to claim a universal wall/throat closure, especially when fibrous or procedural cases degrade.
- The retained production claims should continue to use `main_solver_table_for_table1.csv` and `current_variant_decision_table_for_tex.csv`.
