# Scanning-3DXRD (Friedel Pairs) Processing Pipeline

This pipeline processes scanning-3DXRD (s3DXRD) data using the Friedel-pair route: peaks are segmented, paired by symmetric geometry, phase- and pixel-indexed, assembled into grains, and finally used to compute strain and stress. Notebooks are numbered `fp0`–`fp5` to indicate execution order; `b`-suffixed notebooks are batch/alternative variants of the step they follow.

## Pipeline steps

**fp0 — Segmentation & peak labelling**
Common entry point for all s3DXRD routes (tomo, point-by-point, Friedel pairs). Converts raw HDF5 frames into a sparse peak file, then merges 2D peaks into 4D peaks (along omega and dty) to build the peaks table. Includes optional intensity normalization to a monitor signal (e.g. a pico).

**fp0b — Fit y0**
Precisely fits `y0`, the y-coordinate of the incident beam in the lab frame. This is critical for Friedel pair matching, since pairs are searched symmetrically around `y0`. It is fitted using eta (Friedel) pairs of 4D peaks: their positions are reconstructed in sample space for a range of `y0` guesses, and the value giving the sharpest reconstructed image (max standard deviation) is selected.

**fp1 — Friedel pair matching** *(+ fp1b for batch/SLURM)*
Matches Friedel pairs using `ImageD11.friedel_pairs`. Peaks are split into mirror "chunks" (dty bins for omega pairs, eta bins for eta pairs) so that pairing is done locally and in parallel rather than globally, which is far more memory-efficient. Both omega pairs (main route) and eta pairs (optional) can be matched. `fp1` is for tuning matching parameters on one dataset; `fp1b` (via `ImageD11.match_friedel_pairs.py`) runs the same matching as a standalone script, either locally or submitted as a SLURM job per dataset — recommended for full datasets since it allows more CPU/memory allocation and doesn't require keeping a Jupyter session open.

**fp2 — Phase labelling**
Assigns a dominant crystal phase to each pixel of a 2D grid (`Pixelmap` class) built from the scan geometry. For each candidate phase (loaded from CIF via the `crystal_structure` class), a peak-selection mask is computed from theoretical 2θ positions; pixel-wise conflicts between overlapping phase masks are resolved by maximizing a completeness/uniqueness criterion. Assumes one dominant phase per pixel — not intended for mapping minor/inclusion phases, which can instead be exported as residual peaks and processed separately.

**fp3 — Local (pixel-by-pixel) indexing** *(+ fp3b for batch/SLURM)*
Indexes orientation and unit cell independently for each pixel and writes results onto the `Pixelmap`. `fp3` is used to tune indexing options (tolerances, ring/multiplicity limits, etc.) on a single phase/dataset before scaling up. `fp3b` reuses the saved options to submit one SLURM job per dataset (single phase at a time; polymineralic samples require re-running per phase). Special handling (`flipmats`) is provided for phases where the lattice symmetry is higher than the point-group symmetry (e.g. trigonal quartz indexed within a hexagonal lattice), where intensity-based scoring is needed to disambiguate equivalent lattice orientations.

**fp4 — Grain mapping** *(+ fp4b alternative via DBSCAN)*
Groups indexed pixels into grains. `fp4` segments grains from misorientation-based grain boundaries (connected-component analysis, conceptually similar to MTEX's `calcGrains`), then refines each grain's average UBI using all peaks within its mask, and computes misorientation metrics (KAM, GROD). Results can be exported to MTEX-compatible `.ctf` files. `fp4b` is an alternative approach using density-based clustering (DBSCAN) in orientation(+position) space, better suited to 3D stacks or grain tracking in time series, though slower for large maps.

**fp5 — Strain and stress mapping**
Computes elastic strain (Green-Lagrange, via `ImageD11.finite_strain`) and stress (Hooke's law via `ImageD11.stress`) across the indexed map or grains, using the `grain_stress_fit.py` module. Requires a defined reference unit cell (B₀), a consistent reference frame (crystal vs. sample) matched to the elastic constants used, and phase-specific stiffness tensors. Outputs strain/stress tensors, principal components, and invariants (hydrostatic, von Mises, deviatoric), which can be plotted per-component or exported to VTK/TensorMap for visualization (e.g. in ParaView).

## Batch processing across datasets — `run_papermill`

For steps not yet wrapped as standalone SLURM scripts (fp0, fp0b, fp2, fp4, fp4b, fp5), `run_papermill.ipynb` allows batch execution of a notebook across many datasets using `papermill`. A "parameters" cell is tagged in the target notebook, and `papermill.execute_notebook()` is called in a loop over datasets, overriding parameters such as the dataset path each time. This avoids manually re-running and editing the same notebook for every dataset, though it requires keeping an active session running while the batch executes. For fp1 and fp3, submitting directly to SLURM (via fp1b/fp3b) remains the preferred approach, since it scales better on compute resources and doesn't require an active session.