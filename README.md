# Pair Friedel 3DXRD (pf3dxrd)

## Description
`pf3dxrd` is a Python package for processing scanning 3D X-ray diffraction (s3DXRD) data on a pixel-by-pixel basis, using Friedel pairs. The method is described in detail in [Jacob et al. 2024](https://doi.org/10.1107/S1600576724009634).

**Friedel pairs** are pairs of diffraction spots related by the crystal's centre of symmetry (e.g. $(h,k,l); (-h,-k,-l)$ observed at ω and ω+180°). Matching these pairs makes it possible to relocate the origin of each diffraction event within the sample directly from the 2D peak positions on the detector. This has two main benefits:

- **Point-by-point processing**: peaks can be indexed and mapped independently for each pixel of the sample, rather than requiring a tomographic approach (e.g. using Filtered Back-Projection to reconstruct grain shapes from the sinogram) or peak selection from the sinogram.
- **Source-independent g-vectors**: reciprocal lattice vector coordinates can be computed precisely without depending on an assumed diffraction source position in the sample, removing a significant source of uncertainty tied to peak relocation.

The full processing pipeline — from raw HDF5 frames to indexed grain maps and strain/stress maps — is provided as a series of Jupyter notebooks (see below). It is optimized for ESRF ID11 data, but should work with any s3DXRD dataset following the same file conventions.

`pf3dxrd` is built on top of [FABLE-3DXRD/ImageD11](https://github.com/FABLE-3DXRD/ImageD11), which provides the core diffraction geometry, indexing, and diffraction data management tools. The Friedel pair matching tools have also been moved to ImageD11 and can now be used independently (see ImageD11.friedel_pairs.py and ImageD11.match_friedel_pairs.py)

## pf3dxrd modules
- `crystal_structure.py`: class to store crystal structure information, loaded from a CIF file
- `grain_stress_fit.py`: strain and stress calculation using the `EpsSigSolver` from `ImageD11.stress`
- `local_indexing.py`: tools to perform local (pixel-by-pixel) indexing and refinement of local unit cell matrices, including parallelization and batch processing (SLURM jobs)
- `orientation.py`: crystal orientation module — grain boundary mapping and misorientation analysis
- `phase_mapping.py`: tools to perform phase mapping on a pixel grid
- `peak_mapping.py`: peak-to-pixel and peak-to-grain mapping — manages the labels used to assign diffraction peaks to pixels/grains, and vice versa
- `pixelmap.py`: class to store s3DXRD information (crystal orientation, strain, indexing metrics, etc.) on a 2D pixel grid, including plotting tools
- `refine_ubi.py`: refinement of unit cell matrices from a selection of g-vectors, including tools for merging g-vectors by unique hkl and intensity-based scoring
- `utils.py`: general-purpose functions used across the other modules, mainly for working with ImageD11 columnfiles

### Notebooks
A series of notebooks orchestrate the full pipeline, from raw HDF5 files to indexed grain maps and strain maps. See the README in that folder for details on each step.

There used to be a series of tutorial notebooks with more detailed explanations of each step. These are badly outdated and would need substantial updates to run properly, so they have been moved to the `trash` folder for now. They may still be worth a read if you want to understand the method in more depth.

## Installation
`pf3dxrd` is not yet packaged as a proper Python package — it is currently just a collection of Python modules and scripts. To use it, clone the repository into your project folder:

```git clone git@github.com:jbjacob94/pf3dxrd.git```

## Dependencies
`pf3dxrd` is built on [FABLE-3DXRD/ImageD11](https://github.com/FABLE-3DXRD/ImageD11).

The `crystal_structure.py` module additionally relies on `orix`, `Dans_diffraction`, and `diffpy.structure`:

```python -m pip install Dans_diffraction diffpy.structure orix```


## Usage
To use `pf3dxrd`, import the modules directly into your project (e.g. `from pf3dxrd.pf3dxrd import pixelmap, local_indexing`) after cloning the repository into your project folder.

## Documentation
The processing pipeline is described in detail in [Jacob et al. 2024](https://doi.org/10.1107/S1600576724009634). The notebook series is also densely commented and can be used as a step-by-step guide. For specifics about individual functions, refer to the docstrings — a dedicated documentation file does not yet exist.


## Knows Issues
The Friedel pair relocation method in pf3dxrd relies on discretization of the diffraction peaks: only the centroid positions on the detector $$(y_{det}, z_{det})$$ are used, and diffracting sources are relocated to a single point $$(x_s, y_s, z_s)$$ in the sample. In reality, a diffraction spot has a 2D intensity profile on the detector, and the corresponding diffracting source has a 3D intensity distribution determined by grain boundaries, pencil-beam path across the grain, and orientation / strain gradients. Exploiting the full intensity profiles for more accurate orientation and strain fitting is not possible with this technique and would requires more advanced forward modelling tools.

## License

## Credits
Jean-Baptiste Jacob (jbjacob94)


# Contact
jeanbaptiste.jacob94@gmail.com
