# pair friedel 3DXRD (pf3dxrd)


## Description
This repository contains tools to process scanning 3D X-ray diffraction (3DXRD) data on a per-pixel basis, based on the relocation of diffracted X-rays in the sample using Friedel pairs following the method described in [Jacob et al. 2024](https://doi.org/10.1107/S1600576724009634). Symetric reflections **(h,k,l)**, **(-h,-k,-l)** arising from the same spot in a crystal are identified and paired together. The symmetry properties of these pairs are used to correct the diffraction angle (2-theta) and relocate the center-of-mass of the diffracting region in the sample. Friedel pairs source centers are then mapped onto a 2D pixel grid, which allows per-pixel fitting of crystallographic phase and lattice vectors (local indexing). 

The pf3dxrd package is built upon ImageD11 ([FABLE-3DXRD/ImageD11](https://github.com/FABLE-3DXRD/ImageD11)) and contains the following modules:
### pf3dxrd
- friedel_pairs.py: Friedel pairs identification and geometry corrections allowing diffractiion vector (g-vector) fitting and diffracting source relocation
- crystal_structure.py: A class to store crystal structure information, loaded from a cif file
- grain_stress_fit.py: strain and stress calculation using the EpsSigSolver from ImageD11.stress 
- phase_mapping.py: A class to perform phase mapping on a pixel grid
- peak_mapping.py: Peaks-to-pixel and peaks-to-grains mapping. Manage labels to assign diffraction peaks to pixels/grains and grains/pixels to peaks. Also includes functions for 
- pixelmap.py: A class to store outputs from local indexing on a pixel grid and make 2D plots.
- utils.py: general functions used in other modules, mainly to work on ImageD11 columnfiles

### scripts
Scripts to be executed in command lines from a terminal. Makes batch processing of a series of peakfiles more handy. 
- find_friedel_pairs.py : Run Friedel pairs search for all scans in a peakfile/peaks table, using parallelization on multiple cores for faster computation
- local_indexing.py : Run local indexing (find lattice vectors matrix UBI on each pixel) on a peakfile, using parallelization on multiple cores

### NB_tutorials
A detailed tutorial organized in a series of Jupyter notebooks, to obtain 2D orientation and strain /stress maps from a raw peakfile. 

### NB_examples
Shortened versions of the Jupyter notebooks in NB_tutorials, which can be more easily adapted to build your processing workflow

## Installation
So far, pf3dxrd is not a proper package and only consists of a collection of python modules and scripts.
To use it, just clone the repository to your project folder. 

```git clone git@github.com:jbjacob94/pf3dxrd.git```

## Dependencies
pf3dxrd is built upon [FABLE-3DXRD/ImageD11](https://github.com/FABLE-3DXRD/ImageD11). 

crystal_structure.py module relies on orix, Dans_diffraction and diffpy.structure modules:

```python -m pip install Dans_diffraction diffpy.structure orix```

## Usage
To use the pf3dxr module, just import it into your project file. 

## Documentation
Details about the processing pipeline are explained in the dedicated publication [Jacob et al. 2024](https://doi.org/10.1107/S1600576724009634). Jupyter Notebooks tutorials with example datasets are available for getting started. For more specific information about each function, read the docstrings (no detailed doc file yet).

## Knows Issues
The Friedel pair relocation method in pf3dxrd relies on discretization of the diffraction peaks: only the centroid positions on the detector $$(y_{det}, z_{det})$$ are used, and diffracting sources are relocated to a single point $$(x_s, y_s, z_s)$$ in the sample. In reality, a diffraction spot has a 2D intensity profile on the detector, and the corresponding diffracting source has a 3D intensity distribution determined by grain boundaries, pencil-beam path across the grain, and orientation / strain gradients. Exploiting the full intensity profiles for more accurate orientation and strain fitting is not possible with this technique and requires more advanced forward modelling tools.

## License

## Credits
Jean-Baptiste Jacob (jbjacob94)


# Contact
j.b.jacob@mn.uio.no
