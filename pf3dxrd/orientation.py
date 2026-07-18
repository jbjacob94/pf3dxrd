import os, sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pylab as pl
from tqdm import tqdm
from numba import njit, prange

import scipy.ndimage as ndi

import ImageD11.cImageD11
import ImageD11.columnfile
import ImageD11.grain
import ImageD11.refinegrains
import ImageD11.unitcell
import xfab
from orix import data, io, plot as opl, quaternion as oq, vector as ovec
from pf3dxrd.pf3dxrd import utils, crystal_structure



""" 
crystal orientation module. Grain boundary mapping based on misorientation threshold + misorientation analysis:
Intra-granular misorientation, Kernel Averaged misorientation (KAM)
"""


def segment_grains(xmap, pname, Ucol='U', threshold_deg=10, min_grain_size=3):
    """
    Segment grains from a pixel orientation map using misorientation threshold.
    Similar conceptually to MTEX's calcGrains.

    Parameters
    -----------
    xmap : Pixelmap oject with pixel orientations
    pname (str) : Phase name
    Ucol (str)  : Key for pixel orientation array in xmap (default = 'U')
    threshold_deg : Misorientation threshold (in degree) for grain boudary identification
    min_grain_size (int) : Minimum grain size in pixels. Grains smaller than this are reset as unlabeled

    Returns:
    grain_ids (ndarray) : labeled grain map
    gb_mask  (bool array) : grain boundary mask
    mis_max (ndarray) : max misorientation (between x and y misorientation) map
    """
    assert pname in xmap.phases.pnames, 'Phase name not recognized'

    # Select phase
    xmap_p = xmap.filter_by_phase(pname)
    nx, ny = xmap.grid.nx, xmap.grid.ny
    cs = xmap_p.phases.get(pname)
    uc = ImageD11.unitcell.unitcell(cs.cell, cs.spg_no)
    pm = xmap.get_phase_mask(pname).reshape(nx,ny)
    sym = cs.orix_phase.point_group.laue

    # Get orientation array
    ori = uc.get_orix_orien_fast(xmap_p.get(Ucol))
    ori = ori.reshape((nx, ny))

    # Misorientation with right and bottom neighbors
    mis_x = oq.Misorientation(ori[:, :-1] * ~ori[:, 1:], symmetry=(sym, sym)).to_axes_angles() 
    mis_y = oq.Misorientation(ori[:-1, :] * ~ori[1:, :], symmetry=(sym, sym)).to_axes_angles() 
    mis_x_deg = np.degrees(mis_x.angle)
    mis_y_deg = np.degrees(mis_y.angle)

    # take the max between x and y misorientation + pad image to get back to original grid size
    mis_max = np.maximum(mis_x_deg[:-1,:], mis_y_deg[:,:-1])
    mis_max = np.pad(mis_max, ((1, 0), (1, 0)), mode='constant', constant_values=0)

    # Connectivity masks: True where within same grain
    same_grain_mask = (mis_max < threshold_deg) & pm
    gb_mask = mis_max >= threshold_deg 
    
    # Label connected components
    grain_ids, n_grains = ndi.label(same_grain_mask)  # 4-connectivity
    uniq_labels, npix = np.unique(grain_ids.flatten(), return_counts=True)
    grain_sizes = {gid: n for gid, n in zip(uniq_labels,npix) if gid != 0}

    # Remove small grains if requested
    if min_grain_size > 0:
        small_ids = [gid for gid, n in grain_sizes.items() if n < min_grain_size]
        if small_ids:
            mask_small = np.isin(grain_ids, small_ids)
            same_grain_mask[mask_small] = False
            gb_mask[mask_small] = True
            # recompute connected domains after removal to relabel properly
            grain_ids, n_grains = ndi.label(same_grain_mask)
            uniq_labels, npix = np.unique(grain_ids.flatten(), return_counts=True)
            grain_sizes = {gid: n for gid, n in zip(uniq_labels,npix) if gid != 0}

    n_grain = grain_sizes.keys()
    print(f"Identified {n_grains} grains for phase '{pname}'")

    return grain_ids, gb_mask, mis_max



def local_misorientation(xmap, pname, Ucol='U', kernel_size=3, threshold_deg=10, mode='mean', use_numba_acceleration=True):
    """
    Compute Kernel Average (KAM) or Kernel Median Misorientation (KMM). similar to MTEX's function. 
    
    Parameters
    ----------
    xmap   : Pixelmap Object containing orientation data.
    pname  : (str) Phase name.
    Ucol   : (str) Column containing orientation matrices.
    kernel_size   : (int) Size of local kernel (odd integer).
    threshold_deg : (float) Max misorientation to include in averaging.
    mode : {'mean', 'median'} Averaging mode.
    use_numba_acceleration : bool
        If True -> fast Numba implementation (ignores symmetry but faster).
        If False -> accurate Orix implementation (handles symmetry, slower).
    
    Returns
    -------
    kam_map : (ndarray) Map of local misorientation values (in degrees).
    """
    assert pname in xmap.phases.pnames, 'Phase name not recognized'
    assert mode in ('mean', 'median')

    # Extract orientation data
    xmap_p = xmap.filter_by_phase(pname)
    cs = xmap_p.phases.get(pname)
    sym = cs.orix_phase.point_group.laue

    ori = oq.Orientation.from_matrix(xmap_p.get(Ucol), symmetry=sym)
    nx, ny = xmap.grid.nx, xmap.grid.ny
    U = ori.to_matrix().reshape(nx, ny, 3, 3)

    threshold_rad = np.radians(threshold_deg)

    if use_numba_acceleration:
        print("⚡ Numba-accelerated mode (approximate, ignores symmetry)...")
        kam_map = _local_misorient_numba(U, kernel_size, threshold_rad, mode_mean=(mode == 'mean'))
        kam_map = np.degrees(kam_map)
    else:
        print("Orix symmetry-accurate mode (slower)...")
        kam_map = _local_misorient_orix(U, sym, kernel_size, threshold_rad, mode)

    return kam_map


@njit(parallel=True, fastmath=True)
def _local_misorient_numba(U, kernel_size=3, threshold_rad=0.1745, mode_mean=True):
    """
    Numba-accelerated local misorientation (KAM) computation.
    Returns Local mean/median misorientation per pixel (radians).
    """
    nx, ny = U.shape[:2]
    r = kernel_size // 2
    kam_map = np.zeros((nx, ny))

    for i in prange(r, nx - r):
        for j in range(r, ny - r):
            center = U[i, j]
            vals = np.empty(kernel_size * kernel_size)
            n_valid = 0

            for ii in range(-r, r + 1):
                for jj in range(-r, r + 1):
                    if ii == 0 and jj == 0:
                        continue
                    neigh = U[i + ii, j + jj]

                    # Compute misorientation angle via trace formula
                    Rt = np.dot(neigh, center.T)
                    tr = Rt[0, 0] + Rt[1, 1] + Rt[2, 2]
                    val = (tr - 1.0) * 0.5
                    if val > 1.0: val = 1.0
                    if val < -1.0: val = -1.0
                    angle = np.arccos(val)

                    if angle < threshold_rad:
                        vals[n_valid] = angle
                        n_valid += 1

            if n_valid > 0:
                if mode_mean:
                    kam_map[i, j] = np.mean(vals[:n_valid])
                else:
                    kam_map[i, j] = np.median(vals[:n_valid])
            else:
                kam_map[i, j] = 0.0

    return kam_map


def _local_misorient_orix(U, sym, kernel_size=3, threshold_rad=0.1745, mode='mean'):
    nx, ny = U.shape[:2]
    r = kernel_size // 2
    kam_map = np.zeros((nx, ny))
    iterator = tqdm(range(r, nx - r), total=len(range(r, nx - r)))
    
    for i in iterator:
        for j in range(r, ny - r):
            center = oq.Orientation.from_matrix(U[i, j], symmetry=sym)
            neighbors = []
            for ii in range(-r, r + 1):
                for jj in range(-r, r + 1):
                    if ii == 0 and jj == 0:
                        continue
                    neighbors.append(oq.Orientation.from_matrix(U[i + ii, j + jj], symmetry=sym))
            neighbors = oq.Orientation.stack(neighbors)

            mis = oq.Misorientation(neighbors * ~center, symmetry=(sym, sym))
            angles = mis.angle
            valid = angles[angles < threshold_rad]
            if valid.size > 0:
                if mode == 'mean':
                    kam_map[i, j] = np.degrees(valid.mean())
                else:
                    kam_map[i, j] = np.degrees(np.median(valid))
    return kam_map


def local_orientation_smooth_orix(U, sym, kernel_size=3, global_mask = None, local_mask=None, threshold_deg=None):
    """
    Smooth orientations using local geodesic averaging (SO(3)).

    Parameters
    ----------
    U   : ndarray (nx, ny, 3, 3)
        orientation matrices
    sym : orix.symmetry.Symmetry
        Crystal symmetry
    kernel_size : int 
        Sliding window size (odd)
    threshold_deg : float
        Maximum misorientation (degree) to include neighbor
    global_mask : ndarray (nx, ny), bool
        pixel selection mask applied globally on the map (e.g. for phase selection)
    local_mask : ndarray (nx, ny), int
        label map to apply local mask on per-pixel basis: for a pixel (i,j) with label value k,
        builds local mask to keep only pixel neighbours sharing the same label value k. 
        typical use-case: grain-based selection for orientation refinement

    Returns
    -------
    U_smooth : ndarray (nx, ny, 3, 3) – Smoothed orientation matrices
    """
    
    nx, ny = U.shape[:2]
    r = kernel_size // 2

    if threshold_deg is None:
        threshold_deg =361

    if global_mask is None:
        global_mask = np.full((nx,ny),True)
    if local_mask is None:
        local_mask = np.ones((nx, ny), dtype=int)

    ori = oq.Orientation.from_matrix(U.reshape(-1, 3, 3), symmetry=sym).reshape(nx, ny)

    # sliding window for selection
    windows = sliding_window_view(ori.data, (kernel_size, kernel_size, 4))
    gmask_win = sliding_window_view(global_mask, (kernel_size, kernel_size))
    lmask_win = sliding_window_view(local_mask, (kernel_size, kernel_size))

    U_smooth = np.zeros_like(U)

    for i in tqdm(range(nx - 2*r)):
        for j in range(ny - 2*r):
            
            # Skip smoothing if center pixel is masked out
            if not global_mask[i + r, j + r]:
                continue

            # local selection of orientations
            qwin = windows[i, j].reshape(-1, 4)
            gmwin = gmask_win[i, j].reshape(-1)
            lmwin = lmask_win[i, j].reshape(-1)

            # Mask-based exclusion
            valid = gmwin
            valid &= lmwin == lmwin[len(lmwin)//2]

            # central pixel + neighbour orientations
            center = oq.Orientation(qwin[len(qwin)//2], symmetry=sym)
            neighbors = oq.Orientation(qwin, symmetry=sym)

            # misorientation filtering
            mis = oq.Misorientation(neighbors * ~center, symmetry=(sym, sym))
            valid &= np.rad2deg(mis.angle) < threshold_deg

            # smoothing
            if np.any(valid):
                mean_ori = neighbors[valid].mean()
                U_smooth[i+r, j+r] = mean_ori.to_matrix()

    return U_smooth


def compute_GROD(xmap, phase=None, pixel_orientation="U", reference_frame="sample", axis_coordinates="cartesian", degrees=True):
    """
    Compute Grain Reference Orientation Deviation (GROD) — misorientation of each pixel from its grain mean orientation, modulo crystal symmetry.
    Mean grain orientation is taken from the grain list in xmap.grains. 
    
    Parameters
    ----------
    xmap : PixelMap  object with orientation data and computed grains.
    phase (str): phase name. If none, compute GROD for all phases
    pixel_orientation (str) : data column with pixel orientation matrices (default 'U').
    reference_frame {'sample', 'crystal'}: Frame in which the misorientation axis is expressed.
    axis_coordinates {'cartesian', 'polar'}: Output axis coordinate type: cartesian (x,y,z) or polar (azimuth, dip)
    degrees (bool) :  Return angles and polar coordinates in degrees. Default is True

    Returns: GROD angle and axis ndarrays as a dictionnary

    NOTE: GROD axis orientation can be returned either in polar or cartesian coordinates. To plot an ipf color map of axis orientation, use cartesian coordinates. 
    For axis orientation in the sample reference frame, use a full color key (symmetry='C1'), as the misorientation axes in specimen coordinates have no symmetry. 
    """
    assert pixel_orientation in xmap.titles(), "orientation data not recognized"
    assert len(xmap.grains.glist) > 0, "no grains found — run grain segmentation first"
    assert axis_coordinates in ("cartesian", "polar"), "invalid axis coordinate type"
    assert reference_frame in ("sample", "crystal"), "invalid reference frame"

    # Initialize arrays
    GROD_angle = np.zeros(xmap.xyi.shape, dtype=float)
    if axis_coordinates == "cartesian":
        GROD_axis_xyz = np.zeros(xmap.xyi.shape + (3,), dtype=float)
    else:
        GROD_axis_polar = np.zeros(xmap.xyi.shape + (2,), dtype=float)

    # Get pixel orientations
    U_px = xmap.get(pixel_orientation)

    # Iterate through grains
    for gi, g in tqdm(zip(xmap.grains.gids, xmap.grains.glist), total=len(xmap.grains.glist), desc="Computing GROD"):
        if phase is not None and g.phase != phase:
            continue
            
        gm = g.pxindx
        pname = g.phase
        cs = xmap.phases.get(pname).orix_phase
        sym = cs.point_group.laue

        ori_px = oq.Orientation.from_matrix(U_px[gm], symmetry=sym).map_into_symmetry_reduced_zone()
        ori_ref = oq.Orientation.from_matrix(g.U, symmetry=sym)

        GROD = oq.Misorientation(ori_px * ~ori_ref, symmetry=(sym, sym))
        axis = GROD.axis.flatten()

        if reference_frame == "sample":
            axis = ~ori_ref * axis

        if reference_frame == "crystal":
            axis = axis.in_fundamental_sector(sym)

        GROD_angle[gm] = np.degrees(GROD.angle) if degrees else GROD.angle

        if axis_coordinates == "cartesian":
            GROD_axis_xyz[gm] = axis.data
        else:
            az, dip = axis.azimuth, axis.polar
            if degrees:
                az, dip = np.degrees(az), np.degrees(dip)
            GROD_axis_polar[gm] = np.column_stack((az, dip))

    result = {"angle": GROD_angle}
    if axis_coordinates == "cartesian":
        result["axis_xyz"] = GROD_axis_xyz
    else:
        result["axis_polar"] = GROD_axis_polar

    return result
    
