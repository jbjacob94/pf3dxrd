import os, sys, copy
import numpy as np
from tqdm import tqdm
from numba import njit
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.spatial.transform import Rotation as R

import ImageD11.columnfile, ImageD11.grain, ImageD11.refinegrains, ImageD11.cImageD11
import xfab
import matplotlib.pyplot as plt
from orix import quaternion as oq, vector as ovec
import orix.vector as ovec, orix.quaternion as oq
from orix.crystal_map import Phase

from pf3dxrd.pf3dxrd import utils, crystal_structure, pixelmap, peak_mapping



""" 
grain/pixel UBI refinement : refine UBI using all peaks in the selection
"""


#  -----Low-level functions (numba) -----
##########################################

# 3×3 matrix inverse 
@njit(cache=True)
def _inv3x3(m):
    """Returns (success, inverse) for a 3x3 matrix. success=False if singular."""
    det = (m[0,0]*(m[1,1]*m[2,2]-m[1,2]*m[2,1])
          -m[0,1]*(m[1,0]*m[2,2]-m[1,2]*m[2,0])
          +m[0,2]*(m[1,0]*m[2,1]-m[1,1]*m[2,0]))
    if abs(det) < 1e-10:
        return False, m  # singular
    inv = np.empty((3,3), dtype=np.float64)
    inv[0,0] =  (m[1,1]*m[2,2]-m[1,2]*m[2,1]) / det
    inv[0,1] = -(m[0,1]*m[2,2]-m[0,2]*m[2,1]) / det
    inv[0,2] =  (m[0,1]*m[1,2]-m[0,2]*m[1,1]) / det
    inv[1,0] = -(m[1,0]*m[2,2]-m[1,2]*m[2,0]) / det
    inv[1,1] =  (m[0,0]*m[2,2]-m[0,2]*m[2,0]) / det
    inv[1,2] = -(m[0,0]*m[1,2]-m[0,2]*m[1,0]) / det
    inv[2,0] =  (m[1,0]*m[2,1]-m[1,1]*m[2,0]) / det
    inv[2,1] = -(m[0,0]*m[2,1]-m[0,1]*m[2,0]) / det
    inv[2,2] =  (m[0,0]*m[1,1]-m[0,1]*m[1,0]) / det
    return True, inv


# hkl assignment + residuals
@njit(cache=True)
def _compute_hkl_residuals(ubi, gvecs, hkl_tol):
    """
    For each g-vector, compute fractional hkl and squared residual.
    Returns hkl_int, drlv2, mask (peaks within tolerance).
    """
    ng = gvecs.shape[0]
    hkl_int = np.empty((ng, 3), dtype=np.float64)
    drlv2   = np.empty(ng,      dtype=np.float64)
    mask    = np.empty(ng,      dtype=np.bool_)
    tolsq   = hkl_tol * hkl_tol

    for k in range(ng):
        h0 = ubi[0,0]*gvecs[k,0] + ubi[0,1]*gvecs[k,1] + ubi[0,2]*gvecs[k,2]
        h1 = ubi[1,0]*gvecs[k,0] + ubi[1,1]*gvecs[k,1] + ubi[1,2]*gvecs[k,2]
        h2 = ubi[2,0]*gvecs[k,0] + ubi[2,1]*gvecs[k,1] + ubi[2,2]*gvecs[k,2]
        i0 = round(h0); i1 = round(h1); i2 = round(h2)
        t0 = h0 - i0;   t1 = h1 - i1;   t2 = h2 - i2
        sq = t0*t0 + t1*t1 + t2*t2
        hkl_int[k, 0] = i0
        hkl_int[k, 1] = i1
        hkl_int[k, 2] = i2
        drlv2[k]  = sq
        mask[k]   = sq < tolsq

    return hkl_int, drlv2, mask


#  least-squares UB refinement
@njit(cache=True)
def _refine_ubi(ubi, gvecs, hkl_int, mask):
    """
    Refine UBI from indexed peaks. 
    Returns (success, refined_ubi).
    """
    R = np.zeros((3,3))
    H = np.zeros((3,3))
    for k in range(gvecs.shape[0]):
        if not mask[k]:
            continue
        for i in range(3):
            for j in range(3):
                R[i,j] += hkl_int[k,j] * gvecs[k,i]
                H[i,j] += hkl_int[k,j] * hkl_int[k,i]

    ok_H, H_inv = _inv3x3(H)
    if not ok_H:
        return False, ubi

    UB = R @ H_inv
    ok_UB, UBI_new = _inv3x3(UB)
    if not ok_UB:
        return False, ubi

    return True, UBI_new


# cluster g-vectors by hkl
###########################################################################
def cluster_by_hkl(gvecs, intensities, ubi):
    """  Cluster g-vectors by their integer hkl indices and merge redundant peaks. 
    The merged peak’s g-vector is computed as a weighted average of all contributing g-vectors (weights = intensities)

    Returns
    -------
    gv_mean : (M, 3) ndarray of weighted-mean g-vector for each unique (h,k,l) index.
    hklu : (3, M) ndarray of unique integer (h,k,l) triplets corresponding to each merged peak.
    Isum : (M,) ndarray of summed intensity for each unique hkl.
    Imean : (M,) ndarray of averaged intensity for each unique hkl
    """
    # Compute integer hkl indices for each g-vector
    hkls = np.round(ubi @ gvecs.T)
    hklindx = (10000 * hkls[0] + 100 * hkls[1] + hkls[2]).astype(np.int64)

    #  Sort by hkl index to bring identical hkl values together
    order = np.argsort(hklindx)
    hklindx_sorted = hklindx[order]
    gvecs_sorted = gvecs[order]
    intensities_sorted = intensities[order]

    # Identify group boundaries for each unique hkl
    diffs = np.diff(hklindx_sorted, prepend=hklindx_sorted[0] - 1)
    group_starts = np.nonzero(diffs)[0]
    group_ends = np.r_[group_starts[1:], len(hklindx_sorted)]

    #  Weighted averaging within each group (peak merging)
    gv_mean, Isum, Imean = _reduce_groups(group_starts, group_ends, gvecs_sorted, intensities_sorted)
    hklu = hkls[:, order[group_starts]]
    return gv_mean, hklu, Isum, Imean


@njit
def _reduce_groups(group_starts, group_ends, gvecs, intensities):
    n_groups = len(group_starts)
    gv_mean = np.zeros((n_groups, 3))
    Isum = np.zeros(n_groups)
    Imean = np.zeros(n_groups)
    for i in range(n_groups):
        start, end = group_starts[i], group_ends[i]
        ints = intensities[start:end]
        wsum = np.sum(ints)
        wmean = np.mean(ints)
        gv_sum = np.zeros(3)
        for j in range(start, end):
            gv_sum += gvecs[j] * intensities[j]
        gv_mean[i] = gv_sum / wsum
        Isum[i] = wsum
        Imean[i] = wmean
    return gv_mean, Isum, Imean



#  ----- scoring & refine functions -----
##########################################
def score_and_refine(ubi, gvecs, intensities, cs=None, hkl_tol=0.1,
                     refine=True, mergeHKL=True, useIntensity=False):
    """
    Score (and optionally refine) a UBI candidate.
    similar to cImageD11.score in concept but returns additional metrics (drlv2, completeness)
    and includes an option to merge g-vectors by unique hkl before scoring

    Args
    -------
    ubi          : ndarray (3,3) — UBI matrix
    gvecs        : ndarray (N,3) — g-vectors array (gx,gy,gz)
    intensities  : array (N,)    — reflection intensity
    cs           : CrystalStructure object for structure factor computation (optional)
    hkl_tol      : float — hkl tolerance
    refine       : bool  — if True, refine the UBI matrix (mergeHKL is forced True)
    mergeHKL     : bool  — if True, merge g-vectors by unique hkl before scoring
                           (forced True when refine=True)
    useIntensity : bool  — if True, returns intensity correlation score

    Returns
    -------
    ubi_out      : ndarray (3,3) — refined UBI if refine=True, else input UBI
    nindx        : int    — number of indexed peaks (unique if mergeHKL, raw otherwise)
    drlv2        : float  — mean squared hkl residual of indexed peaks
    completeness : float  — fraction of total intensity indexed
    I_indexed    : float  — total intensity indexed
    Iscore       : float  — Pearson correlation of observed vs predicted intensities
                            (nan if useIntensity=False)
    success      : bool   — True if refinement succeeded; always True if refine=False

    Benchmarks for 10-20k gvectors:
    -----------------------------------------------
    score only (refine = False):
    mergeHKL=False, useIntensity=False : ~0.1-0.3 ms
    mergeHKL=True,  useIntensity=False : ~1-3 ms
    mergeHKL=True,  useIntensity=True  : ~10-20 ms
    cImageD11.score (no merging, just nindx): ~ 0.03-0.05 ms
    
    score and refine (refine = True)
    useIntensity=False : ~1-3 ms
    useIntensity=True  : ~10-20 ms
    cImageD11.score_and_refine (no merging) : ~0.1-0.3 ms
    """
    mergeHKL = mergeHKL or refine  # refinement requires merged g-vectors

    # initial filter on raw g-vectors
    hkl_int, drlv2_all, gv_mask = _compute_hkl_residuals(ubi, gvecs, hkl_tol)

    Itot  = intensities.sum()
    nindx = int(gv_mask.sum())
    if nindx < 3:
        return ubi, np.nan, np.nan, np.nan, np.nan, np.nan, False

    if not mergeHKL:
        # fast path: score on raw g-vectors, no merging or refinement
        drlv2_mean   = drlv2_all[gv_mask].mean()
        I_indexed    = intensities[gv_mask].sum()
        completeness = I_indexed / Itot
        return ubi, nindx, drlv2_mean, completeness, I_indexed, np.nan, True

    # merge g-vectors by unique hkl
    gvecs_merged, _, Isum, Imean = cluster_by_hkl(gvecs[gv_mask], intensities[gv_mask], ubi)

    # re-filter on merged g-vectors
    hkl_uniqs, drlv2_m_all, gv_mask_m = _compute_hkl_residuals(ubi, gvecs_merged, hkl_tol)

    nindx_m      = int(gv_mask_m.sum())
    drlv2_mean   = drlv2_m_all[gv_mask_m].mean()
    I_indexed    = Isum[gv_mask_m].sum()
    completeness = I_indexed / Itot

    if refine:
        success, ubi_out = _refine_ubi(
            np.ascontiguousarray(ubi),
            np.ascontiguousarray(gvecs_merged),
            hkl_uniqs,
            gv_mask_m,
        )
        if not success:
            return ubi, np.nan, np.nan, np.nan, np.nan, np.nan, False
    else:
        ubi_out, success = ubi, True

    if useIntensity:
        sF     = cs.str_dans.Scatter.new_structure_factor(hkl_uniqs)
        Icalc  = np.abs(sF)
        Iscore = float(np.corrcoef(Imean[gv_mask_m], Icalc)[0, 1]) if nindx_m > 2 else np.nan
    else:
        Iscore = np.nan

    return ubi_out, nindx_m, drlv2_mean, completeness, I_indexed, Iscore, success




# high-level function: refine pixel list / grains from xmap
####################################################
# ---- module-level worker state ----
_worker_cf  = None
_worker_cs  = None
_worker_kw  = None   # score_and_refine kwargs (hkl_tol, mergeHKL, useInts)

def _init_refine_worker(cf, cs, score_kw):
    global _worker_cf, _worker_cs, _worker_kw
    _worker_cf = cf
    _worker_cs = cs
    _worker_kw = score_kw

# -- chunk wrappers --
def _refine_px_chunk_wrapper(chunk):
    return [_refine_one_pixel(args) for args in chunk]

def _refine_grains_chunk_wrapper(chunk):
    return [_refine_one_grain(g) for g in chunk]

# -- core refinement functions -- 
def _refine_one_pixel(args):
    px, ubi, kernel_size = args
    default = (np.full((3,3), np.nan), np.nan, np.nan, np.nan, np.nan, np.nan)

    s = peak_mapping.pks_from_px(_worker_cf.xyi.astype(int), px, kernel_size)
    if len(s) < 3:
        return px, *default

    gv   = np.array((_worker_cf.gx[s],
                     _worker_cf.gy[s],
                     _worker_cf.gz[s])).T.copy()
    ints = _worker_cf.norm_intensity[s] if 'norm_intensity' in _worker_cf.titles \
           else _worker_cf.sum_intensity[s]

    res = score_and_refine(ubi, gv, ints, cs=_worker_cs, **_worker_kw)
    return px, *res[:-1]   # drop 'success', keep ubi,nindx,drlv2,compl,I_indexed,I_correl

def _refine_one_grain(g, debug=False):
    
    default = (np.full((3,3), np.nan), np.nan, np.nan, np.nan, np.nan, np.nan)

    if not hasattr(g,'pksindx'):
        raise AttributeError(
            "grain {gid} has no attribute 'pksindx'. Skip it".format(gid=g.gid))
        return g

    gv = np.column_stack([_worker_cf.gx[g.pksindx],
                          _worker_cf.gy[g.pksindx],
                          _worker_cf.gz[g.pksindx]]).copy()
    
    ints = _worker_cf.norm_intensity[g.pksindx] if 'norm_intensity' in _worker_cf.titles \
           else _worker_cf.sum_intensity[g.pksindx]

    res = score_and_refine(g.ubi, gv, ints, cs=_worker_cs, **_worker_kw)

    # add results to grain attributes
    names = 'ubi,nindx,drlv2,completeness,I_indexed,I_corr'.split(',')
    for i, name in enumerate(names):
        setattr(g, name, res[i])
    return g


# ---- main functions ----
def refine_px_ubis(cf, pxlist, UBIs, ncpu=1, chunksize=50, hkl_tol=0.05,
                   cs=None, useInts=False, kernel_size=1):

    assert cf.sortedby == 'xyi', 'peakfile not sorted correctly. please sort by xyi column'

    score_kw = dict(hkl_tol=hkl_tol, refine=True, useIntensity=useInts)
    argslist = [(int(px), ubi, kernel_size) for px, ubi in zip(pxlist, UBIs)]
    chunks     = [argslist[i:i+chunksize] for i in range(0, len(argslist), chunksize)]

    ctx = multiprocessing.get_context('fork')

    with ProcessPoolExecutor(
        max_workers=max(ncpu, 1),
        mp_context=ctx,
        initializer=_init_refine_worker,
        initargs=(cf, cs, score_kw),
    ) as pool:

        futures = {
            pool.submit(_refine_px_chunk_wrapper, chunk): chunk
            for chunk in chunks}

        results = {}

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='pixels refined',
        ):
            try:
                for r in future.result():    # r = (px, ubi, nindx, ...)
                    results[r[0]] = r[1:]
            except Exception as exc:
                chunk = futures[future]
                print(f'[ERROR] chunk starting at px={chunk[0][0]} raised: {exc}')

    return results

def refine_grains_ubis(cf, glist, ncpu=1, chunksize=10, hkl_tol=0.3, cs=None, useInts=False):

    assert cf.sortedby == 'xyi', 'peakfile not sorted correctly. please sort by xyi column'

    score_kw = dict(hkl_tol=hkl_tol, refine=True, useIntensity=useInts)
    chunks     = [glist[i:i+chunksize] for i in range(0, len(glist), chunksize)]

    ctx = multiprocessing.get_context('fork')

    with ProcessPoolExecutor(
        max_workers=max(ncpu, 1),
        mp_context=ctx,
        initializer=_init_refine_worker,
        initargs=(cf, cs, score_kw),
    ) as pool:

        futures = {
            pool.submit(_refine_grains_chunk_wrapper, chunk): chunk
            for chunk in chunks}

        refined_grains = []

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='grains refined',
        ):
            try:
                for g in future.result():    
                    refined_grains.append(g)
            except Exception as exc:
                chunk = futures[future]
                print(f'[ERROR] chunk starting at grain={chunk[0].gid} raised: {exc}')

    return refined_grains

    
def compute_refinement_stats(results):
    """
    Unpack refinement /indexing results, compute summary statistics, and plot distributions.

    results : dict  {px: [ubi, nindx, drlv2, completeness, I_indexed, I_corr]}
    returned by refine_px_ubis / local_indexing.run_indexing_parallel
    

    Returns
    -------
    stats : dict  {metric_name: {'mean': float, 'q05': float, 'q50': float, 'q95': float}}
    """
    # --- unpack ---
    valid = {px: v for px, v in results.items() if not np.isnan(v[1])}  # drop failed pixels
    
    nindx        = np.array([v[1] for v in valid.values()])
    drlv2        = np.array([v[2] for v in valid.values()])
    completeness = np.array([v[3] for v in valid.values()])
    I_indexed    = np.array([v[4] for v in valid.values()])
    I_corr       = np.array([v[5] for v in valid.values()])

    n_total  = len(results)
    n_valid  = len(valid)
    n_failed = n_total - n_valid

    metrics = {
        'nindx'       : nindx,
        'drlv2'       : drlv2,
        'completeness': completeness,
        'I_{indexed}'   : I_indexed,
        'I_{corr}'      : I_corr,
    }
    labels = {
        'nindx'       : ('N indexed peaks',     'counts'),
        'drlv2'       : (r'mean drlv2',         'a.u.'),
        'completeness': ('Completeness',         'fraction'),
        'I_{indexed}'    : ('Intensity indexed',    'a.u.'),
        'I_{corr}'       : ('Intensity correlation', 'Pearson r'),
    }

    # --- stats ---
    def _metric_stats(arr):
        """Compute stats for one metric. Returns None if no valid values."""
        a = arr[~np.isnan(arr)]
        if len(a) == 0:
            return None
        return {
            'mean': np.mean(a),
            'q05' : np.quantile(a, 0.05),
            'q50' : np.quantile(a, 0.50),
            'q95' : np.quantile(a, 0.95)}
    
    stats = {}
    for name, arr in metrics.items():
        s = _metric_stats(arr)
        if s is None:
            print(f'[INFO] {name}: no valid values, skipping.')
        stats[name] = s   # may be None

    # --- print summary ---
    print(f'Pixels total: {n_total}  |  refined: {n_valid}  |  failed: {n_failed}')
    print(f'\n{"metric":<16} {"mean":>12} {"q05":>12} {"q50":>12} {"q95":>12}')
    print('-' * 64)
    for name, s in stats.items():
        if s is None:
            print(f'{name:<16}  —  no valid data')
            continue
        med   = s['q50']
        exp   = int(np.floor(np.log10(abs(med)))) if med != 0 else 0
        scale = 10 ** exp if abs(exp) >= 3 else 1
        suffix = f'  (×10^{exp})' if scale != 1 else ''
        def fmt(v): 
            vs = v / scale
            return f'{vs:.1f}' if abs(vs) >= 10 else f'{vs:.3f}'
        print(f'{name:<16} {fmt(s["mean"]):>12} {fmt(s["q05"]):>12} {fmt(s["q50"]):>12} {fmt(s["q95"]):>12}{suffix}')

    # --- plot ---
    fig, axes = plt.subplots(2,3, figsize=(12,8))
    axes = axes.ravel()
    axes[-1].set_axis_off()
    fig.suptitle(f'Refinement metrics  —  {n_valid}/{n_total} pixels refined', fontsize=11)

    for ax, (name, arr) in zip(axes, metrics.items()):
        s = stats[name]
        label, unit = labels[name]
        if s is None:
            ax.text(0.5, 0.5, 'no valid data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='grey')
            ax.set_title(f'$\\bf{{{name}}}$', fontsize=8)
            ax.set_xlabel(labels[name][0], fontsize=9)
            continue

        # pick a shared scale factor from the median
        med   = s['q50']
        exp   = int(np.floor(np.log10(abs(med)))) if med != 0 else 0
        scale = 10 ** exp if abs(exp) >= 3 else 1   # only rescale if |exp| >= 3
        a_sc  = arr[~np.isnan(arr)] / scale

        ax.hist(a_sc, bins=50, color='steelblue', alpha=0.75, edgecolor='none')

        for q, ls, color in [(s['q05'], '--', 'orange'),
                              (s['q50'], '-',  'red'),
                              (s['q95'], '--', 'orange')]:
            ax.axvline(q / scale, color=color, linestyle=ls, linewidth=1.2)
        ax.axvline(s['mean'] / scale, color='white', linestyle=':', linewidth=1.2)

        # x-axis label with scale factor if applied
        xlabel = f'{label} [{unit}]' if scale == 1 else f'{label} [{unit}] (×10$^{{{exp}}}$)'
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('counts', fontsize=9)

        # format stat values: scaled if needed
        def fmt(v):
            vs = v / scale
            return f'{vs:.1f}' if abs(vs) >= 10 else f'{vs:.3f}'

        ax.set_title(
            f'$\\bf{{{name}}}$\n'
            f'mean={fmt(s["mean"])}  q50={fmt(s["q50"])}\n'
            f'[{fmt(s["q05"])}, {fmt(s["q95"])}]',
            fontsize=8
        )

    plt.tight_layout()
    plt.show()

    return stats, fig
