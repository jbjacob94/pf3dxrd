import os, sys, copy
import numpy as np
from tqdm import tqdm
from numba import njit

from scipy.spatial.transform import Rotation as R

import ImageD11.columnfile, ImageD11.grain, ImageD11.refinegrains, ImageD11.cImageD11
import xfab
from orix import quaternion as oq, vector as ovec
import orix.vector as ovec, orix.quaternion as oq
from orix.crystal_map import Phase

from pf3dxrd.pf3dxrd import utils, crystal_structure, pixelmap



""" 
- Peakfile-to-pixelmap / peakfile-to-grainmap mapping: 

select peaks corresponding to a pixel or a grain mask from a 2D map and assign grain labels / pixel labels to 
the peaks in corresponding peakfile. 

- g-vectors merging:
For a given set of g-vectors and a UBI matrix, group g-vectors by unique hkl index and return mean g-vector + total and average intensity
"""


# Peak mapping on a 2D pixel grid and peak selection by pixel index
###########################################################################

def xyi(xi, yi):
    """ Converts (xi,yi) pixel coordinates to a unique index xyi = xi + 10000 * yi (used in pixelmap).
    Only works if the map is less than 10000 px wide, which should normally be the case"""
    return int(xi+10000*yi)


def xyi_inv(xyi):
    """ converts xyi index to (xi,yi) pixel coordinates"""
    xi = xyi % 10000
    yi = xyi // 10000
    return xi, yi


def add_pixel_labels(cf, ds, y0=None):
    """
    Compute pixel coordinates (xi,yi) and pixel index (xyi) for each peak in a peakfile cf, using (sx,sy) coordinates in sample space. 
    Adds new columns xi, yi and xyi to the peakfile
    
    Args:
    -------
    cf : ImageD11 columnfile, must contain sx, sy columns giving peak coordinates in the sample reference frame
    ds : ImageD11 dataset for binning
    
    See also: xyi
    """
    assert all(['sx' in cf.titles, 'sy' in cf.titles]), '(sx,sy) coordinates in sample reference frame have not been computed'
    if y0 is None:
        y0 = 0.5 * (ds.ymax + ds.ymin)
    # x,y bins
    xb, yb = ds.ybinedges, ds.ybinedges
    # xi, yi: pixel coord label for each peak
    xi = np.round(((cf.sx + y0 - ds.ybinedges[0])/ds.ystep)).astype(np.int32)
    yi = np.round(((cf.sy + y0 - ds.ybinedges[0])/ds.ystep)).astype(np.int32)

    cf.addcolumn( xi, 'xi' )
    cf.addcolumn( yi, 'yi')
    xyi = np.array(xi + yi * 10000)  
    cf.addcolumn( xyi.astype(int), 'xyi')   # do not use np.uint32, for some reasons it is 100x slower when running np.searchsorted
    
    
def sorted_xyi_inds(cf):
    """ 
    Runs np.searchsorted on xyi column in cf. Make sure cf is sorted by xyi before. 
    
    Output constains the list of first index positions (inds) of each unique xyi value. 
    e.g: xyi = [0,0,0,1,1,2,3,3,4,4,4] -> inds = [0,3,5,6,8,10]. This allows to quickly find all peaks with
    the same xyi index in cf (ie all peaks from the same pixel), which are between positions inds[i] and inds[i+1] in the
    sorted xyi array.
    
    Args: 
    --------
    cf : imageD11 columnfile, with xyi column
    
    Returns:
    --------
    xyi_uniq (np.array): unique xyi indices in cf.xyi
    inds (np.array): first index position of each unique value xyi_uniq in cf.xyi 
    """
    
    assert 'xyi' in cf.titles, 'xyi has not been computed. Run add_pixel_labels first'
    
    if not cf.sortedby == 'xyi':
        print('sorting peakfile by xyi...')
        cf.sortby('xyi')
    
    xyi_uniq = np.unique(cf.xyi).tolist()
    inds = np.searchsorted(cf.xyi, xyi_uniq)
    inds = np.append(inds, cf.nrows)  
    return xyi_uniq, inds



def pks_inds(sorted_xyi_array, xyi_list, check_list = False):
    """
    find all peaks belonging to a list of pixels, defined by their xyi index. Useful for peak selection over large 
    multi-pixel domains (e.g. grain mask) .
    
    Args:
    -------
    sorted_xyi_array : array of sorted xyi indices in peakfile. (e.g cf.xyi)
    xyi_list : list of xyi indices of pixels to search
    check_list (bool) : check whether list of provided xyi indices is correct (slower). Default is False
    
    Returns:
    ---------
    pks : array of index positions in cf for all peaks in pixel selection
    """
    if check_list:
        xyi_uniq = np.unique(sorted_xyi_array)
        assert all([xyi in xyi_uniq for xyi in xyi_list]), 'some pixels in xyi_list not found in sorted_xyi_array'
    
    return np.concatenate([pks_from_px(sorted_xyi_array, xy0, kernel_size=1, debug=1) for xy0 in xyi_list])



def pks_inds_fast(sorted_xyi_array, xyi_list, check_list = False):
    """
    Find all peaks belonging to a list of pixels, defined by their xyi index.
    Faster than pks_inds. Usefull for peak to grain mapping
    
    Args:
    -------
    sorted_xyi_array : array of sorted xyi indices in peakfile. (e.g cf.xyi)
    xyi_list : list of xyi indices for pixels to search
    check_list : check whether list of provided xyi indices is correct (slower). Default is False
    
    Returns:
    ---------
    pks : array of index positions in cf for all peaks in pixel selection
    """
    
    if check_list:
        xyi_uniq = np.unique(sorted_xyi_array)
        assert all([xyi in xyi_uniq for xyi in xyi_list]), 'some pixels in xyi_list not found in sorted_xyi_array'
    
    # find index of pixels bounding continuous line blocks in x direction -> to feed np.seachsorted    
    px_inds_list = [xyi_list[0]]   #first pixel = first pixel from first block
    
    for i,px in enumerate(xyi_list[:-1]):  # loop through px in list
        if xyi_list[i+1] > xyi_list[i]+1:   # if consecutive index values (px in same block), skip
            px_inds_list.extend([xyi_list[i]+1, xyi_list[i+1]])   # add last pixel from block n and first pixel from block n+1 to list
    
    # add last pixel. 2 cases: 
    # 1 - last px is an independent block -> even nb of values in list, create a new block just for the last px
    # 2 - last px belong to previous block which has not been closed yet -> odd nb of values in list, just add last one to close the last block
    if len(px_inds_list)%2 == 0: 
        px_inds_list.extend([xyi_list[-1], xyi_list[-1]+1])
    else:
        px_inds_list.extend([xyi_list[-1]+1])
    

    pkbounds = np.searchsorted(sorted_xyi_array, px_inds_list)
    
    return np.concatenate([np.arange(lb,ub) for lb,ub in zip(pkbounds[::2],pkbounds[1::2])])
    
        
        
def pks_from_px(sorted_xyi_array, xy0, kernel_size=1, debug=0):
    """ select all peaks from a pixel using xyi indices in peakfile. Allows selection of peaks within a n x n kernel centered on the pixel.
    
    Args:
    ---------
    sorted_xyi_array : array of sorted xyi indices in peakfile. (e.g cf.xyi)
    xy0  (int)       : pixel xyi index
    kernel_size (int) : kernel size for peak selection arround the central pixel. odd integer >=1.
                        1 corresponds to "normal" selection only from the pixel xy0  
    
    Returns: 
    ---------
    pks : array of index positions in cf for all peaks in selection
    """
    # find index positions to pass to np.searchsorted
    xy0 = int(xy0)
    if kernel_size == 1:
        searchsort_inds =  [(xy0,xy0+1)]
        
    if debug:
        print(f'searchsort_inds: {searchsort_inds}')
    
    else:
        n = kernel_size // 2
        xp, yp = xy0%10000, xy0//10000
        searchsort_inds = [ (xi+10000*yi, xi+10000*yi+1) for yi in range(yp-n, yp+n+1) for xi in range(xp-n,xp+n+1) ]
    
    bounds = [np.searchsorted(sorted_xyi_array, inds) for inds in searchsort_inds]  # pks indices boundaries in sorted xyi array
    pks = np.concatenate([np.arange(b[0],b[1]) for b in bounds])             # full pks array
    return pks

    
# Peaks to grain / grain to peaks mapping
###########################################################################
    
def pks_from_grain(cf, g, check_px_inds=False):
    """find peak indices corresponding to a grain g in a peakfile cf
    
    Args:
    ---------
    cf : peakfile sorted by xyi index
    g  : ImageD11 grain. must contain a "xyi_indx" attribute providing the list of xyi indices over which the grain mask extends
    check_px_inds: check whether all xyi indices in g.xyi_indx are present in cf (slow). Default is False

    Returns :
    ---------
    pks: list of peak indices in cf corresponding to grain g"""
    
    assert 'xyi_indx' in dir(g)
    
    if not cf.sortedby == 'xyi':
        cf.sortby('xyi')
    
    return pks_inds_fast(cf.xyi, g.xyi_indx, check_list = check_px_inds)

   

def map_grains_to_cf(glist, cf, overwrite=False):
    """ 
    For each grain a grain list, find corresponding peaks in the peakfile and do grains-to-peakfile / peakfile-to-grains mapping: 
    - add grain_id column to peakfile
    - add peaks index (pksindx) as a new attribute to all grains in the list
    
    Args: 
    --------
    glist : list of ImageD1 grains. Should have xyi_indx property corresponding to the grain mask on the pixel grid
    cf    : ImageD11 columnfile (peakfile), with xyi column
    overwrite : if True, reset grain_id column in peakfile. default if False
    """
        
    if 'grain_id' not in cf.titles or overwrite:
        cf.addcolumn(np.full(cf.nrows, -1, dtype=np.int16), 'grain_id')

    for g in tqdm(glist):
        assert hasattr(g, 'gid'), 'grain missing label'
        assert hasattr(g, 'xyi_indx'), 'grain missing pixel mask (xyi_indx)'

        gid = g.__getattribute__('gid')
        pksindx = pks_from_grain(cf, g, check_px_inds=False)  # get peaks from grain g
        # map grain to cf and pks to grain
        cf.grain_id[pksindx] = gid
        g.pksindx = pksindx
                
    print('completed')  


# cluster g-vectors by hkl
###########################################################################
def select_by_hkl_family(ubi, gvecs, hkl_tol = 0.1, phase=None, hkl=(0,0,1), symmetrise=False):
    """
    select g-vectors by closest hkl integer indices. return mask for g-vector selection
    parameters:
    ------------
    ubi (3x3 array)    : lattice vectors matrix
    gvecs ((n,3) array : reciprocal lattice vectors (gx,gy,gz)
    hkl_tol (float)    : hkl tolerance for g-vector filtering
    hkl (3-tuple)      : (h,k,l) integer indices to select
    phase (orix.Phase) : orix phase obj (optional). required if symmetrise option is chosen
    symmetrise (bool)  : if True, return all g-vecs corresponding to the symmetry-equivalent {hkl} family

    returns:
    hklmask (bool array) : selection mask for corresponding g-vectors
    """
    # filter gvecs
    gvecs, _, _ = refine_loop(ubi, gvecs, np.full(gvecs.shape, True), hkl_tol)
    hkls = np.round( ubi.dot( gvecs.T ) ).T
    
    # find all hkls in familly
    if phase is None:
        phase = Phase(space_group=1)
    if symmetrise:
        miller = ovec.Miller(hkl=hkl, phase=phase).symmetrise(unique=True)
    else:
        miller = ovec.Miller(hkl=hkl, phase=phase)
    hklgroup = np.round(miller.hkl,2).astype(int)
    print(hkls.shape)
    # mask
    hklmask = np.full(len(hkls),False, dtype=bool)
    hkls = hkls.astype(int)
    for m in hklgroup:
        mhkl = np.all([hkls[:,0] == m[0], hkls[:,1] == m[1], hkls[:,2] == m[2]], axis=0)
        hklmask += mhkl
    return hklmask

    
    
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
 

               
# grain /pixel ubi refinement: refine lattice vectors matrix using the whole set of peaks assigned to the grain/pixel
# to merge with refine_ubi
###########################################################################
    
@njit
def refine_loop(ubi, gvecs, gvmask, hkl_tol):
    """Fast numeric part of the refinement."""
    hkl = ubi @ gvecs.T
    hkli = np.round(hkl)
    drlv = hkli - hkl
    drlv2 = (drlv * drlv).sum(axis=0)
    ret = drlv2 < hkl_tol * hkl_tol
    gvecs = gvecs[ret]
    gvmask = gvmask[ret]
    return gvecs, gvmask, drlv2


def refine_px_ubi(cf, px, UBI, U, hkl_tol=0.1, sym = None, kernel_size=1):
    """ 
    refine lattice vector matrix (ubi) excluding dodgy g-vectors (drlv*drlv > hkl_tol) for a selected pixel.
    g-vectors are weighted by intensity and merged by unique hkl for better orientation fitting.
    returns error (mean drlv squared) and completeness (proportion of indexed intensity) metrics 
    
    Args:
    -------
    cf      : ImageD11 columnfile sorted by xyi indices
    px      : pixel index in xmap (xyi index)
    hkl_tol : hkl tolerance for g-vector filtering
    sym     : crystal symmetry (orix.quaternion.symmetry.Symmetry object). oiptional, used to evaluate rotation between old and new orientation. 
    kernel_size : n-by-n kernel size around central pixel for peak selection. default is 1. 
    UBI/U : (3x3 mat) UBI/U lattice vector matrices / rotation matrices for selected pixel
    
    Returns:
    --------
    ubi     : refined ubi matrix
    gvecs   : g-vectors retained
    gvmask  : boolean mask over cf corresponding to retained g-vectors
    hkli    : integer hkl indices of retained g-vectors
    stats   : statistics: mean drlv2, n indexed, completeness, angle shift (degree) between old and new orientation
    """
    # select peaks from cf
    gvmask = pks_from_px(cf.xyi, px, kernel_size=kernel_size)
    gvecs = np.array([cf.gx[gvmask], cf.gy[gvmask], cf.gz[gvmask]]).T
    if 'norm_intensity' in cf.titles:
        intensities = cf.norm_intensity
    else:
        intensities = cf.sum_intensity
        
    U0 = copy.deepcopy(U)
    Itot_0 = np.sum(intensities[gvmask])   # total intensity over the selection. for completeness estimation

    default_stats = {'mean drlv2':np.nan, 'nindx':0, 'completeness':0, 'angle dev (degree)': np.nan}
        
    # check ubi is correct
    try:
        xfab.tools.ubi_to_u(UBI)
    except ValueError as e:
        print(f'px {px}: {e}')
        return UBI, U, [], gvmask, [], default_stats
        
    # refine ubis
    for i in range(1):
        # filter g-vectors and update mask
        gvecs, gvmask, _ = refine_loop(UBI, gvecs, gvmask, hkl_tol)
        if len(gvmask) == 0:
            return  UBI, U, [], gvmask, [], default_stats   
        ints = intensities[gvmask]

        # merge g-vectors by unique hkl
        gvecs_merged, hkl_uniqs, Isum, _ = cluster_by_hkl(gvecs, ints, UBI)
        # update ubi with merged g-vectors
        nindx, drlv2 = ImageD11.cImageD11.score_and_refine(UBI, np.ascontiguousarray(gvecs_merged), tol=1)  # set large hkltol to take all peaks in selection
        
    # recompute integer hkl index for retained peak
    hkli = np.round(np.dot(UBI, gvecs.T))
    completeness = Isum.sum() / Itot_0
    
    # compute rotation angle between former and new ubi + proportion of peaks retained
    try:
        U = xfab.tools.ubi_to_u(UBI)
    except ValueError as e:
        print(f'px {px}: {e}')
        return UBI, U, [], gvmask, [], default_stats

    if sym is not None:
        o = oq.Orientation.from_matrix(U0, symmetry =sym)  # old orientation
        o2 = oq.Orientation.from_matrix(U, symmetry = sym) # new orientation 
        ang_dev =  o2.angle_with(o, degrees=True)[0]
    else:
        ang_dev = np.nan
    
    stats = {'mean drlv2':drlv2, 'nindx':nindx, 'completeness':completeness, 'angle dev (degree)': ang_dev}
    
    return UBI, U, gvecs, gvmask, hkli, stats


def refine_px_ubi_fast(gvecs, intensities, UBI, hkl_tol=0.1):
    """ 
    same as refine_px_ubi but takes directly g vectors as input and computes orientation shift directly from matrices (without considering crystal symmetry)
    aimed to be used in local_indexing script
    """
    default_output = np.zeros((3,3)), np.nan, np.nan, np.nan, np.nan
    Itot_0 = np.sum(intensities)
    # check ubi is correct
    try:
        U0 = xfab.tools.ubi_to_u(UBI)
    except ValueError as e:
        return default_output  
        
    # refine ubis: there were initially two iterations, but does not seem to make a difference. 
    for i in range(1):   
        #  first refinement using all g-vectors
        gvecs, ints, _ = refine_loop(UBI, gvecs, intensities, hkl_tol)
        if len(gvecs) == 0:
            return default_output

        # merge g-vectors by unique hkl and re-do refinement: better orientation fit
        gvecs_merged, hkl_uniqs, Isum, Imean = cluster_by_hkl(gvecs, ints, UBI)
        nindx, drlv2 = ImageD11.cImageD11.score_and_refine(UBI, np.ascontiguousarray(gvecs_merged), tol=1)  # set large hkltol to take all peaks in selection

    # compute completeness score and orientation shift between former and new ubi
    completeness = np.sum(Isum) / Itot_0  
    
    try:
        U = xfab.tools.ubi_to_u(UBI)
    except ValueError as e:
        return default_output

    rot = R.from_matrix(U @ U0.T)
    angle_shift = rot.magnitude() * 180/np.pi
    
    return UBI, nindx, drlv2, completeness, angle_shift


def refine_px_ubi_fast_2(gvecs, intensities, UBI, cs, hkl_tol=0.1):
    """ 
    same as refine_px_ubi but takes directly g vectors as input and computes orientation shift directly from matrices (without considering crystal symmetry)
    aimed to be used in local_indexing script
    """
    default_output = np.zeros((3,3)), np.nan, np.nan, np.nan, np.nan
    Itot_0 = np.sum(intensities)
    # check ubi is correct
    try:
        U0 = xfab.tools.ubi_to_u(UBI)
    except ValueError as e:
        return default_output  
        
    # refine ubis: there were initially two iterations, but does not seem to make a difference. 
    for i in range(1):   
        #  first refinements using all g-vectors
        gvecs, ints, _ = refine_loop(UBI, gvecs, intensities, hkl_tol)
        if len(gvecs) == 0:
            return default_output

        # merge g-vectors by unique hkl and re-do refinement: better orientation fit
        gvecs_merged, hkl_uniqs, Isum, Imean = cluster_by_hkl(gvecs, ints, UBI)
        nindx, drlv2 = ImageD11.cImageD11.score_and_refine(UBI, np.ascontiguousarray(gvecs_merged), tol=1)  # set large hkltol to take all peaks in selection

    # compute completeness score and Instensity score
    completeness = np.sum(Isum) / Itot_0 
    
    sF = cs.str_dans.Scatter.new_structure_factor(hkl_uniqs.T)  # structure factors
    Icalc = np.absolute(sF) # intensity
    Iscore = np.corrcoef(Imean, Icalc)[0,1]
    
    try:
        U = xfab.tools.ubi_to_u(UBI)
    except ValueError as e:
        return default_output
    
    return UBI, nindx, drlv2, completeness, Iscore



    
def refine_grains(glist, cf, hkl_tol, intensities, nmedian= np.inf, sym = None, return_stats=True):
    """ 
    Refine peaks_to_grain assignement and fit unit cell matrix for all grains in glist.
    - dodgy peaks are removed (drlv*drlv > hkltol and abs(median err) > nmedian
    - peaks to grain labeling (g.pksindx) updated
    - g-vectors merged by unique hkl and weighted by intensity before fitting
    """
    
    stats = {'completeness':[], 'nindx':[], 'angle deviation':[], 'mean drlv2':[]}
    completeness, ang_dev = [], []
    if intensities is None:
        intensities = cf.sum_intensity
    
    for g in tqdm(glist):
        assert 'pksindx' in dir(g), 'grain has not attribute "pksindx"'

        gvecs = np.transpose([cf.gx[g.pksindx], cf.gy[g.pksindx], cf.gz[g.pksindx]]).copy()
        ints = intensities[g.pksindx]
        N0 = len(gvecs)  # initial peak number
        Itot_0 = np.sum(ints)
        ubi0, u0 = g.ubi.copy(), g.U.copy() # keep a copy of old ubi + u mats

        # refine ubis
        #############
        for _ in range(1):
            # compute hkl and drlv2 for each peak and remove outliers
            gvecs, g.pksindx, _ = refine_loop(g.ubi, gvecs, g.pksindx, hkl_tol)
            update_mask(g, cf, cf.parameters, nmedian)
            gvecs_merged, hkl_uniqs, Isum = cluster_by_hkl(gvecs, ints, g.ubi)

            #fit orientation with clean peaks only
            nindx,drlv2 = ImageD11.cImageD11.score_and_refine(g.ubi, gvecs_merged, tol=1)  # set large hkltol to take all peaks in selection
            g.set_ubi(g.ubi)
        
        g.TotalIntensity = Isum.sum()
        completeness = g.TotalIntensity / Itot_0

        # compute rotation angle between former and new ubi + prop of peaks retained
        o1 = oq.Orientation.from_matrix(u0, symmetry =sym)  # old orientation
        o2 = oq.Orientation.from_matrix( g.U, symmetry = sym) # new orientation 
        
        stats['angle deviation'].append( o2.angle_with(o1, degrees=True)[0] )
        stats['completeness'].append(completeness)
        stats['mean drlv2'].append(drlv2)
        stats['nindx'].append(nindx)
        
    if return_stats:
        return stats
    
    

def update_mask( g, cf, pars, nmedian ):
    """
    Remove nmedian*median_error outliers from grains assigned peaks. Modified from s3dxrd.peak_mapper 
    (https://github.com/FABLE-3DXRD/scanning-xray-diffraction)
    """
    # obs data for this grain
    tthobs = cf.tth[g.pksindx]
    etaobs = cf.eta[g.pksindx]
    omegaobs = cf.omega[g.pksindx]
    gobs = np.array( (cf.gx[g.pksindx], cf.gy[g.pksindx], cf.gz[g.pksindx]) )
    # hkls for these peaks
    hklr = np.dot( g.ubi, gobs )
    hkl  = np.round( hklr )
    # Now get the computed tth, eta, omega
    etasigns = np.sign( etaobs )
    g.hkl = hkl.astype(int)
    g.etasigns = etasigns
    ub = np.linalg.inv(g.ubi)
    tthcalc, etacalc, omegacalc = calc_tth_eta_omega( ub, hkl, pars, etasigns )
    # update mask on outliers
    dtth = (tthcalc - tthobs)
    deta = (etacalc - etaobs)
    domega = (omegacalc%360 - omegaobs%360)
    ret  = abs( dtth ) <= np.median( abs( dtth   ) ) * nmedian
    ret &= abs( deta ) <= np.median( abs( deta   ) ) * nmedian
    ret &= abs( domega)<= np.median( abs( domega ) ) * nmedian
    g.pksindx = g.pksindx[ret]
    g.hkl = g.hkl[:,ret]
    return 

               

def calc_tth_eta_omega( ub, hkls, pars, etasigns):
    """
    Predict the tth, eta, omega for each grain. Copied from s3dxrd.peak_mapper (https://github.com/FABLE-3DXRD/scanning-xray-diffraction)
    ub = ub matrix (inverse ubi)
    hkls = peaks to predict
    pars = diffractometer info (wavelength, rotation axis)
    etasigns = which solution for omega/eta to choose (+y or -y)
    """
    gvecs = np.dot(ub, hkls)

    tthcalc, eta2, omega2 = ImageD11.transform.uncompute_g_vectors(gvecs,  pars.get('wavelength'),
                                                            wedge=pars.get('wedge'),
                                                            chi=pars.get('chi'))
    # choose which solution (eta+ or eta-)
    e0 = np.sign(eta2[0]) == etasigns
    etacalc = np.where(e0, eta2[0], eta2[1])
    omegacalc = np.where(e0, omega2[0], omega2[1])
    return tthcalc, etacalc, omegacalc   
