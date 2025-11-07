import os, sys, copy
import numpy as np
import pylab as pl
from tqdm import tqdm

import ImageD11.columnfile, ImageD11.grain, ImageD11.refinegrains, ImageD11.cImageD11
import xfab
from orix import quaternion as oq

from pf_3dxrd import utils, crystal_structure, pixelmap



""" 
Peakfile-to-pixelmap / peakfile-to-grainmap mapping. 
Contains function to find peaks corresponding to a pixel / grain mask on a 2D map and to assign grain labels / pixel labels to 
peaks in a peakfile. 

Also includes function for grain UBI refinment : refine grain unit cell matrix using all peaks corresponding to this grain
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


def add_pixel_labels(cf, ds):
    """
    Compute pixel coordinates (xi,yi) and pixel index (xyi) for each peak in a peakfile cf, using (xs,ys) coordinates in sample space. 
    Adds new columns xi, yi and xyi to the peakfile
    
    Args:
    -------
    cf : ImageD11 columnfile, must contain xs, ys columns giving peak coordinates in the sample reference frame
    ds : ImageD11 dataset for binning
    
    See also: xyi
    """
    assert all(['xs' in cf.titles, 'ys' in cf.titles]), '(xs,ys) coordinates in sample reference frame have not been computed'
    
    # x,y bins
    xb, yb = ds.ybinedges, ds.ybinedges
    # xi, yi: pixel coord label for each peak
    xi = np.round(((cf.xs - ds.ybinedges[0])/ds.ystep)).astype(np.uint32)
    yi = np.round(((cf.ys - ds.ybinedges[0])/ds.ystep)).astype(np.uint32)

    cf.addcolumn( xi, 'xi' )
    cf.addcolumn( yi, 'yi')
    xyi = np.array(xi + yi * 10000)  
    cf.addcolumn( xyi.astype(int), 'xyi')   # do not use np.uint32, for some reasons it is 100x slower when running np.searchsorted
    cf.sortby('xyi')
    
    
def sorted_xyi_inds(cf, is_sorted=False):
    """ 
    Runs np.searchsorted on xyi column in cf. Make sure cf is sorted by xyi before. 
    
    Output constains the list of first index positions (inds) of each unique xyi value. 
    e.g: xyi = [0,0,0,1,1,2,3,3,4,4,4] -> inds = [0,3,5,6,8,10]. This allows to quickly find all peaks with
    the same xyi index in cf (ie all peaks from the same pixel), which are between positions inds[i] and inds[i+1] in the
    sorted xyi array.
    
    Args: 
    --------
    cf : imageD11 columnfile, with xyi column
    is_sorted (bool) : indicates whether cf has been sorted by xyi indices (required). Default is False
    
    Returns:
    --------
    xyi_uniq (np.array): unique xyi indices in cf.xyi
    inds (np.array): first index position of each unique value xyi_uniq in cf.xyi 
    """
    
    assert 'xyi' in cf.titles, 'xyi has not been computed. Run add_pixel_labels first'
    
    if not is_sorted:
        cf.sortby('xyi')
    
    xyi_uniq = np.unique(cf.xyi).tolist()
    inds = np.searchsorted(cf.xyi, xyi_uniq)
    inds = np.append(inds, cf.nrows)  
    return xyi_uniq, inds



def pks_inds(sorted_xyi_array, xyi_list, check_list = False):
    """
    find all peaks belonging to a list of pixels, defined by their xyi index. Useful for peak selection over a mask 
    covering multiple pixels.
    
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
    """ select all peaks from a pixel using xyi indices in cf. Allows selection of peaks within a n x n kernel centered on the pixel.
    
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
    
def pks_from_grain(cf, g, is_cf_sorted = False, check_px_inds=False):
    """find peak indices corresponding to a grain g in a peakfile cf
    
    Args:
    ---------
    cf : peakfile sorted by xyi index
    g  : ImageD11 grain. must contain a "xyi_indx" attribute providing the list of xyi indices over which the grain mask extends
    is_cf_sorted : bool flag indicating whether cf has been sorted by xyi (required for np.searchsorted)
    check_px_inds: check whether all xyi indices in g.xyi_indx are present in cf (slow). Default is False


    Returns :
    ---------
    pks: list of peak indices in cf corresponding to grain g"""
    
    assert 'xyi_indx' in dir(g)
    
    if not is_cf_sorted:
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
        
        pksindx = pks_from_grain(cf, g, is_cf_sorted = True, check_px_inds=False)  # get peaks from grain g
        
        # map grain to cf and pks to grain
        cf.grain_id[pksindx] = gid
        g.pksindx = pksindx
                
    print('completed')  
 

               
# grain refinement: refine lattice vectors matrix using the whole set of peaks assigned to he grain
###########################################################################
               
    
def refine_grains(glist, cf, hkl_tol, nmedian= np.inf, sym = None, return_stats=True):
    """ Refine peaks_to_grain assignement and fit unit cell matrix for all grains in glist
    
    - dodgy peaks are removed (drlv*drlv > hkl_tol)
    - fit outliers are removed abs(median err) > nmedian
    - peaks to grain labeling (g.pksindx) updated
    
    Peaks selection using g.pksindx. If no attribute "pksindx" is found for the grain, run function "map_grain_to_cf" in Pixelmap
    
    Args:
    -------
    glist : list of ImageD11 grains to be refined
    cf : ImageD11 columnfile sorted by xyi indices
    hkl_tol : hkl tolerance for peaks
    nmedian : threshold to remove outliers ( abs(median err) > nmedian ). Default is inf: no outliers removed
    sym : crystal symmetry (orix.quaternion.symmetry.Symmetry object). used to evaluate misorientation between old and new orientation. 
    return_stats: returns list of rotation (angle between old and new crystal orientation) + fraction of peaks retained. Default is True
    """

    prop_indx, ang_dev = [], []
    
    for g in tqdm(glist):
        assert 'pksindx' in dir(g), 'grain has not attribute "pksindx"'
    
        gv = np.transpose([cf.gx[g.pksindx], cf.gy[g.pksindx], cf.gz[g.pksindx]]).copy() 
        N0 = len(gv)  # initial peak number
        ubi = g.ubi.copy() # keep a copy of old ubi
        
        # refine grain ubis
        for i in range(3):
            # compute hkl and drlv for each peak
            hkl = np.dot(g.ubi, gv.T)
            hkli = np.round( hkl )
            # Error on these:
            drlv = hkli - hkl
            drlv2 = (drlv*drlv).sum(axis=0)
    
            # filter out dodgy peaks
            ret = drlv2 < hkl_tol*hkl_tol
            g.pksindx = g.pksindx[ret]
    
            #remove outliers
            update_mask(g, cf, cf.parameters, nmedian)
    
            #fit orientation with clean peaks only
            gv = np.transpose([cf.gx[g.pksindx], cf.gy[g.pksindx], cf.gz[g.pksindx]])
            ImageD11.cImageD11.score_and_refine(g.ubi, gv, tol=1)  # set large hkltol to take all peaks in g.pksindx
            
    
        # compute rotation angle between former and new ubi + prop of peaks retained
        o = oq.Orientation.from_matrix(g.U, symmetry =sym)  # old orientation
        o2 = oq.Orientation.from_matrix( xfab.tools.ubi_to_u(ubi), symmetry = sym) # new orientation 
        
        ang_dev.append( o2.angle_with(o, degrees=True)[0] )
        prop_indx.append( len(g.pksindx) / N0)
        
        
    if return_stats:
        return prop_indx, ang_dev
    

    
def update_mask( g, cf, pars, nmedian ):
    """
    Remove nmedian*median_error outliers from grains assigned peaks. Modified from s3dxrd.peak_mapper 
    (https://github.com/FABLE-3DXRD/scanning-xray-diffraction)
    """
    # obs data for this grain
    tthobs = cf.tthc[g.pksindx]
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
