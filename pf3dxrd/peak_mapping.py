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


     