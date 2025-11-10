""" This module contains functions to identify symmetric (h,k,l), (-h,-k,-l) reflections (aka. Friedel pairs) in a 3DXRD dataset. It works the best with scanning-3DXRD data acquired using a pencil-beam scanning procedure, but seems to provide also good results with regular 3DXRD scans using a box beam or a letter-box beam. 

For more details, see related publication [add it when published]
"""


import os, sys
import numpy as np
import pylab as pl
from tqdm import tqdm

import scipy.spatial
from scipy.sparse import csr_matrix

from ImageD11 import columnfile, transform
from pf3dxrd.pf3dxrd import utils



# pre-processing: form pairs of y-scans (-dty,+dty), which should contain symmetric information
##############################################################

def form_y_pairs(cf, ds, disp=True):
    """ form pairs of symmetric scans (-dty,+dty) and return them as a list of tuples.
    Central scan (dty = 0) is paired with itself. Also checks that central dty scan is
    at position dty = 0 in ds. If not, ymin, ymax and dty in ds are updated and ybins are
    recomputed
    
    Args: 
    --------
    cf          : ImageD11 columnfile. Must contain dty column
    ds          : ImageD11.sinograms.dataset.Dataset class object. contains info on y-bins and omega-bins
    disp (bool) : display y-pairs
    """
    
    # check dty bins are centered on zero
    
    central_bin = len(ds.ybincens)//2
    c0 = ds.ybincens[central_bin]
    
    print('dty_center = ', '%.2f'%c0, ' on bin n°', central_bin)
    
    if abs(c0) > 0.01 * ds.ystep:
        print('dty not centered on zero. Updating dty scan positions in dataset and peakfile...')
        shift  = (ds.ymax + ds.ymin) / 2
        print('shift=', '%.4f'%shift)
        
        ds.dty = ds.dty - shift
        cf.dty = cf.dty - shift
        ds.guessbins()
    
    # form y-pairs
    hi_side = ds.ybincens[central_bin:].tolist()
    lo_side = ds.ybincens[:central_bin+1].tolist()
    
    hi_side.sort()
    lo_side.sort(reverse=True)
    
    ypairs = [(y1,y2) for (y1,y2) in zip(hi_side, lo_side)]
    ds.ypairs = ypairs
    
    if disp:
        print('dty pairs: \n ===========')
        for yp in ypairs:
            print('%.4f'%yp[0], ';', '%.4f'%yp[1])
            
            
def check_y_symmetry(cf, ds, saveplot=False, fname_plot=None):
    """
    check that paired scans contain about the same number of peaks + total peak intensity.
    For each pair, plots total intensity + total nb of peaks in dty scan vs. |dty|. 
    If peak nb and total intensity in dty and -dty subsets do not match, it is likely that something
    went wrong in the acquisiton. either the sample moved, or it is not correclty centered, or there
    as a problem with the beam. 
    
    Args:
    ---------
    cf : ImageD11 columnfile. Must contain dty column
    ds : ImageD11.sinograms.dataset.Dataset object. contains metadata
    saveplot: save output
    fname_plot: name for the saved output. default is ds.dsname+'_dty_alignment.png'
    """
    
    Nrows, Sum_I = [],[]
    
    if not hasattr(ds, 'ypairs'):
        form_y_pairs(cf, ds, disp=False)
    
    # order cf by dty scans to quickly find peaks from each scan
    print('sorting peakfile by dty scans...')
    cf.sortby('dty')
    inds = np.searchsorted(cf.dty, ds.ybinedges)
    
    # compute sum of intensity and Npeaks for each scan
    print('computing stats for each scan...')
    for i,p in tqdm(enumerate(ds.ypairs)):
        sum_I_hi = cf.sum_intensity[inds[-(i+2)]:inds[-(i+1)]]
        sum_I_lo = cf.sum_intensity[inds[i]:inds[i+1]]
        Sum_I.append((np.sum(sum_I_hi) , np.sum(sum_I_lo)) )
        Nrows.append((len(sum_I_hi), len(sum_I_lo)))
    
    central_bin = len(ds.ybincens)//2
    hi_side = np.flip(ds.ybincens[central_bin:])
    
    # make plots
    ydata = [np.log10(Sum_I), Nrows, [np.divide(p[1],p[0]) for p in np.log10(Sum_I)], [np.divide(p[1],p[0]) for p in Nrows] ]
    ylabel = ['log I','N peaks','log I ratio','N peaks ratio']
    legend = [['dty','-dty'],['dty','-dty'],'-dty/dty ratio', '-dty/dty ratio']
    
    f = pl.figure(figsize=(8,8), layout='constrained')
    for i, (dat,lab,leg) in enumerate(zip(ydata,ylabel,legend)):
        f.add_subplot(2,2,i+1)
        pl.plot(hi_side, dat, '.', label=leg)
        pl.xlabel('|dty|')
        pl.ylabel(lab)
        pl.legend()
    
    f.suptitle('dty alignment – '+str(ds.dsname))
    
    if saveplot is True:
        if fname_plot is None:
            fname_plot = os.path.join(os.getcwd(), ds.dsname, ds.dsname+'_dty_alignment.png')
        f.savefig(fname_plot, format='png')
        
        
def select_y_pair(cf, ds, pair_id, is_sorted=False):
    """ select peaks from two symmetric scans and return them as two separate columnfiles.
    
    Args : 
    ---------
    cf     : ImageD11 columnfile with complete data (all dty scans)
    ds     : ImageD11.sinograms.dataset.Dataset object
    pair_id: index of ypair to select in ds.ypairs
    is_sorted: indicate whether peakfile has already been sorted by dty, so it does not need to be done again.
    
    Outputs:
    ---------
    c1, c2 : symmetric columnfiles containing respectively peaks from the hi-side (+dty) and the lo-side (-dty) of the selected ypair
    """
    # order cf by dty scans to quickly find peaks from each scan
    if not is_sorted:
        print('sorting peakfile by dty scans...')
        cf.sortby('dty')
    inds = np.searchsorted(cf.dty, ds.ybinedges)
    bc = len(ds.ybincens)//2  # central bin
    
    inds_lo = np.arange(inds[bc-pair_id],inds[bc-pair_id+1])
    inds_hi = np.arange(inds[bc+pair_id],inds[bc+pair_id+1])
    
    c1 = columnfile.colfile_from_dict({t:cf.getcolumn(t)[inds_lo] for t in cf.titles})
    c2 = columnfile.colfile_from_dict({t:cf.getcolumn(t)[inds_hi] for t in cf.titles})
    
    return c1, c2


    

# functions for pair matching
##############################################################


def search_space(cf, mask, mtth = 5, mI = 1/5, flip_eta_omega=False):
    """
    build rescaled 4D search space in (eta*, omega*, tth*, I*) in which friedel pairs will be searched for. 
    
    coordinates are defined as follows:
    eta*   = eta mod 360
    omega* = omega mod 360
    I*     = log(I) * mI
    tth*   = ln(tan_tth) * mtth
    
    Intensity and 2-theta are re-scaled so that spread along these dimensions is comparable to the spread in eta and omega. 
    For intensity, logarithmic scaling is used because peak intensity spans
    several orders of magnitude and therefore significant intensity variations are  
    better represented on a logarithmic scale than on a linear scale.
    
    For two-theta, the need for rescaling is related to the fact that 2-theta variations related to
    geometrical offset dx increases with the scattering angle: Δtan(2θ) ∝ dx.tan(2θ).
    In contrast, the difference in logarithm of tan(2θ) only depends on the geometrical offset dx: Δln(tan(2θ)) ∝ dx. 
    
    Args:
    ----------
    cf   : input columnfile
    mask : boolean mask of len cf.nrows to select a subset in cf

I don'

    mtth : multiplication factor for tth*: defaut is 5.
    mI   : multiplication factor for log I. default is 1/5. 
    
    flip_eta_omega (Bool). If True, eta and omega coordinate are flipped: 
    eta -> (180-eta) mod 360 ; omega -> (180+omega) mod 360. Needed to match friedel pairs in symmetrical scans.

    Note : multiplication factors mtth and mI are introduced so that the distance between two peaks in a friedel pair
    scales approximately the same along all dimensions (eta*, omega*, I*, tth*). Default values should be about right,
    but I guess it can depend on the experiment (range of peak intensities, max 2theta), so these may need to be adjusted a bit
    
    Note 2: Friedel pair that match in eta and omega tends to be better in (eta,omega) than in tth, intensity. Sometimes, adding these 
    two dimensions in the search space seems to just bring additional noise without much gain on pairig quality. In this case, 
    just use very low values for mtth and mI to "mute" these coordinates. 
    
    """
    if mask is None:
        mask = np.full(cf.nrows, True)
    
    # eta - omega
    if flip_eta_omega:
        eta_n = (180 - cf.eta[mask])%360 
        omega_n = (180+cf.omega[mask])%360
    else:
        eta_n = cf.eta[mask]%360 
        omega_n = cf.omega[mask]%360
    # sum_intensity
    logI = mI * np.log10(cf.sum_intensity[mask])
    # two-theta
    logtth = mtth * np.log(np.tan(np.radians(cf.tth[mask])))   
    
    return eta_n, omega_n, logI, logtth
    

        
def search_space_normalized(cf, mask, flip_eta_omega=False):
    """
    Same as searchspace function above but coordinates are normalized to [0,1]. NOT RECOMMENDED!! (see note below)
    
    eta_n   = (eta/360) mod 1
    omega_n = (omega/360) mod 1
    I_n     = (log(I) - log(I).min) / (log(I).max - log(I).min)
    tth_n   = (ln(tan_tth) - ln(tan_tth).min / (ln(tan_tth).min - ln(tan_tth).max)    where tan_tth = tan(tth)
    
    Args:
    ----------
    cf   : input columnfile
    mask : boolean mask of len cf.nrows to select a subset in cf
    flip_eta_omega (Bool).  If True, eta and omega coordinate are flipped
    
    See also: search_space
    
    Note: Using the normalized search space yields weird results. I suspect it gives too much importance to tth and Intensity.
    Hence, unrealistic pair matchs are found between peaks very far from each other (>>1°) in eta and omega. 
    """
    if mask is None:
        mask = np.full(cf.nrows, True)
    
    # eta - omega
    if flip_eta_omega:
        eta_n = ((180 - cf.eta[mask])/360) % 1
        omega_n = ((180+cf.omega[mask])/360) % 1
    else:
        eta_n = (cf.eta[mask]/360) % 1
        omega_n = (cf.omega[mask]/360) % 1
    # sum_intensity
    logI = np.log10(cf.sum_intensity[mask])
    I_n = (logI - logI.min()) / (logI.max() - logI.min())
    # two-theta
    logtth = np.log(np.tan(np.radians(cf.tth[mask])))
    tth_n = (logtth - logtth.min()) / (logtth.max() - logtth.min())
    
    return eta_n, omega_n, tth_n, I_n



def compute_csr_dist_mat(c1, c2, dist_cutoff=1., mtth=5, mI=1/5):
    """
    Core function for friedel pair matching. 
    
    This function uses the KDTree algorithm from scipy.spatial to construct a distance matrix between peaks
    from c1 (+dty) and c2 (-dty). All values above a given distance threshold (dist_cutoff) are reset to zero,
    so the output is a sparse matrix containing mostly zeros, except when two peaks are close to each other.
    
    This sparse matrix is then cleaned to keep only one non-zero value per row and column, so that each peak
    in c1 is associated to at most one single peak in c2, and conversely (see also:  clean_csr_dist_mat). 
    
    Peak distance is computed in a 4D-space defined using (2-theta, eta, omega, sum_intensity). see also: search_space

    Args:
    --------
    c1, c2        : pair of ImageD11 columnfiles 
    dist_cutoff   : distance threshold for csr_matrix. All values above are reset to zero
    mtth : scaling factor for 2-theta
    mI   : scaling factor for sum_intensity
    
    Outputs:
    ---------
    dij (csr mat) : sparse distance matrix between peaks from c1 and c2. shape (c1.nrows,c2.nrows)
    """
    
    # mask to select non-paired data. fp_id contains the labels for friedel_pairs. It is initialized to -1, and then updated iteratively
    msk1 = c1.fp_id == -1
    msk2 = c2.fp_id == -1
    
    # form KDTrees and compute distance matrix
    g1 = np.transpose(search_space(c1, msk1, mtth, mI, flip_eta_omega=False))
    a = scipy.spatial.cKDTree( g1 )
    g2 = np.transpose(search_space(c2, msk2, mtth, mI, flip_eta_omega=True))
    b = scipy.spatial.cKDTree( g2 )
    
    dij = csr_matrix( a.sparse_distance_matrix( b, dist_cutoff ) )
                      
    return dij    


                      
def clean_csr_dist_mat(dij_csr, verbose=True):
    """ clean the csr distance matrix to avoid pairing a peak from c1 with multiple peaks from c2 and conversely. 
    Keep only the minimal non-zero value in each row and column.
    
    Args:
    ---------
    dij_csr (csr mat) : scipy sparse distance matrix of shape M*N, where M = c1.nrows ad M = c2.nrows
    verbose (bool)    : print some information about pairing
    
    Outputs:
    ----------
    dij_best.data   : distance for selected friedel pairs
    c1_indx, c2_indx: indices of paired peaks in c1[msk1] and c2[msk2] (msk1 and msk2 defined in compute_csr_dist_mat).
    """
    
    n_pairs_all = min(dij_csr.shape)  # number of non-zero elements in the sparse matrix, ie number of possible pairs      
    
    # We want minimum non zero values in each row and column. However, there seems to be no efficient way to find minimum non-zero values in the matrix
    # So here we first do element-wise inversion of non-zero data and find the position of maximum values using np.argmax()
    
    # computes the inverse of non-zero elements in the dij_csr.data.
    dij_best = dij_csr.copy()
    dij_best.data = np.divide(np.ones_like(dij_csr.data), dij_csr.data, out=np.zeros_like(dij_csr.data), where=dij_csr.data!=0)
    
    # Find max values and row index in each column, ie: find the closest match in c2 for data in c1
    maxvals  = np.max(dij_best, axis=0).toarray()[0]
    row_indx = np.argmax(dij_best, axis=0).A[0]     # m.A[0] to convert matrix to 1D numpy array
    col_indx = np.arange(len(row_indx))
    # update csr matrix to keep only maxvals
    dij_best = csr_matrix((maxvals, (row_indx, col_indx)), shape = dij_best.shape )
    
    # do the same as above, but working on rows, ie: find the closest match in c1 for data in c2
    maxvals = np.max(dij_best, axis=1).toarray()[:,0]
    col_indx = np.argmax(dij_best, axis=1).A[:,0]    
    row_indx = np.arange(len(col_indx))
    dij_best = csr_matrix((maxvals, (row_indx, col_indx)), shape = dij_best.shape )
    
    # drop all zero values
    dij_best.eliminate_zeros()  
    
    # inverse data to get back to initial distance
    dij_best.data = np.divide(np.ones_like(dij_best.data), dij_best.data, out=np.zeros_like(dij_best.data), where=dij_best.data!=0)
    
    n_pairs_cleaned = dij_best.nnz   # number of pairs eventually retained
    
    if verbose:
        print(n_pairs_cleaned, 'pairs kept out of ', n_pairs_all, ' possible matches')
    
    c1_indx, c2_indx = dij_best.nonzero()  # friedel pairs indices in c1 and c2 
    
    return dij_best.data, c1_indx, c2_indx
  


def label_friedel_pairs(c1, c2, dist_max=1., dist_step=0.1, mtth = 5, mI=1/5, verbose=True, doplot=False):
    """
     Find Friedel pairs in symmetric columnfiles c1 and c2. It is organized in different blocks:
    
    INITIALIZATION 
    - initialize friedel pairs labels (fp_id) and friedel pair distance (fp_dist) columns in c1 and c2 (init values = -1)
    - initialize lists to store all friedel pair labels (useful in the following)
    
    FRIEDEL PAIR SEARCH LOOP
     - Run 'compute_csr_dist_mat' and 'clean_csr_dist mat -> returns a list of indices in c1 and c2 corresponding to friedel pairs
     - Update the labels fp_id and fp_dist in c1 and c2 with newly identified pairs
     - Iterate the procedure above on remaining non-paired peaks using a larger dist_cutoff threshold, and continue
     these iterations until maximum distance threshold is reached.
     
     dist_cutoff increase and max distance are controled with the two parameters dist_step and dist_max: 
     search_steps = np.arange(dist_step, dist_max+dist_step, dist_step)
    
    MERGE PAIRED DATA
     After the search loop is finished, remove unpaired peaks (fp_id == -1) in c1 and c2 and merge the two columnfiles. 
     Sort the merged columnfile by fp_id, so c_merged.fp_id = [0,0,1,1,2,2,...n,n]
     
    PLOT
     plot some statstics about the distance between paired peaks. Euclidian distance in 4D search space 
     and distance along each individual dimension, expressed in more meaningful quantities
     (angles for eta, omega and two-theta, normalized intensity for I)
     
     
    Args:
    --------
    c1, c2    : set of columnfiles corresonding to symmetric scans [dty, -dty]
    dist_max  : float parameter controlling the max distance threshold to apply for the Friedel pair search loop.
                Anything close to 1 is usually a good starting guess. 
    dist_step : float parameter controlling dist_cutoff increase at each iteration. Must be << dist_max
                (typically 0.1 if dist_max = 1)
    mtth, mI  : scaling factors for two-theta and intensity dimensions in 4D search space. See also: search_space
    
    verbose : (bool) print information about pairing process
    doplot  : (bool) plot some statistics to evaluate quality of pairing
    

    Outputs:
    ---------
    c_merged  : merged columnfile containing paired peaks in c1 and c2, with friedel pair labels (fp_id) and distance
                between paired peaks (fp_dist)
    """
    
    # INITIALIZATION
    ###############################################################################################
    # sort c1 and c2 on spot3d_id at the begining. No sure whether it is useful, but does not harm
    if 'spot3d_id' in c1.titles:
        c1.sortby('spot3d_id')
        c2.sortby('spot3d_id')
    
    # create new friedel pair label + fp_dist for c1 and c2,and initialize all values to -1 (non paired)
    c1.addcolumn(np.full(c1.nrows, -1, dtype=int), 'fp_id')
    c2.addcolumn(np.full(c2.nrows, -1, dtype=int), 'fp_id')
    
    c1.addcolumn(np.full(c1.nrows, -1, dtype = np.float64), 'fp_dist')
    c2.addcolumn(np.full(c2.nrows, -1, dtype = np.float64), 'fp_dist')
    
   
    fp_labels = []   # friedel pair labels list, updated at each iteration with newly found set of friedel pairs
    npkstot = min(c1.nrows, c2.nrows)  # maximum number of pairs to find. Used to compute proportion of paired peaks
    sumI_tot = sum(c1.sum_intensity) + sum(c2.sum_intensity)
    
    # FRIEDEL PAIR SEARCH LOOP
    ###############################################################################################
    dist_steps = np.arange(dist_step, dist_max+dist_step, dist_step)  # list of dist_cutoff steps over which we will iterate
    for it,dist_cutoff in enumerate(dist_steps):
        
        # sort colfiles by fp_id, to put all unpaired peaks at the begining of the columnfile
        c1.sortby('fp_id')
        c2.sortby('fp_id')        
        
        # find Friedel pairs. c1_indx and c2_indx are indices of paired peaks in c1[msk1] and c2[msk2] 
        #(msk1 and msk2 defined in compute_csr_dist_mat)
        # Since c1 and c2 have been sorted by fp_id, putting all unpaired peaks at the begining, indices to select are the same in c1 and c2
        try:
            dij = compute_csr_dist_mat(c1, c2, dist_cutoff, mtth, mI)
            dist, c1_indx, c2_indx = clean_csr_dist_mat(dij, verbose=verbose)
        except Exception as e:
            print(f'Pairing error at dist_step {dist_cutoff:.2f}. Skip it')
            continue
                
        # update the list of friedel pair labels
        if not fp_labels:  # list is empty
            newlabels = np.arange(len(c1_indx))
        else:
            newlabels = np.arange(max(fp_labels)+1, max(fp_labels)+len(c1_indx)+1)
        
        fp_labels.extend(newlabels)
        
        
        # sanity check. make sure we are not overwriting already indexed pairs
        assert np.all([i == -1 for i in c1.fp_id[c1_indx]])
        assert np.all([j == -1 for j in c2.fp_id[c2_indx]])
        
        # update fp_id and fp_dist in c1, c2
        c1.fp_id[c1_indx]   = c2.fp_id[c2_indx]   = newlabels
        c1.fp_dist[c1_indx] = c2.fp_dist[c2_indx] = dist
        
     
    # MERGE PAIRED DATA
    ###############################################################################################
    # keep only paired peaks
    c1.filter(c1.fp_id != -1)
    c2.filter(c2.fp_id != -1)
    
    #merged the two columnfiles and sort again by fp label
    c_merged = utils.merge_peakfiles([c1, c2])
    c_merged.sortby('fp_id')
    
    if verbose: 
        print(f'==============================\nFriedel pair matching Completed.')
        print(f'N pairs = {len(fp_labels)} out of {npkstot} possible candidates')
        print(f'Fraction of peaks matched = {len(fp_labels)/npkstot:.2f}')
        print(f'Fraction of intensity matched = {sum(c_merged.sum_intensity)/sumI_tot:.2f}')
   
    

    # PLOT SOME STATISTICS: distance between paired peaks along each dimension
    ###############################################################################################

    if doplot:
        m1 = c1.fp_id>=0
        m2 = c2.fp_id>=0
        
        if verbose:
            print(f'dstep_max = {dist_steps[-1]:.3f}')
        
        # coordinates in search space
        e1,o1,tth1,sI1 = search_space(c1, mask=m1, flip_eta_omega=False)
        e2,o2,tth2,sI2 = search_space(c2, mask=m2, flip_eta_omega=True)
        
        # distance along each dimension. Those are rescaled to some meaningful range: real angles in eta, omega, tth; sumI is kept in normalized space
        eta_dist   = (e2 - e1)
        omega_dist = (o2 - o1)
        tth_dist   = c2.tth[m2] - c1.tth[m1]
        sumI_dist  = np.log10(c2.sum_intensity[m2]) - np.log10(c1.sum_intensity[m1])
        
        # for scaling axes
        def x_lim(x):
            return np.percentile(x,2), np.percentile(x,98)
        def x_bins(x):
            return np.linspace(np.percentile(x,2), np.percentile(x,98),200)
        
        
        fig = pl.figure(figsize=(8,10))
        
        ax1 = fig.add_subplot(311)
        ax1.hist(c1.fp_dist[m1], bins=x_bins(c1.fp_dist[m1]), density=True);
        ax1.set_xlabel('Normalized distance between pairs in 4D search-space')
        ax1.set_ylabel('density of pairs')
        ax1.set_xlim(x_lim(c1.fp_dist[m1]))
        
        ax2 = fig.add_subplot(323)
        ax2.hist(eta_dist, bins=x_bins(eta_dist), density=True);
        ax2.set_xlabel('eta mismatch (deg)')
        ax2.set_xlim(x_lim(eta_dist))
        
        ax3 = fig.add_subplot(324)
        ax3.hist( omega_dist , bins=x_bins(omega_dist), density=True);
        ax3.set_xlabel('omega mismatch (deg)')
        ax3.set_xlim(x_lim(omega_dist))
        
        ax4 = fig.add_subplot(325)
        ax4.hist(tth_dist , bins=x_bins(tth_dist), density=True);
        ax4.set_xlabel('2θ mismatch (deg)')
        ax4.set_xlim(x_lim(tth_dist))
        
        ax5 = fig.add_subplot(326)
        ax5.hist(sumI_dist, bins=x_bins(sumI_dist),density=True);
        ax5.set_xlabel('log(I) mismatch')
        ax5.set_xlim(x_lim(sumI_dist))
                     
        fig.suptitle('Mismatch between paired peaks')
        
    return c_merged



def process_ypair(args):
    """ group processing of y-pairs in a single function, which takes a list of arguments args as input. Useful for parallelization"""
    cf, ds, pair_id, dist_max, dist_step, mtth, mI, doplot = args
    c1, c2 = select_y_pair(cf, ds, pair_id, is_sorted=True)
    c_merged = label_friedel_pairs(c1, c2, dist_max, dist_step, mtth, mI, verbose=False, doplot=doplot)
    c_merged.sortby('fp_id')
    return c_merged


def find_all_pairs(cf, ds, dist_max=.1, dist_step=0.01, mtth = 5, mI = 1/5):
    """
    Process successively all y-pairs (peakfile subsets containg peaks from (-dty; +dty) scans) in ds.ypairs
    and find friedel pairs match. Returns a list of peakfiles containing friedel pairs match (fp_id and fp_dist columns)
    for each y-pair. 
    
    Args:
    ----------
    cf        : ImageD11 columnfile
    ds        : ImageD11.sinogram.dataset.Dataset metadata, which contains the list of ypairs
    dist_max  : float parameter controlling the max distance threshold to apply for the Friedel pair search loop.
                Anything close to 1 is usually a good starting guess. 
    dist_step : float parameter controlling dist_cutoff increase at each iteration. Must be << dist_max
                (typically 0.1 if dist_max = 1)
    mtth, mI  : scaling factors for two-theta and intensity dimensions in 4D search space. See also: search_space
    
    Returns:
    ----------
    out : list of peakfiles containing friedel pairs for each y-pair. Non-paired peaks are removed. 
    
    
    Note : Initially, this function was also merging outputs in a single peakfile. 
    The function has been splitted and thsi part of the process is now done with merge_outputs (below)
    
    Note 2: This function runs on a signle cpu (no parallelization), which can be slow for large peakfiles. 
    Use script find_friedel_pairs.py (with parallelization) for faster processing. 
    
    See also: merge_outputs, label_friedel_pairs
    """
    
    # check if ds contains ypairs. if not, compute them
    if not hasattr(ds, 'ypairs'):
        form_y_pairs(cf, ds, disp=True)
    
    # Run friedel pair search on each pair.
    ##########################################################################
    # list of arguments to be passed to process_ypair
    args = []
    ypairs = sorted(ds.ypairs)
    for pair_id in range(len(ypairs)):
        args.append((cf, ds, pair_id, dist_max, dist_step, mtth, mI, False))
    
    print('Friedel pair search...')

    out = []
    for arg in tqdm(args):
        o = process_ypair(arg)
        out.append(o)
    
    return out


def merge_outputs(out, cf = None, doplot=True, saveplot=False):
    """
    Merge list of peakfiles with friedel pairs index (fp_id) and distance between paired peaks (fp_dist).
    Make sure each friedel pair in the merged peakfile gets a unique label in fp_id.
    
    Args:
    ----------
    out : list of paired peakfiles (typically an oututs from find_all_pairs)
    cf : initial (non-paired) peakfile. Only needed if you want to get the statistics of pairing
    (proportion  of peaks / total intensity paired)
    doplot (bool) : plot fp_dist distributon
    saveplot (bool) : save plot
    
    
    Returns:
    --------------
    cf_paired : filtered peakfile with matched peaks sorted by friedel pair id
    fp_labels : list of friedel pair labels
    stats     : Friedel pair matching statistics. [frac_pks_matched, frac_intensity_matched]
    """
    
    # group all outputs into one single columnfile, and update fp_labels to make sure each pair has a unique label
    ##########################################################################
    #initialization
    c_cor = out[0]
    nrows = sum([colf.nrows for colf in out])
    fp_labels = np.full(nrows//2, -1)
    fp_labels[:c_cor.nrows//2] = np.unique(c_cor.fp_id)
    
    # update fp_labels
    print('Updating Friedel pair labels')
    for colf in tqdm(out[1:]):
        newlabels = np.arange(fp_labels.max()+1, fp_labels.max()+colf.nrows//2+1)            
        fp_labels[fp_labels.max():fp_labels.max()+colf.nrows//2] = newlabels
        colf.setcolumn(utils.recast(newlabels), 'fp_id')

    
    # merge columnfiles
    print('Merging peakfiles...')
    c_cor = utils.merge_peakfiles(out)
    
    if cf is not None:
        c_cor.parameters.parameters = cf.parameters.parameters
    
    # matching stats
    if cf is not None:
        frac_pks_matched = c_cor.nrows / cf.nrows
        frac_ints_matched = sum(c_cor.sum_intensity) / sum(cf.sum_intensity)
    
    else:
        frac_pks_matched = 0
        frac_ints_matched = 0
    
    print('==============================\nFriedel pair matching Completed.')
    print('N pairs = ', int(c_cor.nrows/2))
    print(f'Fraction of peaks matched = {frac_pks_matched:.2f}')
    print(f'Fraction of total intensity matched = {frac_ints_matched:.2f}')
    
    # plot statistics
    if doplot:
        fig = pl.figure(figsize=(7,4))
        
        ax1 = fig.add_subplot(111)
        ax1.hist(c_cor.fp_dist, bins = np.linspace(0, np.percentile(c_cor.fp_dist,99),200), density=True);
        ax1.set_xlabel('Fpairs distance (tth, eta, omega, I)')
        ax1.set_ylabel('density')
        ax1.set_xlim(0, np.percentile(c_cor.fp_dist,99))            
        fig.suptitle('dist. distribution between peaks in Fpairs')
        
        if saveplot:
            fig.savefig(os.path.join(os.getcwd(), ds.dsname, ds.dsname+'_fp_dist.png'), format='png')
                         
    return c_cor, fp_labels, [frac_pks_matched,frac_ints_matched]



# Geometry correction
##############################################################
        
def update_geometry_s3dxrd(cf, ds, update_gvecs=True):
    """ update peakfile geometry using friedel pairs. Works for scanning 3dxrd data (pencil-beam + dy translations).
    
    Adds new columns to the peakfile cf:
    - tthc : corrected two-theta
    - dsc  : corrected d-spacing (1/d)
    - xs,ys : peak coordinates in the sample reference frame
    - r_dist : distance from rotation centre in the sample reference frame
    
    Args:
    ---------
    cf : ImageD11 columnfle containing friedel pairs
    ds : ImageD11 dataset object 
    update_gvecs (bool) : if True, g-vectors coordinates (gx,gy,gz) are also updated. Default if True
    

    Details:
    If a grain is not positioned on the centre of rotation of the sample, it results in an offset t(dx,dy,dz)
    of the diffraction vector arising from this grain. This offset results in variations of the azimutal and 
    scattering angle (eta, two-theta) which are not related to the d-spacing or the orientation of the lattice.
    
    Traditionally, this offset if determined (alongside with corrected eta and 2-theta) during a refinment process after indexing.
    However, Friedel pairs properties allow to do the correction before indexing. Furthermore, it allows to relocate the origin 
    of the diffraction vector in the sample reference frame, which can then be used to do point-by-point fitting of the lattice.
    
    Principle (in brief):
    In a scanning 3dxrd experiment, the size of the thin pencil beam in y and z is small. This, it can be considered that
    the offset from the rotation center only occurs parallel to the beam, ie in the x-direction, and offset in y and z is negligible. 
    This implies that only the 2-theta angle (tth) is affected by the offset, not eta.
    
    We also know the offset in the y-direction dy, which is basically the translation dty of a given scan,
    stored in the dty column of the peakfile. With these assumptions, we find that for two peaks (p1,p2) forming a Friedel pair:
    
    * tth_cor = 1/2.(tan1 + tan2) 
    * dx = L * (tan1-tan2)/(tan1+tan2)
    
    where tan1 and tan2 are respectively tan(tth1) and tan(tth2) of p1 and  p2 and L is the distance of the detector
    from the rotation center. tth_cor is the real scattering angle, not affected by the grain offset, and dx is the offset along the 
    beam direction (x-axis). xs and ys coordinates are then obtained from dx and dy by applying a back rotation of angle omega:
    
    (xs,ys) = R.(dx,dy), where R is the oration matrix of angle omega
    
    See related publication for more detail. 
    """
    
    # check that friedel pairs have been labeled and that the columnfile is not corrupted, extract fp_ids and fp_dist and reshape them 
    ################################################################################
    assert np.all(['fp_id' in cf.titles, 'fp_dist' in cf.titles]), 'friedel pairs have not been labeled'
    assert np.all([cf.fp_id.min() >= 0, cf.fp_dist.min() != -1])  # check that all peaks in cf have been labeled           
    assert cf.nrows % 2 == 0
    
    cf.sortby('fp_id') # sort by fp label
    
    # define masks to split data into two twin columnfiles, each containing one item of each pair
    m1 = np.arange(0, cf.nrows, 2)
    m2 = np.arange(1, cf.nrows, 2)
    
    # check all fp_ids and fp_dist match
    assert np.all(np.equal(cf.fp_id[m1],cf.fp_id[m2]))
    assert np.all(np.equal(cf.fp_dist[m1], cf.fp_dist[m2]))
    
    
    # compute corrected tth + (xs,ys) peak position in sample space
    #################################################################################
    
    wl = cf.parameters.get('wavelength')
    L = cf.parameters.get('distance')
    
    # tth correction
    #################
    tan1  = np.tan(np.radians(cf.tth[m1]))
    tan2  = np.tan(np.radians(cf.tth[m2]))
    tth_cor = np.degrees(np.arctan( (tan1 + tan2)/2 ) ) 
    ds_cor = 2 * np.sin(np.radians(tth_cor)/2) / wl      # 1/d-spacing
    
    # compute (dx, dy): distance of peak from rot center along x and y axes, in lab reference frame
    ##################
    dy = (cf.dty[m1] - cf.dty[m2]) / 2
    
    if 'frelon' in ds.detector:  # dty is given in mm with the Frelon, so we convert distance to mm 
        L  = L/1000    
    dx = L * (tan1-tan2)/(tan1+tan2)  
    
    # rearrange dx, dy arrays
    dx = utils.recast(dx)
    dy = utils.recast(dy)
    r_dist = np.sqrt(dx**2 + dy**2)  # euclidian distance from rotation center in sample frame
    
    # compute (xs,ys): peak origin in sample reference frame
    ##################
    o = np.radians(cf.omega)
    o[m2] = (o[m2]-np.pi)%(2*np.pi)  # rotate omega by 180° for the second half of the data
    co,so = np.cos(o), np.sin(o)

    # omega can be slightly different between two paired peaks -> would lead to different xs,ys, which causes issues later in the processing
    # for each pair, take average value of omega and assign it to both peaks 
    co = utils.recast((co[m1]+co[m2])/2)
    so = utils.recast((so[m1]+so[m2])/2)
    xs = co*dx + so*dy
    ys = -so*dx + co*dy
    
    
    # recast arrays and add them as new columns in cf
    ##################
    cf.addcolumn(utils.recast(tth_cor), 'tthc')
    cf.addcolumn(utils.recast(ds_cor), 'dsc')
    cf.addcolumn(xs, 'xs')
    cf.addcolumn(ys, 'ys')
    cf.addcolumn(r_dist, 'r_dist')
    
    # update gvectors
    if update_gvecs:
        cf.gx, cf.gy, cf.gz = transform.compute_g_vectors(cf.tthc, cf.eta, cf.omega,
                                                          wvln  = wl,
                                                          wedge = cf.parameters.get('wedge'),
                                                          chi   = cf.parameters.get('chi'))

        
        
        
def update_geometry_boxbeam_3dxrd(cf):
    """
    update peakfile geometry using friedel pairs, in the case of a regular 3dxrd acquisition
    (box beam or letter-box beam)
    
    Adds new columns to the peakfile cf:
    - xl_c, yl_c, zl_c : corrected coordinates for the diffraction vector in the laboratory reference frame
    - tth_c, eta_c : corrected scattering angle and azimutal angle for the diffraction vector
    - g-vectors coordinates (gx,gy,gz) are also updated 
    
    Details:
    If a grain is not positioned on the centre of rotation of the sample, it results in an offset t(dx,dy,dz)
    of the diffraction vector arising from this grain. This offset results in variations of the azimutal and 
    scattering angle (eta, two-theta) which are not related to the d-spacing or the orientation of the lattice.
    
    Visualizing the problem with cartesian coordinates in the lab reference frame (xl, yl, zl) while considering the
    full experimental setup (beam + detector) rotating around the sample during a scan makes it easier to understand.
    In this setup, the peaks forming a Friedel pair (p1 and p2) and the grains they originate from are aligned, regardless
    of the grain's position in the sample. The orientation of the line (p1p2) is only determined by the lattice spacing and
    orientation of the grain. Therefore, the coordinates of the "true" diffraction vector are obtained by halving the vector
    from p2 to p1. Considering the actual experimental setup where the detector remains fixed while the sample rotates, this
    yields the following coordinates for the corrected diffraction vector (xl,yl,zl) in the laboratory reference frame:
    
    xl = 1/2 . (xl1 + xl2)
    yl = 1/2 . (yl1 + yl2)
    zl = 1/2 . (zl1 - zl2)
    
    the symmetric reflection (-h,-k,-l) is then (xl, yl, -zl) When these values have been calculated, tth and eta can be computed using
    ImageD11.transform.compute_tth_eta_from_xyz.
    
    
    Note: 
    In a standard 3DXRD acquisition (unlike scanning 3DXRD), the dimensions of the beam in y and z directions 
    cannot be neglected. Consequently, the offset vector t(dx, dy, dz) has three unknowns, and both 2-theta and
    eta require correction. Unfortunately, it is not possible to solve all these parameters with just the two
    peaks in a Friedel pair. While 2-theta and eta can be determined, solving for the translation 
    vector t(dx, dy, dz) leaves the system of equations underdetermined, with one degree of freedom remaining.
    Therefore, the positions of grains in the sample are adjusted later, using all the indexed peaks for each grain.
    """
    cf.sortby('fp_id')
    
    # get lab coordinates without detector tilt
    #############################################
    
    # replace raw pixel coordinates by corrected coordinates
    cf2 = cf.copy()
    cf2.s_raw = cf2.sc
    cf2.f_raw = cf2.fc
    
    # reset tilt and wedge parameters and run updadeGeometry()
    pars = deepcopy(cf_paired.parameters)
    pars.parameters['tilt_x'] = 0
    pars.parameters['tilt_y'] = 0
    pars.parameters['tilt_z'] = 0
    pars.parameters['wedge']  = 0
    
    cf2.setparameters(pars)
    cf2.updateGeometry()
    
    c1, c2 = friedel_pairs.split_fpairs(cf2)  # peak coordinates (xl,yl,zl) assuming no detector tilt
    
    del cf2  # not needed anymore
    
    
    # compute corrected diffraction vector v
    ##############################################
    # apparent diffraction vectors (v1,v2) in lab coordinates. Both can be described as the sum of the true diffraction vector +  some offset (dx,dy,dz)
    xl1, yl1, zl1 = c1.xl, c1.yl, c1.zl
    xl2, yl2, zl2 = c2.xl, c2.yl, c2.zl
    
    # corrected diffraction vector in lab coordinates
    x = (xl1 + xl2) / 2  #  take average for sanity, but xl1 and xl2 should be equal since xli,yli,zli have been corrected for detector tilt
    y = (yl1 + yl2) / 2
    z = (zl1 - zl2) / 2
    
    # build lab coordinate arrays for both peaks in each friedel pairs: p(hkl)-> (x,y,z); p'(-h-k-l) -> (x,y,-z) 
    x_c = utils.recast(x)
    y_c = utils.recast(y)
    z_c = np.concatenate((z,-z)).reshape((2,len(z))).T.reshape((2*len(z)))
    
    # compute corrected values in peakfile: lab coordinates, (tth,eta) and g-vectors
    ###############################################
    
    # lab coordinates
    cf.addcolumn(x_p, 'xl_c')
    cf.addcolumn(y_p, 'yl_c')
    cf.addcolumn(z_p, 'zl_c')
    
    # tth-eta coordinates
    tth_c, eta_c = ImageD11.transform.compute_tth_eta_from_xyz(np.array([x_c, y_c, z_c]), omega=cf.omega)
    cf.addcolumn(tth_c, 'tth_c')
    cf.addcolumn(eta_c, 'eta_c')
    
    # g-vectors
    gx, gy, gz = ImageD11.transform.compute_g_vectors(cf.tth_c, cf.eta_c, cf.omega,
                                                                          wvln  = cf.parameters.get('wavelength'),
                                                                          wedge = cf.parameters.get('wedge'),
                                                                          chi   = cf.parameters.get('chi'))
    cf.addcolumn(gx, 'gx')
    cf.addcolumn(gy, 'gy')
    cf.addcolumn(gz, 'gz')
    
    
    
# other functions
##############################################################
    
def split_fpairs(cf):
    """ split friedel pairs and return two peakfiles, each one containing one element of each pair
    Args:
    ---------
    cf : peakfile containing friedel pairs (fp_id column)
    
    Returns:
    c1, c2: splitted columnfiles, each containing one element of each fridel pair.
    
    c1.nrows = c2.nrows = 1/2 * cf.nrows
    """
    
    cf.sortby('fp_id') # sort by fp label
    
    # define masks to split data into two twin columnfiles, each containing one item of each pair
    m = np.arange(cf.nrows)%2 == 0
    
    # check that splitting is ok
    assert np.all(np.equal(cf.fp_id[m],cf.fp_id[~m]))
    assert np.all(np.equal(cf.fp_dist[m],cf.fp_dist[~m]))
    
    # filter peaks
    c1 = cf.copy()
    c1.filter(m)
    c2 = cf.copy()
    c2.filter(~m)
    
    return c1, c2


def exclude_singles(cf):
    """ 
    Remove all "single" peaks (which do not have a corresponding twin with the same fp_id) from peakfile.
    These singles may appear after filtering the peakfile in a way that does snot preserve friedel pairs.
    e.g. filtering by intensity
    """
    # filter peakfile to remove all peaks which do not have a Friedel pair associated
    assert 'fp_id' in cf.titles, 'Friedel pairs have not been identified. Run friedel pair search before'
    
    cf_filtered = cf.copy()
    cf_filtered.sortby('fp_id')
    
    # find fp_id values occurring twice
    uniqs, indx, cnt = np.unique(cf.fp_id, return_index=True, return_counts=True)
    to_keep = np.argwhere(cnt==2).T[0]
    
    uniqs = uniqs[to_keep]
    indx = indx[to_keep]
    indx_full = np.concatenate([(i,i+1) for i in indx])

    # define mask and keep only twins 
    mask = np.full(cf_filtered.nrows, False)
    mask[indx_full] = True
    
    cf_filtered.filter(mask)
    
    # sanity check 
    assert np.all(np.equal(cf_filtered.fp_id[::2], cf_filtered.fp_id[1::2]))
    
    return cf_filtered


def find_missing_twins(cf, selection, restrict_search=False, restrict_subset=[]):
    """
    EXPERIMENTAL. Not sure it will work in all circumstances.
    Aimed to fix missing twins (peaks missing their Friedel "twin") when selection of a subset in a peakfile does not
    preserve Friedel pairs. 
    
    - Identifies "singles" peaks (no corresponding twin with the same fp_id) in the subset
    - Find their corresponding twin in the full peakfile

    Args:
    ----------
    cf: Full peakfile with all Friedel pairs complete
    selection: subset of cf with incomplete Friedel pairs
    
    restrict_search: limit search for missing peaks to a second subset of cf (faster if cf is large)
    restrict_subset: subset of cf to search in
    
    Returns:
    new_selection: updated selection with hopefully all Friedel pairs complete
 """
    
    fpids = cf.fp_id[selection]  # fpids in selection
    
    # use np.unique to find "single" peaks: peaks missign their friedel twin
    _, ind, cnt = np.unique(fpids, return_counts=True, return_index=True)
    fp_single = fpids[ind[cnt!=2]]
    
    if len(fp_single)==0:
        return selection
    
    # list of peaks ids to select in columnfile
    if restrict_search is True:
        pks_to_add = np.concatenate([np.argwhere(cf.fp_id[restrict_subset] == i) for i in fp_single])[:,0]   # pks indices to select in subset
        newpks = restrict_subset[pks_to_add]  # pks indices to select in full subset
    else:
         newpks = np.concatenate([np.argwhere(cf.fp_id == i) for i in fp_single])[:,0]
    
    new_selection = np.unique( np.concatenate((selection, newpks)) )  # concatenate with former selection
    return new_selection
