""" 
performs local indexing on  a series of datasets, using indexing parameters specified in input 
"""


# general modules
import argparse
import os, sys, glob
import h5py
import pylab as pl
import numpy as np
import concurrent.futures, multiprocessing
from tqdm import tqdm
from interruptingcow import timeout

# ImageD11
import ImageD11.sinograms.dataset
import ImageD11.columnfile
import ImageD11.parameters
import ImageD11.indexing
import ImageD11.grain
import ImageD11.sym_u


# point-fit 3dxrd module available at https://github.com/jbjacob94/pf_3dxrd.
if '/home/esrf/jean1994b/' not in sys.path:
    sys.path.append('/home/esrf/jean1994b/')    

from pf3dxrd import utils, friedel_pairs, pixelmap, crystal_structure, peak_mapping


# UPDATE OPTIONS HERE
######################################################################

class Options():
    hkltol         = 0.15   #  hkl tolerance parameter for indexing
    minpks         = 10     # minimum number of g-vectors to consider a ubi as a possible match
    minpks_prop    = 0.1    # min. frac. of g-vecs over the selected pixel to consider a ubi as a possible match.
    nrings         = 10      # maximum number of hkl rings to search in 
    max_mult       = 12     # maximum multplicity of hkl rings to search in
    px_kernel_size = 3      # size of peak selection around a pixel: single pixel or nxn kernel
    sym            = ImageD11.sym_u.monoclinic_b()   # crystal symmetry (ImageD11.sym_u symmetry)
    ncpu           = len(os.sched_getaffinity( os.getpid() )) - 1     # ncpu. by default, use all cpus available
    chunksize      = 20                                               # size of chunks passed to ProcessPoolExecutor
    
    
# END EDITABLE PART
######################################################################
    def __init__(self):
        Options.unitcell = None
        
        
    def __str__(self):
        attrdict = {i: self.__getattribute__(i) for i in dir(self) if not i.startswith('__')}
        return '\n'.join([f'{k}:{v}' for (k,v) in attrdict.items()])
        
######################################################################



def load_data(pksfile, xmapfile, dsfile, parfile, pname):
    # paths
    ds = ImageD11.sinograms.dataset.load(dsfile)
    
    # load pixelmap
    xmap = pixelmap.load_from_hdf5(xmapfile)
    print(xmap)
    
    # crystal structure we want to index
    cs = xmap.phases.get(pname)
    pid = cs.phase_id
    
    # load cf  + keep only peaks from the phase we want to index
    cf = ImageD11.columnfile.columnfile(pksfile)
    cf.parameters.loadparameters(parfile)
    cf.updateGeometry()
    
    friedel_pairs.update_geometry_s3dxrd(cf, ds, update_gvecs=True)
    cf.filter(cf.phase_id==pid)
    utils.get_colf_size(cf)
    
    # add pixel labeling to cf + sort by pixel index
    peak_mapping.add_pixel_labels(cf, ds)
    cf.sortby('xyi')
    
    return xmap, cf, cs



def pixel_ubi_fit( args ):
    """ 
    fit ubi pixel-by-pixel. a list of possible UBI matrices matching with g-vectors over the selected pixel is found runing
    ImageD11.indexing. Then, each ubi is scored and the best-matching one is retained.
    
    outputs:
    best_ubi : best UBI matrix
    best_score : score of best_ubi. Tuple (nindx, drlv2), where nindx is the number of g-vectors assigned to best_ubi and
    drlv2 the mean square deviation from the closest integer hkl indices for assigned g-vectors
    """
    px, OPTS = args
    
    # extract keyword arguments
    unitcell    = OPTS.unitcell      # crystal unit cell to pass to ImageD11.indexer
    symmetry    = OPTS.sym           # crystal symmetry (ImageD11.sym_u symmetry) to find unique orientations
    hkltol      = OPTS.hkltol        # hkl tolerance parameter for indexing (see ImageD11.indexing)
    minpks      = OPTS.minpks        # minimum number of g-vectors to consider a ubi as a possible match (see ImageD11.indexing)
    minpks_prop = OPTS.minpks_prop   # minimum fraction of g-vectors over the selected pixel to consider a ubi as a possible match.
    max_mult    = OPTS.max_mult      # maximum multplicity of hkl rings in which possible orientation match will be searched. 
    nrings      = OPTS.nrings        # maximum number of hkl rings to search in 
    ks          = OPTS.px_kernel_size # size of peak selection around a pixel: single pixel or kernel selection
    
    
    default_output = px, np.zeros((3,3)), 0, 0  # default output returned if no ubi is found: px, ubi, nindx, drlv2
    
    # select peaks from px
    s = peak_mapping.pks_from_px(to_index.xyi, px, kernel_size=ks)
    if len(s) == 0:
        return default_output
    
    
    # prepare indexer
    ###########################################################################
    gv = np.array( (to_index.gx[s],to_index.gy[s],to_index.gz[s])).T.copy()
    ImageD11.indexing.loglevel=10  # loglevel set to high value to avoid outputs from indexer
    ind = ImageD11.indexing.indexer( unitcell = unitcell,
                                     gv = gv,
                                     wavelength=to_index.parameters.get('wavelength'),
                                     hkl_tol= hkltol,
                                     cosine_tol = np.cos(np.radians(90-1.)),
                                     ds_tol = 0.005,
                                     minpks = max(minpks, len(gv) * minpks_prop),
                                      )
    # assigntorings sometimes return errors, for a reason that is unclear to me. handle this with an exception and return empty pixel (no ubi indexed)
    try:
        ind.assigntorings()
    except Exception as e:
        print('something went wrong with indexer.assigntorings()')
        return default_output
    
    
    # find possible ubis
    ###########################################################################
    # list of hkl rings to search in
    rings = [] 
    for i, ds in enumerate(ind.unitcell.ringds): # select first nrings
        # select low multiplicity rings with nonzero nb of peaks
        if all([len(ind.unitcell.ringhkls[ds]) <= max_mult, (ind.ra == i).sum()>0]):  
            rings.append(i)
            
        if len(rings) == nrings:
            break
    if len(rings)==0:   # if no rings with peaks, return empty output (notindexed)
        return default_output
    
    # loop through rings and try to match ubis
    for j, r1 in enumerate(rings[::-1]):
        for r2 in rings[j:-1]:
            ind.ring_1 = r1
            ind.ring_2 = r2
            ind.find()
            if ind.hits is None or len(ind.hits) == 0:
                continue
            try:
                with timeout(1.5, exception=RuntimeError):
                    ind.scorethem()
            except RuntimeError:
                #print(f'rings ({r1},{r2}): timeout')
                pass
            except ValueError:
                pass
    # no ubi found
    if len(ind.ubis) ==0:
        return default_output
    
    # Refinement, scoring and selection of best ubi
    ###########################################################################    
    scores = []        # score = (npks_index, mean_drlv2) for each ubi found
    scoreproduct = []  # defined as npks_indexed/mean_drlv2. The higher the better
    nindx = []         # nb of peaks retained by score_and_refine. The higher the better
    ubis = []          # write refined ubit to new list
    
    # compute scores for all ubis found
    for i,ubi in enumerate(ind.ubis):
        sc = ImageD11.cImageD11.score_and_refine( ubi, gv, hkltol ) 
        scores.append(sc)
        scoreproduct.append(sc[0]/sc[1])
        nindx.append(sc[0])
        ubis.append( ImageD11.sym_u.find_uniq_u( ubi, symmetry ) )
    
    if len(ubis) == 0:   # no ubi found
        return default_output
    # select the best ubi: highest scoreproduct
    nindx = [sc[0] for sc in scores]
    best_score = scores[np.argmax(nindx)]
    best_ubi = ubis[np.argmax(nindx)]
    
    return px, best_ubi, best_score[0], best_score[1]



def doinit(parfile, cs):
    global to_index
    ImageD11.cImageD11.cimaged11_omp_set_num_threads(1)
    to_index = ImageD11.columnfile.colfile_from_hdf( './toindex.h5' )
    to_index.parameters.loadparameters(parfile)
    to_index.xyi = to_index.xyi.astype(int)   # convert xyi to default int type, otherwise peaksearch takes forever
    utils.update_colf_cell(to_index, cs.cell, cs.spg, cs.lattice_type, mute=True)  


    
def get_grain_props(UBI):
    try:
        g = ImageD11.grain.grain(UBI)
        return g.U, g.unitcell
    except Exception as e:   
        return np.zeros((3,3)), np.zeros(6)


    
def update_xmap(xmap, xyi_selec, results, pname, drlv2_max = 0.1, overwrite = True):
    """ write indexing outputs in xmap. updates only pixels corresponding to the indexed phase. Also reset pixels to "notindexed" if no
    orientation has been found or if indexing scores are too bad (nindx < nindx_min, drlv2 > drlv2_max
    
    Args:
    xmap    : pixelmap in which results will be written
    results : output from fitting process
    pname   : name of the phase being indexed
    drlv2_max : max threshold for drlv2. If a UBI is identified on a pixel with drlv2 > drlv2_mx, the pixel will be kept unindexed. Avoids dodgy UBIs
    overwrite : if True, reset all pixels that have already been indexed pixels for the selected phase pname before writing new data.
                Useful to set this option ot False when doing multiple tests on small subsets of the map.
    """
    
    cs = xmap.phases.get(pname)
    pid = cs.phase_id
    
    # initialize new data arrays (and add them to xmap if not yet present)
    #####################################################################
    lx = xmap.xyi.shape
    #initialization
    dnames = 'nindx drlv2 U UBI unitcell'.split(' ')
    dshapes = [lx, lx, lx+(3,3), lx+(3,3), lx+(6,)]
    initvals = [-1, -1, 0, -10, 0]
    dtypes = [np.int32, np.float64, np.float64, np.float64, np.float64]
    
    # add arrays to xmap if not yet present
    for n,shp,ival,dt in zip(dnames, dshapes, initvals, dtypes):
        ary = np.full(shp, ival, dt)               
        if n not in xmap.titles():   
            print(n, ary.shape)
            xmap.add_data(ary,n)
        
        if overwrite:
            sel = xmap.phase_id == pid
            xmap.update_pixels(xmap.xyi[sel], n, ary[sel])
    
    
    # update xmap with results
    #####################################################################
    print('extracting results...')
    UBI   =  np.array([results[px][0] for px in xyi_selec])
    nindx =  np.array([results[px][1] for px in xyi_selec])
    drlv2 =  np.array([results[px][2] for px in xyi_selec])
    
    gprops = [get_grain_props(m) for m in UBI]
    U = np.array([gp[0] for gp in gprops])
    unitcell = np.array([gp[1] for gp in gprops])
    
    print('updating xmap...')
    xmap.update_pixels(xyi_selec, 'UBI', UBI)
    xmap.update_pixels(xyi_selec, 'nindx', nindx)
    xmap.update_pixels(xyi_selec, 'drlv2', drlv2)
    xmap.update_pixels(xyi_selec, 'U', U)
    xmap.update_pixels(xyi_selec, 'unitcell', unitcell)
    
    
    # filter out bad pixels (drlv2 too high)
    #####################################################################
    bad = xmap.drlv2 > drlv2_max
    
    for n,shp,ival,dt in zip(dnames, dshapes, initvals, dtypes):
        ary = np.full(shp, ival, dt)               
        xmap.update_pixels(xmap.xyi[bad], n, ary[bad])

        
        
        
        
#####################################################################
#####################################################################
    
def main():
    
    args = parser.parse_args()

    # load data
    #################################################################
    print('\n=============================-')
    print('load data...\n')
    xmap, cf, cs = load_data(args.pksfile, args.xmapfile, args.dsfile, args.parfile, args.pname)
    
    # load Options for indexing
    OPTS = Options()
    OPTS.unitcell = ImageD11.unitcell.unitcell( cs.cell , cs.lattice_type)
    
    print('\n---------------------------------')
    print(f'PARAMETERS FOR INDEXING:')
    print(OPTS)
    print('---------------------------------')
    
    
    # prepare g-vectors to index and list of arguments to pass to local indexing function
    #################################################################   
    print('\n=============================')
    print('prepare g-vectors for indexing...')
    titles =  'gx gy gz xyi sum_intensity'.split()
    to_index = ImageD11.columnfile.colfile_from_dict( { name: cf.getcolumn(name) for name in titles } )
    #to_index.filter(to_index.sum_intensity > 50)    # optional : filter to remove some weak peaks

    # sort by pixel index and save
    to_index.sortby('xyi')
    utils.colf_to_hdf(to_index, 'toindex.h5', save_mode='full')
    
    #list of pixels to indeX
    # FULL MAP
    xyi_selec = xmap.xyi[xmap.phase_id == cs.phase_id]

    # RECTANGULAR SUBSET: uncomment these lines if you want to index only a subset of the map
    #sel = np.all([np.abs(xmap.xi-1540) < 80, np.abs(xmap.yi - 580) < 80, xmap.phase_id == cs.phase_id], axis=0)
    #xyi_selec = xmap.xyi[sel]
    
    # argument list to pass to pixel_ubi_fit
    argslist = [(px,OPTS) for px in xyi_selec]
    
    print(f'Number of pixels to process: {len(xyi_selec)}')
        
    # run local indexing in parallel
    ################################################################# 
    results = {}
    print('\n=============================')
    print('local indexing...')

    with concurrent.futures.ProcessPoolExecutor() as pool:
        pool.max_workers=max(OPTS.ncpu-1,1)
        pool.initializer = doinit(args.parfile, cs),
        pool.mp_context=multiprocessing.get_context('fork')
            
        for r in tqdm(pool.map( pixel_ubi_fit, argslist, chunksize=OPTS.chunksize), total=len(xyi_selec) ):
            results[r[0]] = r[1:]

    # add results to pixelmap
    update_xmap(xmap, xyi_selec, results, args.pname, drlv2_max = 0.1, overwrite = True)

    # make plots and save
    #################################################################   
    print('\n=============================')
    print('Make plots and save')
    for vec in [(0,0,1),(0,1,0),(1,0,0)]:
        xmap.plot_ipf_map(args.pname, ipfdir = vec, smooth=True, mf_size=3, save=True, hide_cbar=False, out=False)
    
    var_to_plot = ['nindx','drlv2']
    kw = {'cmap':'viridis'}

    for var in var_to_plot:
        xmap.plot(var, autoscale=True, smooth=True, mf_size=3, save=True, **kw)
        
    xmap.save_to_hdf5()
    
    print('DONE\n==============================================\n')
    
    
    
#####################################################################
#####################################################################

    
parser = argparse.ArgumentParser(description='Local indexing (pixel-by-pixel')
parser.add_argument('-pksfile', help='absolute path to peakfile', required=True)
parser.add_argument('-xmapfile', help='absolute path to pixelmap file', required=True)
parser.add_argument('-dsfile', help='absolute path to datset file', required=True)
parser.add_argument('-parfile', help='absolute path to parameters file', required=True)   
parser.add_argument('-pname', help='name of the phase to index. Must be in pixelmap.phases', required=True)  
    

if __name__ == "__main__":
    main()
        
    

        
        


    
