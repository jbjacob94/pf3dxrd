""" 
performs local indexing on  a series of datasets, using indexing parameters specified in input 
"""


# general modules
import os, sys, site, time, glob, argparse, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import h5py
from tqdm import tqdm
import matplotlib.pyplot as pl
import numpy as np
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import datetime
import json
import pprint

# ImageD11
import ImageD11.sinograms.dataset
import ImageD11.columnfile
import ImageD11.parameters
import ImageD11.indexing
import ImageD11.grain
import ImageD11.sym_u

# add user site package + custom paths to sys.path. may not be required depending on your python setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.extend([project_root, site.getusersitepackages()])


# pf3dxrd module available at https://github.com/jbjacob94/pf_3dxrd.
from pf3dxrd.pf3dxrd import utils, friedel_pairs, pixelmap, crystal_structure, peak_mapping
from interruptingcow import timeout
import subprocess


@contextmanager
def suppress_stdout():
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout = saved_stdout


@contextmanager
def log_to_file(logfile_path):
    """
    Redirect all stdout/stderr output to both console and a logfile:
        with log_to_file("run.log"):
            main()
    """
    class Tee(object):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    log = open(logfile_path, "a", buffering=1)
    tee_out = Tee(sys.stdout, log)
    tee_err = Tee(sys.stderr, log)

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = tee_out, tee_err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log.close()

        

class Options:
    """
    Stores all parameters for local indexing.
    Designed for easy serialization and Jupyter tuning.
    """

    def __init__(self):
        # ---- Default numeric and logic parameters ----
        self.hkltol1        = 0.05      # hkl tolerance parameter for indexing
        self.hkltol2        = 0.03      # hkl tolerance parameter for refinement
        self.minpks         = 10        # minimum number of g-vectors to consider a ubi as a possible match
        self.maxpks         = 5000      # cutoff value for peaks number in 1st-round indexing
        self.minpks_prop    = 0.1       # min fraction of g-vectors over pixel to consider a ubi match
        self.nrings         = 10        # max number of hkl rings to search
        self.max_mult       = 12        # max multiplicity of hkl rings to search
        self.px_kernel_size = 3         # kernel size around pixel (1 = single pixel)
        self.chunksize      = 20        # chunk size for ProcessPoolExecutor
        self.symmetry       = 'cubic'   # symmetry: must be one of ['cubic', 'hexagonal', 'trigonal', 'rhombohedralP', 'trigonalP', 'tetragonal', 'orthorhombic', 'monoclinic_c', 'monoclinic_a', 'monoclinic_b', 'triclinic']

        # ---- Derived or system-dependent parameters ----
        self.unitcell = None
        self.sym = getattr(ImageD11.sym_u, self.symmetry)()
        self.ncpu = len(os.sched_getaffinity(os.getpid())) - 1  # use all but one CPU by default


    # ----- save/load methods (for Jupyter <-> batch sync) -----

    def to_dict(self):
        """Convert current options to a plain dict."""
        d = {}
        for k, v in self.__dict__.items():
            # skip unserializable fields like unitcell or ImageD11 objects
            if k in ['sym', 'unitcell']:
                continue
            d[k] = v
        return d

    def save(self, path="indexing_options.json"):
        """Save current options to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"[SAVED] Options saved to {path}")

    @classmethod
    def load(cls, path="indexing_options.json"):
        """Load options from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            setattr(obj, k, v)
        # reinitialize symmetry object
        obj.sym = getattr(ImageD11.sym_u, obj.symmetry)()
        print(f"[LOADED] Options loaded from {path}")
        return obj

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if not callable(v))
        return f"<Options {params}>"
        
    def __str__(self):
        attrs = []
        for attr, value in self.__class__.__dict__.items():
            if not attr.startswith('__') and not callable(getattr(self, attr, None)):
                attrs.append(f"{attr}: {getattr(self, attr, None)}")
        for attr, value in self.__dict__.items():
            if not (attr.startswith('__') or attr == 'sym'):
                attrs.append(f"{attr}: {value}")
        return "\n".join(attrs)

    def setsymmetry(self,symmetry=None):
        if symmetry is None:
            symmetry = self.symmetry
        else:
            assert symmetry in ['cubic', 'hexagonal', 'trigonal', 'rhombohedralP', 'trigonalP', 'tetragonal', 'orthorhombic', 'monoclinic_c', 'monoclinic_a', 'monoclinic_b', 'triclinic'], 'symmetry not recognized'
            self.__setattr__('symmetry', symmetry)
        
        self.sym = getattr(ImageD11.sym_u, self.symmetry)()
        print(f"updated symmetry to {self.symmetry}. Symmetry group defined in self.sym.group")

    
        
######################################################################

# Loading
###########
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
    friedel_pairs.update_geometry_fpairs(cf, ds)
    cf.filter(cf.phase_id==pid)
    utils.get_colf_size(cf)
    
    # add pixel labeling to cf + sort by pixel index
    peak_mapping.add_pixel_labels(cf, ds)
    #cf.sortby('xyi')
    
    return xmap, cf, cs


# parallel computing stuff
############

# ── Global worker state ──────────────────────────────────────────────────────
to_index = None
cs = None

def _init_worker(to_index_obj, cs_obj):
    global to_index, cs
    # Force 1 thread per worker
    ImageD11.cImageD11.cimaged11_omp_set_num_threads(1)
    to_index = to_index_obj
    cs = cs_obj
    # debug
    #print(cs, cs.str_dans.Cell)
    #print(f"[Worker {os.getpid()}] to_index received ({to_index.nrows} rows)")


def run_indexing_parallel(argslist, OPTS, parfile, cs):
    """
    Runs the local indexing in parallel
    
    Parameters
    ----------
    argslist : list of tuples
        Chaque élément est (pixel, OPTS)
    OPTS : Options
        Options globales
    parfile : str
        Chemin vers le fichier de paramètres
    cs : dataset/phase object
        Structure cristalline
    
    Returns
    -------
    dict
        {pixel: (best_ubi, N_indexed, drlv2_average)}
    """
    
    print('\n=============================')
    print('Loading to_index.h5...')
    
    to_index = ImageD11.columnfile.colfile_from_hdf('to_index.h5')
    to_index.parameters.loadparameters(parfile)
    to_index.xyi = to_index.xyi.astype(int)
    
    utils.update_colf_cell(to_index, cs.cell, cs.spg, cs.lattice_type, mute=True)
    wl = to_index.parameters.get('wavelength')
    cs.str_dans.Scatter.setup_scatter(scattering_type='xray', energy_kev=utils.get_Xray_energy(wl))
    
    size_mb = utils.get_colf_size(to_index, disp=False)
    print(f"to_index loaded: {to_index.nrows} rows, {size_mb:.2f} MB")
    
    print('\n=============================')
    print('Local indexing...')
    
    ctx = multiprocessing.get_context('fork')  
    
    with ProcessPoolExecutor(
        max_workers=max(OPTS.ncpu, 1),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(to_index, cs)          
    ) as pool:
        
        futures = {
            pool.submit(pixel_ubi_fit, a): a[0]   #a[0] = pixel
            for a in argslist}
        
        results = {}
        
        #with suppress_stdout():
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='pixels indexed'
            ):
            pixel = futures[future]
            
            try:
                r = future.result()  # (px, ubi, N, drlv2, ...)
                results[pixel] = r[1:]  # pixel from dict
            except Exception as exc:
                print(f'[ERROR] Pixel {pixel} raised: {exc}')
    
    return results


# Testing
######################
@contextmanager
def indexing_context(parfile, cs_obj):
    """
    Context manager to load and setup to_index and clean afterwards
    Usage
    -----
    with indexing_context(parfile, cs):
        result1 = pixel_ubi_fit((px1, OPTS))
        result2 = pixel_ubi_fit((px2, OPTS))
    """
    global to_index, cs
    cs = cs_obj
    
    to_index = ImageD11.columnfile.colfile_from_hdf('to_index.h5')
    to_index.parameters.loadparameters(parfile)
    to_index.xyi = to_index.xyi.astype(int)
    
    utils.update_colf_cell(to_index, cs.cell, cs.spg, cs.lattice_type, mute=True)
    wl = to_index.parameters.get('wavelength')
    cs.str_dans.Scatter.setup_scatter(scattering_type='xray', energy_kev=utils.get_Xray_energy(wl))
    
    ImageD11.cImageD11.cimaged11_omp_set_num_threads(1)
    
    size_mb = utils.get_colf_size(to_index, disp=False)
    print(f"to_index loaded: {to_index.nrows} rows, {size_mb:.2f} MB\n")
    
    try:
        yield to_index, cs
    finally:
        to_index = None
        cs  = None


# CORE FUNCTION
######################################
def pixel_ubi_fit( args , loginfo=False):
    """ 
    fit ubi pixel-by-pixel. a list of possible UBI matrices matching with g-vectors over the selected pixel is found runing
    ImageD11.indexing. Then, each ubi is scored and the best-matching one is retained.
    
    outputs:
    best_ubi : best UBI matrix
    best_score : score of best_ubi. Tuple (nindx, drlv2), where nindx is the number of g-vectors assigned to best_ubi and
    drlv2 the mean square deviation from the closest integer hkl indices for assigned g-vectors
    completeness : fraction of indexed intensity over total intensity (from peak_mapping.refine_px_ubi_fast)
    """
    px, OPTS = args
    
    # extract keyword arguments
    unitcell    = OPTS.unitcell      # crystal unit cell to pass to ImageD11.indexer
    symmetry    = OPTS.sym           # crystal symmetry (ImageD11.sym_u symmetry) to find unique orientations
    hkltol1     = OPTS.hkltol1       # hkl tolerance parameter for indexing (see ImageD11.indexing)
    hkltol2     = OPTS.hkltol2       # hkl tolerance parameter for refinement
    minpks      = OPTS.minpks        # minimum number of g-vectors to consider a ubi as a possible match (see ImageD11.indexing)
    minpks_prop = OPTS.minpks_prop   # minimum fraction of g-vectors over the selected pixel to consider a ubi as a possible match.
    max_mult    = OPTS.max_mult      # maximum multplicity of hkl rings in which possible orientation match will be searched. 
    nrings      = OPTS.nrings        # maximum number of hkl rings to search in 
    ks          = OPTS.px_kernel_size # size of peak selection around a pixel: single pixel or kernel selection
    maxpks      = OPTS.maxpks       # max nb of g-vectors to keep for first stage indexing. Refinement is then done using all peaks
    
    default_output = px, np.zeros((3,3)), np.nan, np.nan, np.nan  # default output returned if no ubi is found: px, ubi, nindx, drlv2, completeness
    
    # select peaks from px. 2 selection windows: s_l (large) and s (normal) respectively for indexing and refinement
    s = peak_mapping.pks_from_px(to_index.xyi, px, kernel_size=ks)
    #s_l = peak_mapping.pks_from_px(to_index.xyi, px, kernel_size=ks+2)
    
    if len(s) == 0:
        return default_output
        
    # subset gv for first indexing: take the N-strongest gvecs in s_l only. 
    #p = min(maxpks/len(s_l),1) * 100
    p = min(maxpks/len(s),1) * 100
    #cut = to_index.norm_intensity[s_l] >= np.percentile(to_index.norm_intensity[s_l],100-p)
    cut = to_index.norm_intensity[s] >= np.percentile(to_index.norm_intensity[s],100-p)

    # prepare indexer
    ###########################################################################
    #gvecs_l = np.array( (to_index.gx[s_l],to_index.gy[s_l],to_index.gz[s_l])).T.astype(np.float64)
    #gv = gvecs_l[cut]
    gvecs = np.array( (to_index.gx[s],to_index.gy[s],to_index.gz[s])).T.astype(np.float64)
    gv = gvecs[cut]

    if loginfo:
        print(f'ngvecs: total:{len(gvecs)}, selec:{len(gv)}') 
    
    ImageD11.indexing.loglevel=10  # loglevel set to high value to avoid outputs from indexer
    ind = ImageD11.indexing.indexer( unitcell = unitcell,
                                     gv = gv,
                                     wavelength=to_index.parameters.get('wavelength'),
                                     hkl_tol= hkltol1,
                                     cosine_tol = np.cos(np.radians(90-.5)),
                                     ds_tol = 0.005,
                                     minpks = max(minpks, len(gv) * minpks_prop),
                                      )
    # assigntorings sometimes return errors, for a reason that is unclear to me. handle this with an exception and return empty pixel (no ubi indexed)
    try:
        ind.assigntorings()
    except Exception as e:
        print('something went wrong with indexer.assigntorings()')
        return default_output
    
    
    # find list of matching ubis
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
                with timeout(1., exception=RuntimeError):
                    ind.scorethem()
            except RuntimeError:
                #print(f'rings ({r1},{r2}): timeout')
                pass
            except ValueError:
                pass
    # no ubi found
    if len(ind.ubis) ==0:
        return default_output
    
    # score and select best ubi
    ###########################################################################    
    scores = []        # score = (npks_index, mean_drlv2) for each ubi found
    scoreproduct = []  # defined as npks_indexed/mean_drlv2. The higher the better
    nindx = []         # nb of peaks retained by score_and_refine. The higher the better
    ubis = []          # write refined ubit to new list

    if loginfo:
        print(f'UBI guesses:\n')
        for ubi in ind.ubis:
            print(f'{ImageD11.sym_u.find_uniq_u( ubi, symmetry )}\n')
    
    # compute scores (nindfx, drlv2 mean) for all ubis found
    for i,ubi in enumerate(ind.ubis):
        sc = ImageD11.cImageD11.score_and_refine( ubi, np.ascontiguousarray(gvecs), hkltol2 )    # np.ascontiguousarray: avoids C extension making a copy of the array and returning annoying warning message
        scores.append(sc)
        scoreproduct.append(sc[0]/sc[1])
        nindx.append(sc[0])
        ubis.append( ImageD11.sym_u.find_uniq_u( ubi, symmetry ) )
        if loginfo:
            print(f'ubi:\n{ubis[i]}\nscore:{sc}, {scoreproduct[i]}\n') 
    
    if len(ubis) == 0:   # no ubi found
        return default_output
    # select the best ubi that maximizes ratio nindx/drlv2 
    nindx = [sc[0] for sc in scores]
    drlv2 = [sc[1] for sc in scores]
    best  =  np.argmax(scoreproduct)
    best_score, best_ubi = scores[best], ubis[best]

    # 2nd stage refinement for best ubi: exclude dodgy g-vectors and recompute ubi using g-vectors merged by hkl (better orientation fit)
    ints = to_index.norm_intensity[s]
    ubi = best_ubi
    out = peak_mapping.refine_px_ubi_fast(gvecs, ints, ubi, hkltol2)   # out: (ubi_refined, nindx, drlv2, completeness, ang_shift)

    return px, out[0], out[1], out[2], out[3]


# Extract & write outputs
#############################
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
    overwrite : if True, reset all pixels that have already been indexed for the selected phase pname before writing new data.
                Useful to set this option ot False when doing multiple tests on small subsets of the map.
    """
    
    cs = xmap.phases.get(pname)
    pid = cs.phase_id
    
    # initialize new data arrays (and add them to xmap if not yet present)
    #####################################################################
    lx = xmap.xyi.shape
    #initialization
    dnames = 'nindx drlv2 indx_completeness U UBI unitcell'.split(' ')
    dshapes = [lx, lx, lx, lx+(3,3), lx+(3,3), lx+(6,)]
    initvals = [-1, -1, 0, 0, 0, 0]
    dtypes = [np.int32, np.float64, np.float64, np.float64, np.float64, np.float64]
    
    # add arrays to xmap if not yet present
    for n,shp,ival,dt in zip(dnames, dshapes, initvals, dtypes):
        ary = np.full(shp, ival, dt)               
        if n not in xmap.titles():   
            print(n, ary.shape)
            xmap.add_data(ary,n)
        
        if overwrite:
            # reset all pixels for the selected phase
            sel = xmap.phase_id == pid
            xmap.update_pixels(n, ary[sel], xyi_indx = xmap.xyi[sel])
    
    
    # update xmap with results
    #####################################################################
    print('extracting results...')
    UBI   =  np.array([results[px][0] for px in xyi_selec])
    nindx =  np.array([results[px][1] for px in xyi_selec])
    drlv2 =  np.array([results[px][2] for px in xyi_selec])
    compl =  np.array([results[px][3] for px in xyi_selec])    
    
    gprops = [get_grain_props(m) for m in UBI]
    U = np.array([gp[0] for gp in gprops])
    unitcell = np.array([gp[1] for gp in gprops])
    
    print('updating xmap...')
    xmap.update_pixels('UBI', UBI, xyi_indx = xyi_selec)  
    xmap.update_pixels('nindx', nindx, xyi_indx = xyi_selec)
    xmap.update_pixels('drlv2', drlv2, xyi_indx = xyi_selec)
    xmap.update_pixels('indx_completeness', compl, xyi_indx = xyi_selec)
    xmap.update_pixels('U', U, xyi_indx = xyi_selec)
    xmap.update_pixels('unitcell', unitcell, xyi_indx = xyi_selec)
    
    # filter out bad pixels (drlv2 too high, indexing error)
    #####################################################################
    # build boolean mask from xyi_selec -> selection of pixels being indexed
    selec = np.full(xmap.xyi.shape, False)
    pxindx = np.searchsorted(xmap.xyi, xyi_selec)
    selec[pxindx] = True
    # boolean mask for bad indexing
    bad = np.any([xmap.drlv2 > drlv2_max, xmap.nindx <= 0], axis=0)
    
    
    for n,shp,ival,dt in zip(dnames, dshapes, initvals, dtypes):
        ary = np.full(shp, ival, dt)               
        xmap.update_pixels(n, ary[bad*selec], selection_mask = bad*selec)
    xmap.phase_id[bad*selec] = -1   # reset bad pixels to 'notindexed' 

        
        
          
#####################################################################
#####################################################################
    
def main():
    args = parser.parse_args()

    # --- Load data
    print('\n=============================-')
    print('load data...\n')
    xmap, cf, cs = load_data(args.pksfile, args.xmapfile, args.dsfile, args.parfile, args.pname)

    if 'norm_intensity' not in cf.titles:
        lf = ImageD11.refinegrains.lf(cf.tth, cf.eta)  # lorentz factor for intensity scaling
        cf.addcolumn(cf.sum_intensity * lf, 'norm_intensity')

    # --- Initialize options
    if os.path.exists(args.indexing_pars):
        print(f'loading options from {os.path.basename(args.indexing_pars)}')
        OPTS = Options.load(args.indexing_pars)
    else:
        OPTS = Options()  # use default if no option file found
    OPTS.setsymmetry()
    OPTS.unitcell = ImageD11.unitcell.unitcell(cs.cell, cs.lattice_type)
    print('\n---------------------------------')
    print('PARAMETERS FOR INDEXING:')
    print(OPTS)
    print('---------------------------------')

    # --- Prepare data for indexing
    print('\n=============================')
    print('prepare g-vectors for indexing...')
    titles = 'gx gy gz xyi norm_intensity'.split()
    gv_to_index = ImageD11.columnfile.colfile_from_dict(
        {name: cf.getcolumn(name) for name in titles}
    )
    if not cf.sortedby == 'xyi':
        gv_to_index.sortby('xyi')
    utils.colf_to_hdf(gv_to_index, 'to_index.h5', save_mode='full')

    xyi_selec = xmap.xyi[xmap.phase_id == cs.phase_id].astype(int)
    argslist = [(px, OPTS) for px in xyi_selec]

    print(f'Number of pixels to process: {len(xyi_selec)}')

    # --- Run parallel indexing
    results = run_indexing_parallel(argslist, OPTS, args.parfile, cs)

    # --- Update maps and finalize
    update_xmap(xmap, xyi_selec, results, args.pname, drlv2_max=0.1, overwrite=True)
    subprocess.run('rm to_index.h5'.split(), check=True)

    print('\n=============================')
    print('Make plots and save')
    xmap.plot_ipf_orientation(args.pname, ipf_directions='xyz', save=True)
    
    for var in ['nindx', 'drlv2', 'indx_completeness']:
        xmap.plot(var, autoscale=True, smooth=False, save=True, cmap='viridis')

    xmap.save_to_hdf5()
    print('DONE\n==============================================\n')
    
    
    
#####################################################################
#####################################################################

    
parser = argparse.ArgumentParser(description='Local indexing (pixel-by-pixel')
parser.add_argument('-pksfile', help='absolute path to peakfile', required=True)
parser.add_argument('-xmapfile', help='absolute path to pixelmap file', required=True)
parser.add_argument('-dsfile', help='absolute path to datset file', required=True)
parser.add_argument('-parfile', help='absolute path to parameters file', required=True)   
parser.add_argument('-indexing_pars', help='indexing parameters saved in json file (optional)', required=False)
parser.add_argument('-pname', help='name of the phase to index. Must be in pixelmap.phases', required=True)  
    

if __name__ == "__main__":
    # timestamped logfile name
    args = parser.parse_args()
    root = os.path.dirname(args.dsfile)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(root,f"local_indexing_{ts}.log")

    print(f"\n[LOG] Writing output to {logfile}\n")

    with log_to_file(logfile):
        main()

    print(f"\n[LOG] Completed. Full log saved to {logfile}\n")
    

        
        


    
