""" 
local indexing script for the friedel pairs pipeline. allows intensity scoring to better match twin domains
"""
# general modules
import os, sys, site, time, glob, argparse, subprocess

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Python path: needs to be sorted out. I copied the stuff we put at the beginning of notebooks to load ImageD11 from local user folder (cloned from github), but in production this should not be here
# python environment stuff
IMAGED11_PATH = '/home/esrf/jean1994b/ImageD11_jbjacob'  # None means do not use git, otherwise enter the name of the folder to use for the git checkout "ImageD11" or "ImageD11_version_xx", etc
CHECKOUT_PATH = 'ImageD11'  # the name of the git checkout folder within path. None means guess

if IMAGED11_PATH is not None:
    if '/data/id11/nanoscope' not in sys.path:
        sys.path.append('/data/id11/nanoscope')
    import install_ImageD11_from_git
    PYTHONPATH = install_ImageD11_from_git.setup_ImageD11_from_git(IMAGED11_PATH,CHECKOUT_PATH)

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
from  ImageD11 import friedel_pairs

# add user site package + custom paths to sys.path. may not be required depending on your python setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.extend([project_root, site.getusersitepackages()])

# pf3dxrd module available at https://github.com/jbjacob94/pf_3dxrd.
from pf3dxrd.pf3dxrd import utils, pixelmap, crystal_structure, peak_mapping, refine_ubi

import orix.vector as ovec, orix.quaternion as oq
from orix.crystal_map import Phase

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



# =============================================================================
#  Options
# =============================================================================
class Options:
    """
    Stores all parameters for local indexing.
    Designed for easy serialization and Jupyter tuning.
    """
 
    # attributes that are non-serialisable and/or recomputed on load
    _SKIP = frozenset(['sym', 'flipmats', 'unitcell', 'ncpu'])
 
    def __init__(self):
        # ---- Default numeric and logic parameters ----
        self.hkltol1        = 0.05      # hkl tolerance parameter for indexing
        self.hkltol2        = 0.03      # hkl tolerance parameter for refinement
        self.useIntensity   = False     # include intensity score for matching best ubi (refinement stage)
        self.minpks         = 10        # minimum number of g-vectors to consider a ubi as a possible match
        self.maxpks         = 5000      # cutoff value for peaks number in 1st-round indexing
        self.minpks_prop    = 0.1       # min fraction of g-vectors over pixel to consider a ubi match
        self.nrings         = 10        # max number of hkl rings to search
        self.max_mult       = 12        # max multiplicity of hkl rings to search
        self.ds_tol         = 0.005     # ds tolerance in ImageD11.indexer
        self.cosine_tol     = np.cos(np.radians(90 - 0.5))    # cosine tolerance in ImageD11.indexer
        self.px_kernel_size = 3         # kernel size around pixel (1 = single pixel)
        self.chunksize      = 20        # chunk size for ProcessPoolExecutor
        self.symmetry       = 'cubic'   # symmetry: must be one of ['cubic', 'hexagonal', 'trigonal', 'rhombohedralP', 'trigonalP', 'tetragonal', 'orthorhombic', 'monoclinic_c', 'monoclinic_a', 'monoclinic_b', 'triclinic']
        self.symmetrize_ubi = True      # If True, return the symmetry-reduced ubi for the given symmetry.
 
        # ---- Derived or system-dependent parameters ----
        self.unitcell = None
        self.sym = getattr(ImageD11.sym_u, self.symmetry)()
        self.flipmats = self.sym.group
        self.ncpu = len(os.sched_getaffinity(os.getpid())) - 1  # use all but one CPU by default
 
        # ---- slurm / cluster options ----
        self.slurm_partition = 'nice'
        self.slurm_mem_G     = 64
        self.slurm_time      = '04:00:00'
        self.slurm_cpus      = 16        # should roughly match ncpu + 1
 
    # ----- save/load methods (for Jupyter <-> batch sync) -----
 
    def to_dict(self):
        """Convert current options to a plain dict."""
        d = {}
        for k, v in self.__dict__.items():
            if k in self._SKIP:
                continue
            try:
                json.dumps(v)   # cheap serialisability check
                d[k] = v
            except (TypeError, ValueError):
                print('Warning: skipping non-serialisable attribute '
                      '"{}" ({})'.format(k, type(v).__name__))
        return d
 
    def save(self, path="indexing_pars.json"):
        """Save current options to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        print("[SAVED] Options saved to {}".format(path))
 
    @classmethod
    def load(cls, path="indexing_options.json"):
        """Load options from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            setattr(obj, k, v)
        # reinitialize symmetry-dependent / system-dependent fields
        obj.sym      = getattr(ImageD11.sym_u, obj.symmetry)()
        obj.flipmats = obj.sym.group
        obj.ncpu     = len(os.sched_getaffinity(os.getpid())) - 1
        print("[LOADED] Options loaded from {}".format(path))
        return obj
 
    def __repr__(self):
        params = ", ".join("{}={}".format(k, v) for k, v in self.__dict__.items() if not callable(v))
        return "<Options {}>".format(params)
 
    def __str__(self):
        attrs = []
        for attr, value in self.__class__.__dict__.items():
            if not attr.startswith('__') and not callable(getattr(self, attr, None)):
                attrs.append("{}: {}".format(attr, getattr(self, attr, None)))
        for attr, value in self.__dict__.items():
            if not (attr.startswith('__') or attr == 'sym'):
                attrs.append("{}: {}".format(attr, value))
        return "\n".join(attrs)
 
    def setsymmetry(self, symmetry=None):
        if symmetry is None:
            symmetry = self.symmetry
        else:
            assert symmetry in ['cubic', 'hexagonal', 'tetragonal', 'trigonal', 'rhombohedralP', 'trigonalP', 'orthorhombic', 'monoclinic_c', 'monoclinic_a', 'monoclinic_b', 'triclinic'], 'symmetry not recognized'
            self.__setattr__('symmetry', symmetry)
 
        self.sym = getattr(ImageD11.sym_u, self.symmetry)()
        print("updated symmetry to {}. Symmetry group defined in self.sym.group".format(self.symmetry))

 
######################################################################
#  Path helpers
######################################################################
def _basedir(dsfile):
    """Directory containing the dataset file."""
    return os.path.dirname(os.path.abspath(dsfile))
 
 
def _dsname(dsfile):
    """Dataset base name, stripped of '_dataset.h5' / '.h5' suffixes."""
    base = os.path.basename(dsfile)
    for suffix in ('_dataset.h5', '.h5'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return base
 
 
def _slurm_dir(dsfile):
    """Path to the slurm output folder next to the dataset."""
    d = os.path.join(_basedir(dsfile), 'slurm_fp_index')
    os.makedirs(d, exist_ok=True)
    return d


    

######################################################################
#  Slurm submission
######################################################################
 
def prepare_bash_script(pksfile, xmapfile, dsfile, parfile, pname,
                         indexing_options_file, opts):
    """
    Write a slurm batch script to the slurm/ subfolder next to the dataset.
 
    Parameters
    ----------
    pksfile               : str — path to peakfile
    xmapfile              : str — path to pixelmap file
    dsfile                : str — path to dataset file
    parfile               : str — path to parameters file
    pname                 : str — phase name to index
    indexing_options_file : str | None — path to indexing options JSON
    opts                  : Options
 
    Returns
    -------
    script_path : str — path to the written .sh file
    """
    sdir        = _slurm_dir(dsfile)
    dsname      = _dsname(dsfile)
    script_path = os.path.join(sdir, '{}_local_indexing_{}.sh'.format(dsname, pname))
 
    this_script = os.path.abspath(__file__)
 
    py_args = [
        '-pksfile  {}'.format(pksfile),
        '-xmapfile {}'.format(xmapfile),
        '-dsfile   {}'.format(dsfile),
        '-parfile  {}'.format(parfile),
        '-pname    {}'.format(pname),
    ]
    if indexing_options_file:
        py_args.append('-indexing_pars {}'.format(indexing_options_file))
 
    py_args_str = ' \\\n    '.join(py_args)
 
    script = (
        '#!/bin/bash\n'
        '#SBATCH --job-name=index_{dsname}_{pname}\n'
        '#SBATCH --nodes=1\n'
        '#SBATCH --ntasks=1\n'
        '#SBATCH --cpus-per-task={cpus}\n'
        '#SBATCH --mem={mem}G\n'
        '#SBATCH --time={time}\n'
        '#SBATCH --partition={partition}\n'
        '#SBATCH --output={sdir}/{dsname}_{pname}_%j.out\n'
        '#SBATCH --error={sdir}/{dsname}_{pname}_%j.err\n'
        '\n'
        '# ── job info ─────────────────────────────────────────────────\n'
        'echo "------------------------------------------------------------"\n'
        'echo "Job ID    : $SLURM_JOB_ID"\n'
        'echo "Node      : $SLURM_NODELIST"\n'
        'echo "CPUs      : $SLURM_CPUS_PER_TASK"\n'
        'echo "Started   : $(date)"\n'
        'echo "dsfile    : {dsfile}"\n'
        'echo "phase     : {pname}"\n'
        'echo "------------------------------------------------------------"\n'
        '\n'
        '# ── run pipeline ─────────────────────────────────────────────\n'
        'python {script} \\\n'
        '    {py_args}\n'
        '\n'
        'echo "------------------------------------------------------------"\n'
        'echo "Finished  : $(date)"\n'
        'echo "------------------------------------------------------------"\n'
    ).format(
        dsname    = dsname,
        pname     = pname,
        cpus      = opts.slurm_cpus,
        mem       = opts.slurm_mem_G,
        time      = opts.slurm_time,
        partition = opts.slurm_partition,
        sdir      = sdir,
        dsfile    = dsfile,
        script    = this_script,
        py_args   = py_args_str,
    )
 
    with open(script_path, 'w') as f:
        f.write(script)
 
    os.chmod(script_path, 0o755)
    return script_path
 
 
def submit_to_slurm(pksfile, xmapfile, dsfile, parfile, pname,
                     indexing_options_file, opts):
    """
    Prepare the batch script and submit it with sbatch.
 
    Returns
    -------
    job_id : str
    """
    script_path = prepare_bash_script(
        pksfile, xmapfile, dsfile, parfile, pname,
        indexing_options_file, opts)
 
    print('Batch script written : {}'.format(script_path))
 
    result = subprocess.run(
        ['sbatch', '--parsable', script_path],
        capture_output=True, text=True)
 
    if result.returncode != 0:
        raise RuntimeError(
            'sbatch failed:\n{}'.format(result.stderr.strip()))
 
    job_id = result.stdout.strip()
    sdir   = _slurm_dir(dsfile)
    dsname = _dsname(dsfile)
 
    print('Submitted job        : {}'.format(job_id))
    print('Monitor queue        : squeue -j {}'.format(job_id))
    print('Live stdout          : tail -f {}/{}_{}_{}.out'.format(sdir, dsname, pname, job_id))
    print('Live stderr          : tail -f {}/{}_{}_{}.err'.format(sdir, dsname, pname, job_id))
    print('Cancel               : scancel {}'.format(job_id))
 
    return job_id
 


######################################################################
#  Loading
######################################################################
 
def load_data(dsfile, pname, parfile=None, pksfile=None, xmapfile=None):
    # dataset paths
    ds = ImageD11.sinograms.dataset.load(dsfile)

    # guess other paths if not provided
    #dependent_paths = [parfile, pksfile, xmapfile]
    #if any(p is None for p in dependent_paths):
    #    par_g, pks_g, xmap_g = _guess_paths(dsfile)
    #    if parfile is None: parfile  = par_g
    #    if pksfile is None: pksfile  = pks_g
    #    if xmapfile is None: xmapfile = xmap_g

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
    cf.filter(cf.phase_ids == pid)
    utils.get_colf_size(cf)
 
    # add pixel labeling to cf
    peak_mapping.add_pixel_labels(cf, ds)
    # cf.sortby('xyi')
    return xmap, cf, cs

def _guess_paths(dsfile):
    """
    filepath helper for xmapfile, parfile, pksfile. reads from dataset if not provided
    """
    ds = ImageD11.sinograms.dataset.load(dsfile)
    parfile  = ds.parfile
    pksfile = os.path.join(_basedir(dsfile),
                            _dsname(dsfile)+'_peaks_2d_paired.h5')
    xmapfile = os.path.join(_basedir(dsfile),
                            _dsname(dsfile)+'_xmap.h5')

    for p in [parfile, xmapfile, pksfile]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                                    "file {} not found. check file path in dataset or pass it as argument in main() ".format(p)
            )
    return parfile, pksfile, xmapfile 


def _update_paths():
    """ update dependent paths with guesses from dsfile if not provided"""
    args = parser.parse_args()
    
    dependent_paths = [args.parfile, args.pksfile, args.xmapfile]
    if any(p is None for p in dependent_paths):
        par_g, pks_g, xmap_g = _guess_paths(args.dsfile)
        if args.parfile is None: args.parfile  = par_g
        if args.pksfile is None: args.pksfile  = pks_g
        if args.xmapfile is None: args.xmapfile = xmap_g
    return args

                            

######################################################################
#  Parallel computing
######################################################################
 
# ── Global worker state ──────────────────────────────────────────────────────
to_index = None
cs = None
 
# ── shared objects for fork-based worker initialisation ────────────────────
_shared_to_index = None
_shared_cs       = None
 
 
def _init_worker():
    """
    Called once in each forked worker.
    Picks up to_index / cs from the module-level globals, which are
    inherited at zero copy cost via fork/copy-on-write.
    """
    global to_index, cs
    # Force 1 thread per worker
    ImageD11.cImageD11.cimaged11_omp_set_num_threads(1)
    to_index = _shared_to_index
    cs       = _shared_cs
 
 
def _chunk_wrapper(chunk):
    """Generic chunk wrapper: applies the worker function to a list of args."""
    return [pixel_ubi_fit(args) for args in chunk]
 
 
def run_indexing_parallel(argslist, gv_to_index, OPTS, parfile, cs):
    """
    Runs the local indexing in parallel.
 
    Parameters
    ----------
    argslist    : list of (pixel, OPTS) tuples
    gv_to_index : columnfile of g-vectors to index (gx, gy, gz, xyi, norm_intensity)
    OPTS        : Options
    parfile     : str — path to parameters file
    cs          : crystal structure object
 
    Returns
    -------
    dict
        {pixel: (best_ubi, N_indexed, drlv2_average, completeness, I_indexed, I_correl)}
    """
    global _shared_to_index, _shared_cs
 
    print('\n=============================')
    print('Preparing g-vectors for indexing...')
 
    to_index_local = gv_to_index
    to_index_local.parameters.loadparameters(parfile)
    to_index_local.xyi = to_index_local.xyi.astype(int)
 
    utils.update_colf_cell(to_index_local, cs.cell, cs.spg, cs.lattice_type, mute=True)
    wl = to_index_local.parameters.get('wavelength')
    cs.str_dans.Scatter.setup_scatter(scattering_type='xray',
                                      energy_kev=utils.get_Xray_energy(wl))
 
    size_mb = utils.get_colf_size(to_index_local, disp=False)
    print('to_index ready: {} rows, {:.2f} MB'.format(to_index_local.nrows, size_mb))
 
    print('\n=============================')
    print('Local indexing...')
 
    chunks = [argslist[i:i + OPTS.chunksize] for i in range(0, len(argslist), OPTS.chunksize)]
    ctx    = multiprocessing.get_context('fork')
 
    # expose to_index/cs to workers via module-level globals — workers
    # inherit these at zero copy cost when forked
    _shared_to_index = to_index_local
    _shared_cs       = cs
 
    results = {}
    try:
        with ProcessPoolExecutor(
            max_workers=max(OPTS.ncpu, 1),
            mp_context=ctx,
            initializer=_init_worker,
        ) as pool:
 
            futures = {
                pool.submit(_chunk_wrapper, chunk): chunk
                for chunk in chunks}
 
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='pixels indexed'):
 
                try:
                    for r in future.result():    # r = (px, ubi, nindx, ...)
                        results[r[0]] = r[1:]
                except Exception as exc:
                    chunk = futures[future]
                    print('[ERROR] chunk starting at px={} raised: {}'.format(chunk[0][0], exc))
    finally:
        # release references so the parent does not keep the large
        # columnfile alive unnecessarily after the pool is done
        _shared_to_index = None
        _shared_cs       = None
 
    return results
 
 

# Testing
######################
@contextmanager
def indexing_context(parfile, to_index_obj, cs_obj):
    """
    Context manager to setup to_index and clean afterwards. 
    Mean to be used for testing pixel_ubi_fit for a list of pixels
    Usage
    -----
    with indexing_context(parfile, cf, cs):
        result1 = pixel_ubi_fit((px1, OPTS))
        result2 = pixel_ubi_fit((px2, OPTS))
    """
    global to_index, cs
    cs = cs_obj
    to_index = to_index_obj
    to_index.parameters.loadparameters(parfile)
    to_index.xyi = to_index.xyi.astype(int)
    
    utils.update_colf_cell(to_index, cs.cell, cs.spg, cs.lattice_type, mute=True)
    wl = to_index.parameters.get('wavelength')
    cs.str_dans.Scatter.setup_scatter(scattering_type='xray',
                                      energy_kev=utils.get_Xray_energy(wl))
    
    ImageD11.cImageD11.cimaged11_omp_set_num_threads(1)
    
    size_mb = utils.get_colf_size(to_index, disp=False)
    print(f"to_index loaded: {to_index.nrows} rows, {size_mb:.2f} MB\n")
    
    try:
        yield to_index, cs
    finally:
        to_index = None
        cs  = None



######################################################################
#  CORE FUNCTION
###################################################################### 
def pixel_ubi_fit(args, loginfo=False):
    """ 
    fit ubi pixel-by-pixel. a list of possible UBI matrices matching with g-vectors over the selected pixel is found runing
    ImageD11.indexing. Then, each ubi is scored and the best-matching one is retained.
    
    outputs:
    px : pixel position (xyi index)
    best_ubi: fitted ubi matrix refined
    nindx: number of peaks indexed
    drlv2: mean g-vector residuals
    compl: indexing completeness, defined as I_indexed / I_total
    I_indexed: total indexed intensity
    Iscore: Intensity correlation: Pearson corr coeff I_measured vs I_predicted
    """
    px, OPTS = args
 
    # extract options
    unitcell    = OPTS.unitcell      # crystal unit cell to pass to ImageD11.indexer
    symmetry    = OPTS.sym           # crystal symmetry (ImageD11.sym_u symmetry) to find unique orientations
    hkltol1     = OPTS.hkltol1       # hkl tolerance parameter for indexing (see ImageD11.indexing)
    hkltol2     = OPTS.hkltol2       # hkl tolerance parameter for refinement
    ds_tol      = OPTS.ds_tol        # ds tolerance in ImageD11.indexer
    cosine_tol  = OPTS.cosine_tol    # cosine tolerance for finding pairs of peaks in ImageD11.indexer
    useInts     = OPTS.useIntensity  # include intensity score (correlation with predicted peaks) for matching best ubi (refinement stage)
    minpks      = OPTS.minpks        # minimum number of g-vectors to consider a ubi as a possible match (see ImageD11.indexing)
    maxpks      = OPTS.maxpks        # max nb of g-vectors to keep for first stage indexing. Refinement is then done using all peaks
    minpks_prop = OPTS.minpks_prop   # minimum fraction of g-vectors over the selected pixel to consider a ubi as a possible match.
    max_mult    = OPTS.max_mult      # maximum multplicity of hkl rings in which possible orientation match will be searched. 
    nrings      = OPTS.nrings        # maximum number of hkl rings to search in 
    ks          = OPTS.px_kernel_size # size of peak selection around a pixel: single pixel or kernel selection
   
    symmetrize_ubi = OPTS.symmetrize_ubi  # If True, return the symmetry-reduced ubi for the given symmetry.
    
    # default output returned if no ubi is found: px, ubi, nindx, drlv2, completeness, I_indexed, Icorrel
    default_output = px, np.full((3,3),np.nan), np.nan, np.nan, np.nan, np.nan, np.nan 
 
    # peak selection. needs at least 3 peaks for indexing
    s = peak_mapping.pks_from_px(to_index.xyi, px, kernel_size=ks)
    if len(s) < 3:
        return default_output
        
    # subset gv for first indexing: take the N-strongest gvecs only (reduces computation time). 
    #p = min(maxpks/len(s),1) * 100
    #cut = to_index.norm_intensity[s] >= np.percentile(to_index.norm_intensity[s],100-p)
    cut = _strong_peaks(to_index.norm_intensity[s], frac=0.9, min_peaks=minpks, max_peaks=maxpks)
    
    # prepare indexer
    ###########################################################################
    gvecs = np.array( (to_index.gx[s],to_index.gy[s],to_index.gz[s])).T.astype(np.float64)
    ints = to_index.norm_intensity[s]
    gv = gvecs[cut]
 
    if loginfo:
        print(f'ngvecs: total:{len(gvecs)}, selec:{len(gv)}') 
    
    ImageD11.indexing.loglevel=10  # loglevel set to high value to avoid outputs from indexer
    ind = ImageD11.indexing.indexer( unitcell = unitcell,
                                     gv = gv,
                                     wavelength=to_index.parameters.get('wavelength'),
                                     hkl_tol= hkltol1,
                                     cosine_tol = cosine_tol,
                                     ds_tol = ds_tol,
                                     minpks = max(minpks, len(gv) * minpks_prop),
                                      )
    # handle possible assigntorings() error with an exception and return default output 
    try:
        ind.assigntorings()
    except Exception as e:
        print(f'px:{px}: indexer.assigntorings() ERROR:',e)
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
    if len(ind.ubis) == 0:
        return default_output
 
    if loginfo:
        print(f'UBI guesses:\n')
        for ubi in ind.ubis:
            print(f'{ubi}\n')
    
    # score and select best ubi
    ###########################################################################    
    stats = {
        'nindx'       : [],
        'drlv2'       : [],
        'compl'       : [],
        'I_indexed'   : [],
        'I_correl'    : [],
        'score_global': [] }   # product of completeness and Icorrel
    
    ubis =  []
    
    # symmetry-reduced ubi
    for i, ubi in enumerate(ind.ubis):
        ubi_uniq = ImageD11.sym_u.find_uniq_u( ubi, symmetry )
        
        # for twins identification: loop over possible candidates and score them using reflection intensities
        if OPTS.flipmats is None:
            ubis.append(ubi_uniq)
        else:
            for j, group_rot in enumerate(OPTS.flipmats):
                ubi_rot = group_rot.dot(ubi_uniq)
                ubis.append(ubi_rot)
            
    # compute scores for all ubi candidates
    for i, ubi in enumerate(ubis):
        # scores
        res = refine_ubi.score_and_refine(ubi, gvecs, ints, cs,
                                          hkl_tol  = hkltol2,
                                          refine   = False,
                                          mergeHKL = True,
                                          useIntensity = useInts)
    
        sc_glob = res[3]*res[5] if useInts else res[3]
    
        stats['nindx'].append(res[1])
        stats['drlv2'].append(res[2])
        stats['compl'].append(res[3])
        stats['I_indexed'].append(res[4])
        stats['I_correl'].append(res[5])
        stats['score_global'].append(sc_glob)
        
        if loginfo:
            print(f'ubi:\n{ubi}\nn_indexed:{res[1]}, completeness:{res[3]:.3f}, I_correl:{res[5]:.3f}, score_global:{sc_glob:.3f}\n')
        
    # convert lists to arrays for easier handling
    for key in stats:
        stats[key] = np.array(stats[key])
    
    # select best ubi: maximizes score_global
    best     = np.argmax(stats['score_global'])
    best_ubi = ubis[best] 
    
    # Refine best ubi. Use merged HKLs. 2-stage refinement
    ###########################################################################   
    for i in range(2):
        # only compute intensity score for the last run if specified in options
        if (i == 1) and (useInts):
            use_intensity = True
        else:
            use_intensity = False
 
        # refined ubi
        res = refine_ubi.score_and_refine(best_ubi, gvecs, ints, cs,
                                          hkl_tol  = hkltol2,
                                          refine   = True,
                                          mergeHKL = True,
                                          useIntensity = use_intensity)
        
        # refinement success check returned from score_and_refine
        success = res[6]
        if not success:
            return default_output
        if loginfo:
            print(f'refinement stage {i+1}:\nubi:\n{best_ubi}\nn_indexed:{res[1]}, completeness:{res[3]:.3f}, I_correl:{res[5]:.3f}\n')
 
    # return symmetry-reduced ubi
    if symmetrize_ubi:
        ubi_final = ImageD11.sym_u.find_uniq_u( res[0], symmetry )
    else:
        ubi_final = res[0]
    
    # return final refined values
    return px, ubi_final, res[1], res[2], res[3], res[4], res[5]   # best_ubi, nindx, drlv2, compl, I_indexed, I_correl


# --- helpers --
def _strong_peaks(intensities, frac=0.9, min_peaks=3, max_peaks=None):
    """
    Select the strongest peaks until cumulative intensity reaches a given fraction of the total.

    Args
    ----
    intensities : array (N,) — peak intensities (e.g. cf.sum_intensity[s])
    frac        : float — target cumulative intensity fraction (e.g. 0.9 for 90%)
    min_peaks   : int   — minimum number of peaks to keep, regardless of frac
    max_peaks   : int or None — optional cap on number of peaks retained
    """
    n = len(intensities)
    order = np.argsort(intensities)[::-1]          # indices, brightest first
    cum_frac = np.cumsum(intensities[order]) / intensities.sum()

    # number of peaks needed to reach frac of total intensity
    n_sel = int(np.searchsorted(cum_frac, frac) + 1)
    n_sel = max(n_sel, min_peaks)
    if max_peaks is not None:
        n_sel = min(n_sel, max_peaks)
    n_sel = min(n_sel, n)   # can't select more than available

    mask = np.zeros(n, dtype=bool)
    mask[order[:n_sel]] = True
    return mask




######################################################################
#  Extract & write outputs
######################################################################
 
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
                Useful to set this option to False when doing multiple tests on small subsets of the map.
    """
    
    cs = xmap.phases.get(pname)
    pid = cs.phase_id
    
    # initialize new data arrays (and add them to xmap if not yet present)
    #####################################################################
    lx = xmap.xyi.shape
    #initialization
    dnames = 'nindx drlv2 indx_completeness intensity Icorrel U UBI unitcell'.split(' ')
    dshapes = [lx, lx, lx, lx, lx, lx+(3,3), lx+(3,3), lx+(6,)]
    initvals = [-1, -1, 0, 0, 0, np.nan, np.nan, np.nan]
    dtypes = [np.int32, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]
    
    # add arrays to xmap if not yet present
    for n,shp,ival,dt in zip(dnames, dshapes, initvals, dtypes):
        ary = np.full(shp, ival, dt)               
        if n not in xmap.titles():   
            print(n, ary.shape)
            xmap.add_data(ary,n)
        
        if overwrite:
            # reset all pixels for the selected phase
            sel = xmap.phase_ids == pid
            xmap.update_pixels(n, ary[sel], xyi_indx = xmap.xyi[sel])
    
    # update xmap with results
    #####################################################################
    print('extracting results...')
    UBI    =  np.array([results[px][0] for px in xyi_selec])
    nindx  =  np.array([results[px][1] for px in xyi_selec])
    drlv2  =  np.array([results[px][2] for px in xyi_selec])
    compl  =  np.array([results[px][3] for px in xyi_selec])
    Iindx  =  np.array([results[px][4] for px in xyi_selec]) 
    Iscore =  np.array([results[px][5] for px in xyi_selec]) 
    
    gprops = [get_grain_props(m) for m in UBI]
    U = np.array([gp[0] for gp in gprops])
    unitcell = np.array([gp[1] for gp in gprops])
    
    print('updating xmap...')
    xmap.update_pixels('UBI', UBI, xyi_indx = xyi_selec)  
    xmap.update_pixels('nindx', nindx, xyi_indx = xyi_selec)
    xmap.update_pixels('drlv2', drlv2, xyi_indx = xyi_selec)
    xmap.update_pixels('indx_completeness', compl, xyi_indx = xyi_selec)
    xmap.update_pixels('intensity', Iindx, xyi_indx = xyi_selec)
    xmap.update_pixels('Icorrel', Iscore, xyi_indx = xyi_selec)
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
    # xmap.phase_ids[bad*selec] = -1   # reset bad pixels to 'notindexed'. Risky... better keep the phase masks intact and filter when needed
 

        
#####################################################################
#####################################################################
    

def main():
    args = parser.parse_args()
    args = _update_paths()
    
    # --- Load data
    print('\n=============================-')
    print('load data...\n')
    xmap, cf, cs = load_data(args.dsfile, args.pname, args.parfile, args.pksfile, args.xmapfile)
 
    if 'norm_intensity' not in cf.titles:
        lf = ImageD11.refinegrains.lf(cf.tth, cf.eta)  # lorentz factor for intensity scaling
        cf.addcolumn(cf.sum_intensity * lf, 'norm_intensity')
 
    # --- Initialize options
    if args.indexing_pars and os.path.exists(args.indexing_pars):
        print('loading options from {}'.format(os.path.basename(args.indexing_pars)))
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
    if not gv_to_index.sortedby == 'xyi':
        print('sorting peakfile by xyi? may take some time...')
        gv_to_index.sortby('xyi')
 
    xyi_selec = xmap.xyi[xmap.phase_ids == cs.phase_id].astype(int)
    argslist = [(px, OPTS) for px in xyi_selec]
 
    print('Number of pixels to process: {}'.format(len(xyi_selec)))
 
    # --- Run parallel indexing
    results = run_indexing_parallel(argslist, gv_to_index, OPTS, args.parfile, cs)
    stats, fig = refine_ubi.compute_refinement_stats(results)
    fname = args.xmapfile.replace('xmap.h5','indx_stats.svg')
    fig.savefig(fname)
 
    # --- Update maps and finalize
    update_xmap(xmap, xyi_selec, results, args.pname, drlv2_max=0.1, overwrite=True)
 
    print('\n=============================')
    print('Make plots and save')
    xmap.save_to_hdf5()
    
    xmap.plot_ipf_orientation(args.pname, ipf_directions='xyz', save=True)
    
    for var in ['nindx', 'drlv2', 'indx_completeness', 'Icorrel']:
        xmap.plot(var, autoscale=True, smooth=False, save=True, cmap='viridis')
 
    print('DONE\n==============================================\n')
 
 

#####################################################################
#####################################################################


def _parse_bool(v):
    """Argparse helper: accept 'True'/'False'/'1'/'0' as bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected, got: {}'.format(v))

 
parser = argparse.ArgumentParser( description='Local indexing (pixel-by-pixel)')
parser.add_argument('-dsfile',
                    help='absolute path to datset file',
                    required=True)
parser.add_argument('-pname',
                    help='name of the phase to index. Must be in pixelmap.phases',
                    required=True)
parser.add_argument('-parfile',
                    help='absolute path to parameters file – optional, guessed from dataset if None',
                    required=False, default=None)
parser.add_argument('-pksfile',
                    help='absolute path to peakfile – optional, guessed from dataset if None',
                    required=False,default=None)
parser.add_argument('-xmapfile',
                    help='absolute path to pixelmap file – optional, guessed from dataset if None',
                    required=False, default=None)
parser.add_argument('-indexing_pars',
                    help='indexing parameters saved in json file – optional, use default params if None',
                    required=False, default=None)
parser.add_argument('-usecluster',
                    help='if True, write a slurm script and submit via sbatch instead of running locally',
                    required=False, default=False, type=_parse_bool)


if __name__ == "__main__":
    args = parser.parse_args() 
    # load options now so slurm parameters are available for both paths
    if args.indexing_pars and os.path.exists(args.indexing_pars):
        OPTS = Options.load(args.indexing_pars)
    else:
        OPTS = Options()
 
    if args.usecluster:
        # ── slurm path: write script + submit, then exit ──────────────
        args = _update_paths()
        submit_to_slurm(
            pksfile               = args.pksfile,
            xmapfile              = args.xmapfile,
            dsfile                = args.dsfile,
            parfile               = args.parfile,
            pname                 = args.pname,
            indexing_options_file = args.indexing_pars,
            opts                  = OPTS)
    else:
        # ── local path: run pipeline in this process ──────────────────
        # timestamped logfile name
        root = os.path.dirname(args.dsfile)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join(root, "local_indexing_{}_{}.log".format(args.pname, ts))
 
        print("\n[LOG] Writing output to {}\n".format(logfile))
 
        with log_to_file(logfile):
            main()
 
        print("\n[LOG] Completed. Full log saved to {}\n".format(logfile))

    

        
        


    
