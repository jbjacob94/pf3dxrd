"""
Get 2D peaks files from sparsefile. Because 2D peaks table is absurdingly large, it cannot be handled even with 300 GB memory. Create peakfiles chunks by dty scans and save them temporarily in a new folder pk2d
"""

import os, sys, numpy as np, pylab as pl
from tqdm import tqdm
import argparse
import subprocess

import ImageD11.sinograms.dataset
import ImageD11.sinograms.properties
import ImageD11.columnfile, ImageD11.blobcorrector

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from contextlib import contextmanager

if '/home/esrf/jean1994b' not in sys.path:
    sys.path.append('/home/esrf/jean1994b')

from pf_3dxrd import utils, friedel_pairs


#################################################################################

# root paths
dataroot = '/data/visitor/es1190/id11/20230421/RAW_DATA/'
sparseroot = '/data/visitor/es1190/id11/20230421/PROCESSED_DATA/SPARSEPIXELS/'
analysisroot = '/data/visitor/es1190/id11/3DXRD_apr23'

# multiprocessing
ncpu = len(os.sched_getaffinity( os.getpid() )) - 1
chunksize = 1

#################################################################################


def load_dataset(sample, dset):
    # import dataset info and save to ds file
    ds = ImageD11.sinograms.dataset.DataSet(
        dataroot = dataroot,
        analysisroot = analysisroot,
        sample = sample,
        dset = dset)

    if not os.path.exists(ds.dsfile):
        ds.import_all()
        ds.save()
    else:
        ds = ImageD11.sinograms.dataset.load(ds.dsfile)
        
    print(f' dataset {ds.dsname}: \n==============================')   
    items = 'n_ystep,n_ostep,ymin,ymax,ystep,omin,omax,ostep'.split(',')
    vals  = [ds.shape[0], ds.shape[1], ds.ymin, ds.ymax, ds.ystep, ds.omin, ds.omax, ds.ostep]
    for i,j in zip(items, vals):
        if np.issubdtype(type(j),np.integer):
            print(f'{i}: {j:d}')
        else:
            print(f'{i}: {j:.2f}')
        
    print('==============================')

    return ds



def loadpeaks(args):
    ds, scan_no = args
    
    #load data from scan (scan = scan index in ds.scans)
    sparsefile = os.path.join(sparseroot,ds.sample,'_'.join([ds.sample,ds.dset,'sparse.h5']))
    
    # if peakfile laready exists,load it and return peakfile
    h5name = os.path.join(ds.analysispath, 'pk2d', f'pk2d_{scan_no}.h5')
    if os.path.exists(h5name):
        with suppress_stdout():
            cf = ImageD11.columnfile.columnfile(h5name)
            return cf
    
    try:
        pkst = ImageD11.sinograms.properties.pks_table_from_scan( sparsefile, ds, scan_no  )
    except Exception as e:
        print(f'problem with scan {scan_no}. skip it')
        
        # create empty colfile for consistency
        titles = ['s_raw','f_raw', 'omega','dty','Number_of_pixels','sum_intensity','spot3d_id','sc','fc']
        return ImageD11.columnfile.colfile_from_dict({t:[] for t in titles})
        
    pk2d = pkst.pk2d(ds.omega, ds.dty)

    cf = ImageD11.columnfile.colfile_from_dict(pk2d)
    correct_distortion(cf)
    
    del pkst, pk2d
    
    #print(f'loaded scan {scan}\ndty =  {cf.dty.mean():.4f}\nNrows = {cf.nrows:d}')
    #utils.get_colf_size(cf)
    
    return cf


def correct_distortion(cf):
    es = ImageD11.blobcorrector.eiger_spatial(dxfile = '/data/id11/nanoscope/Eiger/e2dx_E-08-0173_20231127.edf',
                                              dyfile = '/data/id11/nanoscope/Eiger/e2dy_E-08-0173_20231127.edf') 

    d = { 's_raw': cf['s_raw'], 'f_raw': cf['f_raw'] }
    es(d)
    cf.addcolumn(d['sc'], 'sc')
    cf.addcolumn(d['fc'], 'fc')
    
    
def sort_y_scans(ds):
    """ return list of index of paired scan in ds.scans """
    central_bin = np.argmin(np.abs(ds.ybincens))
    hi_side = ds.ybincens[central_bin:]
    
    yscan_pairs_id = [(central_bin+i, central_bin-i) for i in range(len(hi_side))]
    
    return yscan_pairs_id, hi_side


@contextmanager
def suppress_stdout():
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout = saved_stdout

#################################################################################

def main():
    
    args = parser.parse_args()
    
    # Import dataset
    ###############
    ds = load_dataset(args.sample, args.dset)
    
    # load peaks_2D scan by scan and merge into one big peakfile
    ###############
    # new folder for 2dpeaks
    pk2d_folder = os.path.join(ds.analysispath,'pk2d')
    subprocess.run(f'mkdir -p {pk2d_folder}'.split(' '), check=True)
    
    # initialization
    argslist = [(ds, i) for i,_ in enumerate(ds.scans)]
    Npks = []
    SumI = []
    
    print('loading peaks from sparse...')
    
    with ProcessPoolExecutor(max_workers = ncpu) as pool:
        for i, cf in tqdm( enumerate(pool.map(loadpeaks, argslist, chunksize = chunksize)), total = len(argslist), desc = 'scans loaded'):
    
            Npks.append(cf.nrows)
            SumI.append(np.sum(cf.sum_intensity))
            
            if cf.nrows == 0:
                continue
            
            h5name = f'{pk2d_folder}/pk2d_{i}.h5'
            cols = [c for c in cf.titles if c not in ['s_raw','f_raw']]
            cf_sub = utils.select_subset(cf, cols=cols)
            utils.colf_to_hdf(cf_sub, h5name, save_mode='minimal')
            
    
       
    
    # check y symmetry
    ###############
    print('checking y-symmetry...')
    
    # reshape Npks and SumI list in tuples
    scans_pairs, hi_side = sort_y_scans(ds)
    SumI_p = [(SumI[p[0]],SumI[p[1]]) for p in scans_pairs]
    Npks_p = [(Npks[p[0]],Npks[p[1]]) for p in scans_pairs]
    
    ydata = [np.log10(SumI_p), Npks_p, [np.divide(p[1],p[0]) for p in np.log10(SumI_p)], [np.divide(p[1],p[0]) for p in Npks_p] ]
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
    f.savefig(os.path.join(ds.analysispath, ds.dsname+'_dty_alignment.png'), format='png')
    

    
        

#################################################################################
    
parser = argparse.ArgumentParser(description='load peaks from sparsefile and split by dty pairs')
parser.add_argument('-sample', help='sample name', required=True)
parser.add_argument('-dset', help='dataset name', required=True)    
    
    
if __name__ == "__main__":
    main()
