import os, sys, glob, pprint
import numpy as np, pylab as pl, h5py,hdf5plugin, fast_histogram
import skimage.transform

# Functions to work on raw hdf5 data, for quick data visualization during acquisition. Includes functions to read data and scans, plot sinogram + iradon reconstruction + background estimation function

def read_h5data( h5name, counter, scans ):
    """
    Reads rawdata from hdf5 file. h5name is the name of the hdf5 file, scans is the list of scans to read from the hdf5 file and counter is the parameter for  which we want to read the data
    """
    data = []
    with h5py.File(h5name,'r') as hin:
        for scan in scans:
            if not scan.endswith('.1'):
                continue
            data.append( hin[scan][counter][()] )
    return data

def read_h5scans(h5name):
    """
    Returns list of scans stored in a hdf5 data file
    """
    with h5py.File(h5name,'r') as hin:
        scans = [scan for scan in list(hin['/']) if scan.endswith('.1')]
        #print(list(hin[scans[0]]['measurement']))
        print(scans[0], hin[scans[0]]['start_time'][()])
        print(scans[-1], hin[scans[-1]]['start_time'][()])
        if 'end_time' not in list(hin[scans[-1]]):
            print( 'skipping', scans[-1] )
            scans = scans[:-1] # skip if still collecting
    return scans

def plotsino(h5name, **kw):
    """
    plot sinogram + iradon reconstruction from the raw data. Aimed to be used for data collected with Frelon detector
    """
    # read data and arrange bins for plotting histogram
    scans = read_h5scans(h5name)
    rot = read_h5data( h5name,'measurement/hrz_center' , scans )
    npx = [len(r) for r in rot]
    rot = np.concatenate( rot )
    ctr = np.concatenate( read_h5data( h5name, 'measurement/frelon6_roi3_avg' , scans ) )
    dty = read_h5data( h5name,'instrument/positioners/diffy', scans )
    dty = np.concatenate( [ np.full( n, y ) for n,y in zip(npx, dty)  ] )
    
    rsteps = read_h5data( h5name,'measurement/hrz_delta' , [scans[0]] )
    rstep = np.abs(np.asarray(rsteps)).mean()
    rmin = rot.min()-rstep/2
    rmax = rot.max()+rstep/2
    rbins = np.arange( rmin, rmax+rstep*0.1, rstep )
    
    ysteps = np.unique(dty)
    ystep = np.mean( [ysteps[i+1] - ysteps[i] for i in range(len(ysteps)-1) ] )
    ymin = dty.min() - ystep/2
    ymax = dty.max() + ystep/2
    ybins = np.arange( ymin, ymax+ystep*0.1, ystep )
    print(rot.shape,dty.shape,ctr.shape)
    
    # compute sinogra + recon
    sino = fast_histogram.histogram2d( rot, dty, bins=(len(rbins)-1, len(ybins)-1), 
                           range=[(rmin, rmax), (ymin,ymax)],
                           weights = ctr )
                        
    recon = skimage.transform.iradon(np.log(sino.T+1), theta=rbins[1:])
    
    # plot figure
    fig = pl.figure(figsize=(12,5))
    
    a1 = fig.add_subplot(121)
    f1 = a1.pcolormesh( rbins, ybins, sino.T/sino.max(), **kw)
    fig.colorbar(f1, ax=a1, shrink=0.7)
    a1.set_xlabel('omega')
    a1.set_ylabel('dty')
        
    a2 = fig.add_subplot(122)
    f2 = a2.pcolormesh(ybins,ybins,recon)
    fig.colorbar(f2, ax=a2, shrink=0.7)
    a2.set_xlabel('x')
    a2.set_ylabel('y')
    fig.suptitle(h5name)
    
def plotsino_eiger(h5name, roiname = 'eiger_roi1', **kw):
    """
    plot sinogram + iradon reconstruction from the raw data. For data collected with eiger detector (nanofocus station at ID11)
    """
    # read data and arrange bins for plotting histogram
    scans = read_h5scans(h5name)
    rot = read_h5data( h5name,'measurement/rot_center' , scans )
    npx = [len(r) for r in rot]
    rot = np.concatenate( rot )
    ctr = np.concatenate(read_h5data( h5name, f'measurement/{roiname}' , scans ) )
    dty = read_h5data(h5name, 'instrument/dty/value', scans)
    dty = np.concatenate( [ np.full( n, y ) for n,y in zip(npx, dty)  ] )
    
    rsteps = read_h5data( h5name,'measurement/rot_delta' , [scans[0]] )
    rstep = np.abs(np.asarray(rsteps)).mean()
    rmin = rot.min()-rstep/2
    rmax = rot.max()+rstep/2
    rbins = np.arange( rmin, rmax+rstep*0.1, rstep )
    
    ysteps = np.unique(dty)
    ystep = np.mean( [ysteps[i+1] - ysteps[i] for i in range(len(ysteps)-1) ] )
    ymin = dty.min() - ystep/2
    ymax = dty.max() + ystep/2
    ybins = np.arange( ymin, ymax+ystep*0.1, ystep )
    print(rot.shape,dty.shape,ctr.shape)
    
    # compute sinogra + recon
    sino = fast_histogram.histogram2d( rot, dty, bins=(len(rbins)-1, len(ybins)-1), 
                           range=[(rmin, rmax), (ymin,ymax)],
                           weights = ctr )
                        
    recon = skimage.transform.iradon(sino.T, theta=rbins[1:], circle=True)
    
    # plot figure
    fig = pl.figure(figsize=(10,5))
    
    a1 = fig.add_subplot(121)
    f1 = a1.pcolormesh( rbins, ybins, sino.T, **kw)
    fig.colorbar(f1, ax=a1, shrink=0.7)
    a1.set_xlabel('omega')
    a1.set_ylabel('dty')
        
    a2 = fig.add_subplot(122)
    f2 = a2.pcolormesh(ybins,ybins,recon, **kw)
    fig.colorbar(f2, ax=a2, shrink=0.7)
    a2.set_xlabel('x')
    a2.set_ylabel('y')
    fig.suptitle(h5name)
    
    return rbins, ybins, sino, recon


def estimate_background( h5name, scan, detector ):
    """
    estimate background from a list of diffraction frames of a given scan
    """
    with h5py.File(h5name,'r') as hin:
        frames = hin[scan][detector]
        off = list(range(0,10,3))
        bg = np.zeros( (len(off),frames.shape[1],frames.shape[2]), dtype=frames.dtype)
        for j,o in enumerate(off):
            bg[j] = frames[o]
            for i in range(o,len(frames),45):
                f = frames[i]
                bg[j] = np.where( f<bg[j], f, bg[j])
                print('.',end='')
    return np.median(bg, axis=0)


    
    
    
    
    
    
