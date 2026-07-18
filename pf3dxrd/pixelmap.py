import os, sys, copy, h5py, tqdm
import numpy as np, pylab as pl
import subprocess
import inspect, re

from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm, matplotlib.colors as mcolors

import scipy.ndimage as ndi
import skimage.transform, skimage.morphology

import ImageD11.cImageD11
import ImageD11.columnfile
import ImageD11.grain
import ImageD11.refinegrains
import ImageD11.unitcell
import ImageD11.sinograms.tensor_map as tensor_map
import xfab

from orix import data, io, plot as opl, quaternion as oq, vector as ovec
from pf3dxrd.pf3dxrd import utils, crystal_structure, local_indexing, peak_mapping, orientation, refine_ubi

"""
plot scanning 3DXRD outputs on a 2D pixelmap. 
"""
    
       
# Pixelmap Class
###########################################################################
###########################################################################

class Pixelmap:
    """ A class to store pixel information on a 2d grid """
    
    ##########################
    def __init__(self, xbins, ybins, h5name=None):
        # grid + pixel index
        self.grid = self.GRID(xbins, ybins)
        self.xyi = np.asarray([i + 10000*j for j in ybins for i in xbins]).astype(np.int32)
        self.xi = np.array(self.xyi % 10000, dtype=np.int16)
        self.yi = np.array(self.xyi // 10000, dtype=np.int16)
        
        # phase / grain labeling  + crystal structure information
        self.phases = self.PHASES() 
        self.phase_ids = np.full(self.xyi.shape, -1, dtype=np.int8)   # map of phase_ids
        self.grain_ids = np.full(self.xyi.shape, -1, dtype=np.int16)   # map of grain_ids
        
        # grains
        self.grains = self.GRAINS_DICT()
        
        self.h5name = h5name
        self.dsname = os.path.basename(h5name).split('_x')[0]
    
    def __str__(self):
        return f"Pixelmap:\n size: {self.grid.shape},\n phases: {self.phases.pnames},\n phase_ids: " +\
               f"{self.phases.pids},\n titles: {self.titles()}, \n grains: {len(self.grains.glist)}"
    
    def get(self,attr):
        """ alias for __getattribute__"""
        return self.__getattribute__(attr).copy()

    def get_phase_mask(self, phase):
        """ method to mask map by selected phase """
        if phase not in self.phases.pnames:
            raise ValueError('phase not in self.phases')
        return self.phase_ids == self.phases.get(phase).phase_id

    def as_grid(self):
        """
        Reshape all flat (N, ...) ndarray attributes into grid form
        using self.grid_shape.
        Returns full copy of xmap with reshaped arrays
        """
        grid_shape = self.grid.shape
        N = np.prod(grid_shape)
        target = self.copy()

        for name in self.titles():
            val = self.get(name)
            if isinstance(val, np.ndarray) and val.ndim >= 1:
                if val.shape[0] == N:
                    # reshape to the grid layout & flip to match ImageD11 convention
                    reshaped_val = val.reshape((*grid_shape,
                                                *val.shape[1:])
                                              )
                    rotated_val = np.flip(reshaped_val, axis=(0, 1))
                    setattr(target, name, rotated_val)
        return target

    def to_tensor_map(self):
        """ export xmap data columnbs to ImageD11.sinograms.tensor_map"""
        phase_dict = self.phases.as_dict()
        phase_dict_tmap = {cs.phase_id:cs.to_ImageD11_unitcell() for cs in phase_dict.values()}
        
        xmap_g = self.as_grid()
        tmap = tensor_map.TensorMap(maps={t:xmap_g.get(t)[np.newaxis,...] for t in xmap_g.titles()},
                                    phases = phase_dict_tmap)
        tmap.get_ipf_maps()
        del xmap_g
        return tmap
        
    
    # subclasses
    ###########################################################################
    class GRID:
        """subclass for grid properties: bins, shape, pixel size, pixel unit"""
        def __init__(self, xbins, ybins):
            self.xbins = xbins
            self.ybins = ybins
            self.shape = (len(xbins),len(ybins))
            self.nx = len(xbins)
            self.ny = len(ybins)
            self.pixel_size = 1
            self.pixel_unit = 'um'
            
            
        def __str__(self):
            return f"grid: size: {self.shape}, pixel size: {self.pixel_size:.2f} {self.pixel_unit}"
   
        def scalebar(self):
            """ scalebar for plotting maps"""
            scalebar =  ScaleBar(dx = self.pixel_size,
                                     units = self.pixel_unit,
                                     length_fraction=0.2,
                                     location = 'lower left',
                                     box_color = 'w',
                                     box_alpha = 0.5,
                                     color = 'k',
                                     scale_loc='top')
            return scalebar
        
    
    class PHASES:
        """ sub-class to store information on crystal structures."""
        def __init__(self):
            self.notIndexed = crystal_structure.CS(name='notIndexed')
            self.pnames = ['notIndexed']
            self.pids = [-1]
           
        def __str__(self):
            return f"phases: {self.pnames}"
        
        
        def get(self,attr):
            return self.__getattribute__(attr)

        def as_dict(self):
            return {pname: self.get(pname) for pname in self.pnames if pname != 'notIndexed'}
            
            
            
        def add_phase(self, pname, cs):
            """ add phase to pixelmap.phases. 
            
            Parameters
            ----------
            pname (str) : phase name
            cs          : crystal_structure.CS object
            """
            # if this phase name already exists, delete it
            if pname in self.pnames:
                print(pname, ': There is already phase with this name in self.phases. Will overwrite it.')
                self.delete_phase(pname)
                
            # write new phase and update pnames and pids lists    
            setattr(self, pname, cs)
            self.pnames.append(pname)
            self.pids.append(cs.phase_id)
            self.sort_phase_lists()
            
            
        def delete_phase(self, pname):
            cs = self.get(pname)
            pid = cs.get('phase_id')
            path = cs.get('cif_path')
            self.pnames = [p for p in self.pnames if p != pname]
            self.pids = [i for i in self.pids if i != pid]
            delattr(self, pname)
            self.sort_phase_lists()
             
                   
        def sort_phase_lists(self):
            """ sort pnames and pids by phase id """
            sorted_pids = [l1 for (l1, l2) in sorted(zip(self.pids, self.pnames), key=lambda x: x[0])]
            sorted_pnames = [l2 for (l1, l2) in sorted(zip(self.pids, self.pnames), key=lambda x: x[0])]
            self.pids = sorted_pids
            self.pnames = sorted_pnames
            
            
            
    class GRAINS_DICT:
        """ sub-class to store grains information. Wrapper containing a dictionnary of ImageD11.grain.grain objects """
        def __init__(self):
            self.dict = {}
            self.gids = list(self.dict.keys())
            self.glist = list(self.dict.values())
            
            
        def __str__(self):
            return f"nb grains: {len(self.glist)}"
        
        
        def get(self,prop, grain_ids):
            """ Return a property for a grain. Shortcut for g.__getattribute__(prop)"""
            g = self.dict[grain_ids]
            return g.__getattribute__(prop)
        
            
        def get_all(self, prop, pname=None):
            """ 
            return selected grain property for all grains in grains_dict as an array.
            
            prop (str): property to select in grains , e.g. 'UBI'
            pname : select specific phase. If None, all grains are selected. Default is None 
            """
            if pname is None:
                glist = self.glist
            else:
                glist = self.select_by_phase(pname)
            return np.array( [g.__getattribute__(prop) for g in glist] )
        
        
        def select_by_phase(self, pname):
            """ return all grains corresponding to a given phase.
            pname : phase name, must be in xmap.phases"""
            gsel = [g for g in self.glist if g.phase == pname]
            return gsel
        
        
        def add_prop(self, prop, grain_ids, val):
            """ add new property to a grain.
            prop : name for new property to add
            val  : value of new property"""
            g = self.dict[grain_ids]
            setattr(g, prop, val)
        
                     
        def plot_grains_prop(self, prop, s_factor=10, autoscale=False, percentile_cut=[5,95], out=False, **kwargs):
            """ Make scatter plot of grains colored by selected scalar property, where (x,y) is grain centroid position
            and s is grainsize. 
            
            Args:
            ---------
            prop (str)    : a scalar grain property (e.g. "grains size", "GOS", etc.). 
                            If prop="strain" or prop="stress", all strain /stress components will be combined
                            in a single plot.
            
            s_factor   : scaling factor to adjust spot size on the scatter plot. size = grainsize / s_factor
            autoscale  : (bool) automatically adjust color scale to distribution for each strain / stress component.
            Default is False
            percentile_cut [low,up]: percentile thresholds to cut distribution and adjust colorbar limits (with autoscale)
            out (bool) : return figure. Default is False
            
            **kwargs : additional keyword arguments for plotting
            """
            
            try:
                cen = self.get_all('centroid')
                gs = self.get_all('grainsize')
            except:
                print('missing grainSize or centroid position')
                return
            
            if prop not in 'strain,stress'.split(','): 
                assert np.all( [hasattr(g, prop) for g in self.glist] )
                colorsc = self.get_all(prop) # color scale defined by selected property
        
                fig = pl.figure(figsize=(6,6))
                ax = fig.add_subplot(111, aspect='equal')
                ax.set_axis_off()
                sc = ax.scatter(cen[:,0], cen[:,1], s = gs/s_factor, c = colorsc, **kwargs)
                ax.set_title(prop)
                cbar = pl.colorbar(sc, ax=ax, orientation='vertical', pad = 0.05, shrink=0.65)
                cbar.formatter.set_powerlimits((-1, 1)) 
            
            else:
                vals = self.get_all(prop)
                if prop == 'strain':
                    titles = 'e11,e22,e33,e23,e13,e12'.split(',')
                else:
                    titles = 's11,s22,s33,s23,s13,s12'.split(',')
                
                fig, ax = pl.subplots(2,3, figsize=(10,7), sharex=True, sharey=True)
                ax = ax.flatten()
                
                for i, (a,t) in enumerate(zip(ax, titles)):
                    a.set_aspect('equal')
                    a.set_axis_off()
                    x = vals[:,i]
                    low, up = np.percentile(x, (percentile_cut[0],percentile_cut[1]))
    
                    # plots
                    if autoscale:
                        norm=pl.matplotlib.colors.CenteredNorm(vcenter=np.median(x), halfrange=up)
                        sc = a.scatter(cen[:,0], cen[:,1], s = gs/s_factor, c = x, norm=norm, **kwargs)
                    else:
                        sc = a.scatter(cen[:,0], cen[:,1], s = gs/s_factor, c = x, **kwargs)
                    a.set_title(t)
            
                    # colorbar
                    cbar = pl.colorbar(sc, ax=a, orientation='vertical', pad=0.04, shrink=0.65)
                    cbar.formatter.set_powerlimits((-1, 1)) 
            
            # Adjust layout
            fig.tight_layout()
            fig.suptitle('grain scatterplot - '+prop, y=1.0)
                    
            if out:
                return fig
            
            
            
        def hist_grains_prop(self, prop, percentile_cut=[2,98], nbins = 100, out=False, **kwargs):
            """ plot histogram of selected grains property. 
            Args:
            --------
            prop (str): scalar grain property (e.g "GOS", "grainsize").
            percentile_cut [low,up]: percentile thresholds to trim distribution and adjust histogram width
            nbins (int) : number of bins in histogram
            out (bool): return figure """
            
            assert np.all( [hasattr(g, prop) for g in self.glist] )
            x = self.get_all(prop) 
            low, up = np.percentile(x, (percentile_cut[0],percentile_cut[1]))
            bins = np.linspace(low,up, nbins)
        
            fig = pl.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            h = ax.hist(x, bins, **kwargs)
            ax.vlines(np.median(x), ymin=0, ymax=h[0].max(), colors='r', label='median')
            ax.set_xlim(low, up)
            ax.set_title(prop)
            
            # Adjust layout
            fig.tight_layout()
            fig.suptitle('distribution - '+prop, y=1.0)
            
            if out:
                return fig

                

    # methods
    ###########################################################################
    def add_data(self, data, datacolname):
        """ add a data column to pixelmap.
        preferentially use numpy array or ndarray of shape(nx*ny,n), but lists may work as well"""
        assert len(data) == self.grid.nx * self.grid.ny
        setattr(self, datacolname, data)
        
        
        
    def rename_data(self, oldname, newname):
        """ rename data column """
        data = self.__getattribute__(oldname)
        setattr(self, newname, data)
        delattr(self, oldname)
        
        
        
    def titles(self):
        return [t for t in self.__dict__.keys() if t not in ['grid', 'phases', 'grains', 'h5name','dsname'] ]
        
    
    
    def copy(self):
        """ returns a deep copy of the pixelmap """
        pxmap_new = copy.deepcopy(self)
        return pxmap_new


    def update_pixels(self, datacolname, newvals, xyi_indx=None, selection_mask=None, debug=False):
        """
        Update a subset of pixels in a data column without touching others.
        2 modes:
        - selection by xyi indices (useful when working with peakfile, e.g. for indexing)
        - by boolean mask array
        If no selection, full array is updated. 

        Args:
        ---------
        datacolname (str): data column to update
        newvals: array of new values. Must have same length as the pixel selection
        xyi_indx (array, int): xyi index of pixels to update
        selection_mask (array, bool): boolean array the same length as datacolumn.
        """

        # ---- sanity checks ----
        if datacolname not in self.titles():
            raise KeyError(f"Data column '{datacolname}' not found")

        # ---- retrieve data (make a copy to avoid side effects) ----
        dat = np.array(self.get(datacolname), copy=True)
        n_pix = dat.shape[0]

        # ---- build pixel index array ----
        if xyi_indx is None and selection_mask is None:
            pxindx = np.arange(n_pix)

        elif selection_mask is not None:
            selection_mask = np.asarray(selection_mask, dtype=bool)
            if len(selection_mask) != n_pix:
                raise ValueError("selection_mask has incorrect length")
            pxindx = np.flatnonzero(selection_mask)

        else:
            xyi_indx = np.asarray(xyi_indx)
            if not np.all(np.isin(xyi_indx, self.xyi)):
                raise ValueError("Some xyi_indx values not found in map")

            # robust mapping from xyi value → pixel index
            lookup = {v: i for i, v in enumerate(self.xyi)}
            pxindx = np.array([lookup[v] for v in xyi_indx], dtype=int)

        if debug:
            print(f"Updating {len(pxindx)} pixels "
                  f"(min={pxindx.min()}, max={pxindx.max()})")

        # ---- validate newvals shape ----
        newvals = np.asarray(newvals)

        if dat.ndim == 1:
            if newvals.shape != (len(pxindx),):
                raise ValueError("newvals must have shape (N,)")
            dat[pxindx] = newvals.astype(dat.dtype)

        else:
            if newvals.shape != (len(pxindx),) + dat.shape[1:]:
                raise ValueError(
                    f"newvals must have shape {(len(pxindx),) + dat.shape[1:]}"
                )
            dat[pxindx, ...] = newvals.astype(dat.dtype)

        # ---- write back ----
        setattr(self, datacolname, dat)
    
            
    def update_grains_pxindx(self, mask=None, update_map=False):
        """ update grains pixel masks (pxindx / xyi_indx in grain properties), according to criterions defined in mask.
        Allows to remove bad pixels (large misorientation, low npks indexed, high drlv2, etc.) from grain masks. 
        
        Args:
        --------
        mask: bool array of same shape as data columns (grid.nx*grid.ny,) to filter bad pixels
        update_map: if True, grain_ids in pixelmap will also be updated. Default is False
        """
        
        if mask is None:
            mask = np.full(self.xyi.shape, True)
    
        assert mask.shape == self.xyi.shape # make sure mask is the good size
    
        for gi,g in tqdm.tqdm(zip(self.grains.gids, self.grains.glist)):
            gm = np.all([mask, self.grain_ids==gi], axis=0)  # select pixels for each grain
            g.pxindx = np.argwhere(gm)[:,0].astype(np.int32)  # reassign pxindx
            g.xyi_indx = self.xyi[g.pxindx].astype(np.int32)    # pixel labeling using XYi indices. needed to select peaks from cf
        # update grain ids
        if update_map:
            self.grain_ids[~mask] = -1
        
        
        
    def filter_by_phase(self, pname):
        """ Returns a new map containing only the selected phase. Makes a deep copy of the pixelmap obj and reinitialize
        all pixels not corresponding to the selected phase. Also update h5name in new pixelmap, to avoid overwriting the former file
        
        pname : phase name. must be in self.phases
        Returns : xmap_p: new pixelmap with only the selected phase """
        
        # make a copy of pixelmap
        xmap_p = self.copy()
        xmap_p.h5name = self.h5name.replace('.h5','_'+pname+'.h5')
        # select phase
        phase = xmap_p.phases.get(pname)
        pid = phase.phase_id
        
        # update columns
        for datacolname in self.__dict__.keys():
            if datacolname in ['grid', 'xyi', 'xi', 'yi', 'phases', 'h5name', 'grains', 'dsname']:
                continue
            
            msk = self.get_phase_mask(pname)
            array = self.get(datacolname)
            
            if 'strain' in datacolname or 'stress' in datacolname:
                new_array = np.full(array.shape, float('inf'))
            elif datacolname == 'phase_ids' or datacolname == 'grain_ids':
                new_array = np.full(array.shape, -1, dtype=int)
            else:
                new_array = np.zeros_like(array)
                
            new_array[msk] = array[msk]
            xmap_p.add_data(new_array, datacolname)
        
        # update phases
        for p in xmap_p.phases.pnames:
            if p == 'notIndexed' or p == pname:
                continue
            xmap_p.phases.delete_phase(p)
            
        # update grain list
        glist = xmap_p.grains.select_by_phase(pname)
        xmap_p.grains.dict = {g.gid:g for g in glist}
        xmap_p.grains.glist = list(xmap_p.grains.dict.values())
        xmap_p.grains.gids = list(xmap_p.grains.dict.keys())  
        
        return xmap_p
    
    
    
    def add_grains_from_map(self, pname, Ucol='U', overwrite=False):
        """ 
        Use grain masks defined in grain_ids column to compute grains and add them to self.grains.

        An initial guess of the average grain UBI is obtained by averaging the pixel values over the grain mask: 
        UBI_grain = inv(U_mean.B_med), where:
        B_med is  computed from the median unit cell (a,b,c,alpha,beta,gamma) of pixels over the grain mask
        U_mean is averaged using orix.quaternion.mean() for all pixel orientations over the grain mask

        This is just an initial guess, which needs to be refined using self.refine_grain_ubis. 
        This refinement stage needs the peakfile used for indexing,and can only be done after peaks to grains mapping has been completed. 
        See: "self.map_pks_to_grains" "self.refine_grain_ubis"

        Args:
        ----------
        pname: str, name of phase to select. must be in self.phases
        Ucol : str, orientation column to use
        overwrite: re-initialize grains dict. Default is False

        To refine grain lattice vector matrices (UBI), you need the peakfile used for indexing: first map peaks to grains ("self.map_pks_to_grains") and 
        then use all assigned peaks to fit the new unit cell matrix ("self.refine_grain_ubis") """
        
        assert 'UBI' in self.__dict__.keys()
              
        # crystal structure
        cs = self.phases.get(pname)
        pid = cs.phase_id
        sym = cs.orix_phase.point_group.laue
        
        # masks for pixel selection
        pm = np.any([self.get_phase_mask(pname), self.get_phase_mask('notIndexed')], axis=0) # phase mask. 
        isUBI = np.asarray( [np.trace(ubi) != -3 for ubi in self.UBI] )   # mask for pixels that have a consistent unit cell matrix assigned
      
        # list of unique grain_ids for the selected phase
        gid_u = np.unique(self.grain_ids[pm]).astype(np.int16)  
        
        # if overwrite, re-initialize grains dict. Otherwise, keep existing grains in grains dict and append new ones
        if overwrite:
            self.grains.__init__()
        
        # loop through unique grain_ids: for each unique grain_ids, select pixels, compute mean orientation and grain properties 
        ########################################
        for i in tqdm.tqdm(gid_u):
            # skip notindexed domains
            if i == -1:
                continue  
            # selection mask
            gm = self.grain_ids==i
            
            # compute mean grain orientation (use quaternion space for this) and return it as a matrix U_g
            ori_gi_mask = oq.Orientation.from_matrix(self.get(Ucol)[pm*gm*isUBI], symmetry = sym)
            ori_mean = ori_gi_mask.mean()
            ori_mean.symmetry = cs.orix_phase.point_group.laue
            ori_mean = ori_mean.map_into_symmetry_reduced_zone()
            U_g = ori_mean.to_matrix()
        
            # compute median B matrix
            uc_med = np.nanmedian(self.unitcell[pm*gm*isUBI], axis=0)
            try:    
                B_med = ImageD11.unitcell.unitcell(uc_med).B
            except Exception as e:
                print(f'grain_ids:{i}: {e}, {uc_med}')
                self.grain_ids[gm] = -1  # reset grain_ids in xmap
                continue
                
            # compute mean ubi and create new grain
            try:
                UBI_g = np.linalg.inv(U_g.dot(B_med))[0]
            except np.linalg.LinAlgError as e:
                print(f'grain_ids:{i}: {e}')
                self.grain_ids[gm] = -1  # reset grain_ids in xmap
                continue
    
            try:
                g = ImageD11.grain.grain(UBI_g)  
            except Exception as e:
                print(f'grain_ids:{i}:{e},{uc_med}')
                self.grain_ids[gm] = -1  # reset grain_ids in xmap
                continue
            
            # grain to xmap mapping
            g.gid = i
            g.phase = pname
            g.pxindx = np.argwhere(gm*isUBI)[:,0].astype(np.int32)  # pixel indices in grainmap matching with this grain
            g.grainsize = len(g.pxindx)
            g.surf = g.grainsize * self.grid.pixel_size**2  # grain surface in pixel_unit square
            g.xyi_indx = self.xyi[g.pxindx]    # pixel labeling using XYi indices. needed to select peaks from cf
            

            # Grain orientation spread: compute misorientation angle and take the median over the grain
            try:
                og = oq.Orientation.from_matrix(g.U, symmetry = sym)
                #opx = oq.Orientation.from_matrix(self.U[gm*isUBI], symmetry=sym)
                misOrientation = og.angle_with(ori_gi_mask, degrees=True)
                g.GOS = np.median(misOrientation)  # grain orientation spread
            except Exception as e:
                print(f'grain_ids:{i}:error computing misorientations')
                continue
                
            # grain centroid
            cx = np.average(self.xi[g.pxindx], weights = self.nindx[g.pxindx])
            cy = np.average(self.yi[g.pxindx], weights = self.nindx[g.pxindx])
            g.centroid = np.array([cx,cy])
                
            # add grain to grains dict
            self.grains.glist.append(g)
            self.grains.gids.append(g.gid)
        
        # update grains dict
        self.grains.dict = dict(zip(self.grains.gids, self.grains.glist))
        
        
        
    def map_pks_to_grains(self, pname, cf, overwrite=False):
        """ peaks to grains mapping. Map peaks from peakfile (cf) to grains in pixelmap for all grains in self.grains.glist. 
        updates cf.grain_ids column in peakfile and grain.pksindx for each grain in self.grain.glist        
        
        Args:
        ---------
        pname : phase name to select
        cf    : peakfile which has been used for indexing. 
        overwrite : if True, reset 'grain_ids' column in cf. default if False
        See also: peak_mapping.map_grains_to_cf
        """
        glist = self.grains.select_by_phase(pname)        
        print('peaks to grains mapping...')
        peak_mapping.map_grains_to_cf(glist, cf, overwrite=overwrite)
        self.grains.dict = dict(zip(self.grains.gids, self.grains.glist))
    
    
    
    def refine_grain_ubis(self, pname, cf, hkl_tol=0.3, ncpu=1, chunksize=10, useInts=False):  
        """  
        Run refine_ubi.refine_grain for each grain from a given phase in self.grains.glist
        Updates UBI, U and indexing metrics in grains attributes
        plot indexing metrics
        
        Args:
        ---------
        pname : phase name to select
        cf    : peakfile used for indexing
        hkl_tol : tolerance to pass to score_and_refine; use large value to account for grain orientation spread
        useInts : bool (default: False); computes intensity correlation if True
        ncpu, chunksize: parallelization options. number of workers / chunk size passed to each process
        
        Output: 
        ---------
        prop of peaks retained, angle deviation (deg) between old and new grain orientation
        """
        glist = self.grains.select_by_phase(pname)
        cs = self.phases.get(pname)

        if cf.sortedby!='xyi':
            raise ValueError(
                "cf not sorted by xyi. peaks to grains assignment has probably corrupted"
                "run cf.sortby('xyi') then self.map_pks_to_grain() to fix the peaks to grain mapping"
            )
        
        refined_glist = refine_ubi.refine_grains_ubis(cf, glist, ncpu, chunksize, hkl_tol, cs, useInts)
        
        # update self.grains.glist and self.grains.dict
        id_to_grain = {g.gid: g for g in refined_glist}
        for i, g in enumerate(self.grains.glist):
            if g.gid in id_to_grain:
                self.grains.glist[i] = id_to_grain[g.gid]
        self.grains.dict = dict(zip(self.grains.gids, self.grains.glist))

        # plot refinement stats
        # refined grains list to a results dict compatible with compute_refinement_stats.
        attrs = ['ubi', 'nindx', 'drlv2', 'completeness', 'I_indexed', 'I_corr']
        results = {getattr(g, 'id', i):
                   [getattr(g, a, np.nan) for a in attrs]
                   for i, g in enumerate(refined_glist)
                  }
                   
        stats, fig = refine_ubi.compute_refinement_stats(results)
        return stats
    
    
    
    def refine_px_ubis(self, pname, cf, UBI_col='UBI', hkl_tol=0.1, useInts=False,
                       kernel_size=1, ncpu = 1, chunksize=50):
        """
        Run the refinement part of indexing over all pixels of the selected phase.
        Updates UBI, U and indexing metrics in pixelmap. 

        Args:
        --------
        pname    : str, phase name. must be in self.phases
        cf       : ImageD11 columnfile. must be sorted by xyi indices
        UBI_col  : str, UBI column name
        hkl_tol  : float; hkl tolerance for refinement
        useInts  : bool (default: False); computes intensity correlation if True
        kernel_size : kernel size for peak selection arround the central pixel. odd integer >=1.
        ncpu, chunksize:  parallelization options. number of workers / chunk size passed to each process
        """
        # pixel selection
        cs = self.phases.get(pname)
        pid = cs.phase_id
        
        sel = self.get_phase_mask(pname)
        pxlist = self.xyi[sel]
        UBIs = self.get(UBI_col)[sel]

        # run refinement
        res = refine_ubi.refine_px_ubis(cf, pxlist, UBIs, ncpu, chunksize, hkl_tol,
                                        cs, useInts, mergeHKL, kernel_size)

        local_indexing.update_xmap(self, pxlist, res, pname, drlv2_max = 1, overwrite = False)        
        stats, fig = refine_ubi.compute_refinement_stats(res)
        return stats, fig
    
    
    def calcGrains(self, pname, Ucol='U', threshold_deg=10, min_grain_size=3, update_grain_ids = True):
        """
        segment grains based on a misorientation threshold and add grain labels and grain boundaries to xmap. 
        See pf3dxrd.orientation.segment_grains for details

        Adds/update the following columns to pixelmap:
        - grain_ids: update column with new grain labels. Does not modify non-selected phases.
        If grain_ids is non empty, adds new grain labels on top of pre-existing ones: e.g. unique grain_ids = -1,0,...n, -> new labels start at n+1
        - gb_mask_<pname> : grain boundary mask for the selected phase
        - gb_angle_<pname> : grain bundary misorientation for the selected phase (in degree)
        - update_grain_ids  : bool, update grain_ids column in xmap if True. Otherwise, just computes grain boundaries
        """
        phase_mask = self.get_phase_mask(pname)

        # run segment_grains from orientation module
        grain_ids, gb_mask, gb_misorientation = orientation.segment_grains(self, pname, Ucol, threshold_deg, min_grain_size)
        grain_ids = grain_ids.flatten()
        gb_mask = gb_mask.flatten()
        gb_misorientation = gb_misorientation.flatten()

        # add grain boundary columns
        self.add_data(gb_mask,f'gb_{pname}')
        self.add_data(gb_misorientation,f'gb_misorientation_{pname}')
    
        # update grain_ids column
        if update_grain_ids:
            if self.grain_ids.max() == -1:
                grain_ids = np.where(grain_ids > 0, grain_ids, -1)
            else:
                grain_ids = np.where(grain_ids > 0, grain_ids + self.grain_ids.max() + 1, -1)
          
            self.update_pixels('grain_ids', grain_ids[phase_mask], selection_mask=phase_mask)
        


    def calcKAM(self, pname, Ucol='U', kernel_size=3, threshold_deg=1, mode='mean', use_numba_acceleration=True):
        """ compute local misorientation (kernel-averaged misorientation) and add results to xmap """
        assert pname in self.phases.pnames, 'phase name not recognized'
        phase_mask = self.get_phase_mask(pname)
        
        kam = orientation.local_misorientation(self, pname, Ucol, kernel_size, threshold_deg, mode, use_numba_acceleration)
        kam = kam.flatten()
        kam[~phase_mask] = 0
        self.add_data(kam,f'KAM_{pname}')


    def calcGROD(self, pname=None, pixel_orientation="U", reference_frame="sample", axis_coordinates="cartesian", degrees=True):
        """ Computes grain reference orientation deviation (GROD) for all grains in self.grains. See orientation.compute_grod for details"""
        if not self.grains.glist:
            print('no grains in xmap.grains. use calcGrain and add_grains_from_map to create a grain list')
            return

        GROD = orientation.compute_GROD(self, pname, pixel_orientation, reference_frame, axis_coordinates, degrees=degrees)
        for k,v in GROD.items():
            if f'GROD_{k}' not in self.titles() or pname is None:
                self.add_data(v, f'GROD_{k}')
            else:
                pm = self.get_phase_mask(pname)
                self.update_pixels(f'GROD_{k}', v[pm], selection_mask=pm)
    
        
    
    def map_grain_prop(self, prop, pname=None, debug=0):
        """ map a grain property (U, UBI, unitcell, grainsize, etc.) taken from grains in grains.dict to the 2D grid.
        For a grain property p, this function creates a new data column 'p_g' in pixelmap and assign the mean grain value
        of the selected property to each pixel of the grain mask on the 2D grid.
        
        FOR GRAIN MISORIENTATION: see compute_GROD
        
        FOR STRAIN/STRESS: To quickly map all six strain / stress components, simply type 'stress" or 'strain'
        as a prop and the function will look for all tensor components and return a single output as a ndarray.
        
        
        Args:
        ------
        prop : grain property to map. Must be in grains attributes 
        pname : phase name to select
        
        New attribute added to pixelmap:
        prop_g : grain property mapped onto the 2D grains masks
        """
        
        # Initialize new array
        #####################################
        array_shape = [ self.grid.nx * self.grid.ny ]  # size of pixelmap
        
        if any([s in prop for s in 'stress,strain,eps,sigma'.split(',')]):   # special case for stress / strain related data
            prop_name = prop+'_g'
            prop_shape = list( self.grains.get_all(prop, pname).shape[1:] )
            array_shape.extend(prop_shape)
            newarray = np.full(array_shape, np.inf)  # default value = inf to avoid confonding zero strain / stress with no data
        else:
            prop_shape = list( self.grains.get_all(prop, pname).shape[1:] )
            prop_name = str(prop)+'_g'  # add g suffix to make it clear it is derived from a grain property
            array_shape.extend(prop_shape)
        
            # special values to initialize grain/phase id: -1. Default: 0
            if any([s in prop for s in 'gid,grain_id,phase_id'.split(',')]):
                init_val = -1
            elif any([s in prop for s in 'I1,J2,P_hyd,von_Mises'.split(',')]):
                init_val = np.inf
            else:
                init_val = 0
            
            # dtype: float (default) or int 
            try:
                isinstance(self.grains.get_all(prop)[0], 'int')
                dtype = 'int'
            except:
                dtype = 'float'
            
            newarray = np.full(array_shape, init_val, dtype = dtype)
            
                
        # update with values from grains in graindict
        #####################################  
        for gi,g in tqdm.tqdm(zip(self.grains.gids, self.grains.glist)):
            if (g.phase != pname) and pname is not None:
                continue
                
            gm = self.grain_ids == gi
            #gm = np.argwhere(gid_map == gi).T[0]  # grain mask
            
            # fill newarray. Different cases depending of prop shape
            if len(prop_shape) == 0:
                newarray[gm] = self.grains.get(prop,gi)
            else:
                newarray[gm,:] = self.grains.get(prop,gi)
    
        # add newarray to pixelmap
        self.add_data(newarray, prop_name)
            
            
                  
    def plot(self, prop, phase=None, label = None, dim=0, save=False, hide_cbar=False, autoscale=False, hist_tails_cut = [2,98],
             smooth=False, mf_size=1, show_gb=False, gb_res_fact = 1, gb_color='k', out=False, **kwargs):
        """ Plot colormap of data in column datacolname using pcolormesh
        
        Args:
        --------
        prop             : str or lambda func. property to plot. Either data column in pixelmap (str), or function of one or multiple columns (labda func)
                         e.g. prop = 'Npks'  (str) ; prop = lambda xmap: xmap.Npks (func)
        dim (int)        : for ndarray of shape (nx*ny,M), M>1, dimension M of the data array to plot
        phase (str)      : select phase to plot. If none, use the full map. required to show grain boundaries
        label (str)      : property label. default is prop (str)
        save (bool)      : save plot (default is False)
        hide_cbar (bool) : hide colorbar from plot (delault is False)
        smooth (bool)    : apply median filter for smoothing
        mf_size (int)    : median filter kernel size. default is 1 
        show_gb (bool)   : overlap grain boundaries. default is False. 
        gb_res_fact / gb_color: controls grain boundaries aspect. see self.add_grain_boundaries
        out (bool)       : return figure as output (default is False)
        autoscale (bool) : automatically adjust color scale to distribution (default is False)
        hist_tails_cut   : percentile thresholds ([low,up]) to cut distribution (for autoscale). Default is [2,98]
        kwargs (dict)    : keyword arguments passed to matplotlib"""
        
        # xmap grid
        nx, ny = self.grid.nx, self.grid.ny
        xb, yb = self.grid.xbins, self.grid.ybins
        
        # select data to plot
        if callable(prop):
            data = prop(self)
            if label is None:
                label = get_lambda_expression(prop)
        else:
            data = self.get(prop)
            if label is None:
                label = prop
        
        if phase is not None:
            mask = self.get_phase_mask(phase)
            default_val = data[self.phase_ids==-1][0]
            data[~mask] = default_val

        # reshape
        if len(data.shape) == 1:
            data2D = data.reshape(nx,ny)
            title = f'{label}'
        else:
            data2D = data[:,dim].reshape(nx,ny)
            title = f'{label}_{dim}'
        
        if smooth:
            data2D = ndi.median_filter(data2D, size=mf_size)
        
        # plot
        fig = pl.figure(figsize=(6,6))
        ax = fig.add_subplot(111, aspect ='equal')
        ax.set_axis_off()
        
        if autoscale:
            m = np.all([data!=0, data!=-1, data!=float('inf')], axis=0)
            low, up = np.percentile(data[m], hist_tails_cut)
            im = ax.pcolormesh(xb, yb, data2D, vmin=low, vmax=up, rasterized=True, **kwargs)
        else:
            im = ax.pcolormesh(xb, yb, data2D, rasterized=True, **kwargs)
        
        # add grain boundaries to plot 
        if show_gb:
            assert phase is not None, "you need to select a phase to show grain boundaries"
            self.add_grain_boundaries(phase, ax=ax, resolution_factor=gb_res_fact, gb_color=gb_color)
        
        ax.set_title(title)
        ax.add_artist(self.grid.scalebar())
        
        # colorbar
        fig.suptitle(self.h5name.split('/')[-1].split('.h')[0], y=.9)
        if 'phase_id' in label:
            cbar = pl.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7, ticks = self.phases.pids)
            cbar.ax.set_yticklabels(self.phases.pnames)
            cbar.ax.set_rasterized(True)
        else:
            cbar = pl.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7, label=label)
            cbar.ax.set_rasterized(True)
            try:
                cbar.formatter.set_powerlimits((-1, 1)) 
            except:  # cbar formatter does not work when LogNOrm is used 
                pass
                
        if hide_cbar:
            fig.suptitle(self.h5name.split('/')[-1].split('.h')[0], y=1.)
        if save:
            self.saveplot(fig, title)
        if out:
            return fig
            
            
            
    def plot_strain_stress(self, datacolname, phase=None, autoscale=True, hist_tails_cut = [2,98], show_gb=False,
                          gb_res_fact=1, gb_color='k', save=False, hide_cbar=False, smooth=False, mf_size=1, out=False, **kwargs):
        """ plot all components of strain / stress tensor in a single figure. The strain/stress column must be in voigt-like format ((6,1) shape)
        
        Args: 
        ---------
        datacolname (str) : name of data array. data in self.datacolname must be a Nx6 array with strain / stress components
                            in the following order: e11,e22,e33,e23,e13,e12
        phase (str)     : Add a mask to select only selected phase. If None, keep the full map. Required for show_gb 
        autoscale (bool): automatically adjust color scale to distribution for each strain / stress component (default is True)
        percentile_cut  : percentile thresholds ([low,up]) to cut distribution (for autoscale). Default is [,98]
        save (bool)     : save plot (default is False)
        hide_cbar (bool): hide colorbar from plot (delault is False)
        smooth (bool)   : apply median filter for smoothing plot
        mf_size (int)   : median filter kernel size. default is 1 
        show_gb (bool)   : overlap grain boundaries. default is False. 
        gb_res_fact / gb_color: controls grain boundaries aspect. see self.add_grain_boundaries
        hist_tails_cut  : percentile thresholds ([low,up]) to cut distribution (for autoscale). Default is [2,98]
        out (bool)      : return figure as output (default is False)
        kwargs (dict)   : keyword arguments passed to matplotlib""" 
        
        nx, ny = self.grid.nx, self.grid.ny 
        xb, yb = self.grid.xbins, self.grid.ybins
        data_array = self.get(datacolname)
        
        if phase is not None:
            mask = self.get_phase_mask(phase)
            data_array[~mask] = np.inf
        
        # figures layout
        fig, ax = pl.subplots(2,3, figsize=(10,7), sharex=True, sharey=True)
        ax = ax.flatten()
        
        if any(['strain' in datacolname, 'eps' in datacolname]):
            titles = 'e11,e22,e33,e23,e13,e12'.split(',')
            main_title = 'strain_components'
            if 'voigt' in datacolname:
                data_array = data_array * np.array([1,1,1,1/2,1/2,1/2])
        elif any(['stress' in datacolname, 'sigma' in  datacolname]):
            titles = 's11,s22,s33,s23,s13,s12'.split(',')
            main_title = 'stress_components'
        else:
            print('data name not recognized. Should contain either "strain"/"eps" or "stress"/"sigma"')
            return
                  
        # loop through strain / stress components and plot them in map
        for i, (a,t) in enumerate(zip(ax, titles)):
            a.set_aspect('equal')
            a.set_axis_off()
            x = data_array[:,i].reshape(nx,ny)
            if smooth:
                x = ndi.median_filter(x, size=mf_size)
            x_u = np.unique(data_array[data_array != float('inf')])  # select unique values to get distribution across grains  
    
            # plots
            if autoscale:
                low, up = np.percentile(x_u, (hist_tails_cut[0],hist_tails_cut[1]))
                norm=pl.matplotlib.colors.CenteredNorm(vcenter=np.median(x_u), halfrange=up)
                im = a.pcolormesh(xb, yb, x, norm=norm, rasterized=True, **kwargs)
            else:
                im = a.pcolormesh(xb, yb, x, rasterized=True, **kwargs)
            a.set_title(t)
            
            if show_gb:
                assert phase is not None, "you need to select a phase to show grain boundaries"
                self.add_grain_boundaries(phase, ax=a, resolution_factor=gb_res_fact, gb_color=gb_color)
            
            # colorbar
            if not hide_cbar:
                cbar = pl.colorbar(im, ax=a, orientation='vertical', pad=0.04, shrink=0.7)
                cbar.formatter.set_powerlimits((-1, 1)) 
                cbar.ax.set_rasterized(True)
            
        # Adjust layout
        fig.tight_layout()
        fig.suptitle(f'{main_title} – {self.dsname}', y=1.0)

        if save:
            self.saveplot(fig, main_title)
        if out:
            return fig
            
        
    ## To remove: replace with function to compute distribution + summary statistics for a selected property of the map    
    def hist_strain_stress(self, datacolname, percentile_cut=[2,98], nbins=100, save=False, out=False, **kwargs):
        """ plot histogram for all components of strain / stress tensor (voigt notation)
        
        Args: 
        --------
        datacolname (str)    : name of data array. data in self.datacolname must be a Nx6 array with strain / stress components
                         in the following order: e11, e22, e33, e23, e13, e12
        percentile_cut : percentile thresholds ([low,up]) to cut distribution and adjust histogram width Default is [2,98]
        save (bool)    : save plot (default is False)
        nbins (int)    : number of bins in the histogram. Default is 100
        out (bool)     : return figure as output (default is False)
        kwargs (dict)  : keyword arguments """ 
        
        data_array = self.get(datacolname)
        # figures layout
        fig, ax = pl.subplots(2,3, figsize=(10,6))
        ax = ax.flatten()
        
        if any(['strain' in datacolname, 'eps' in datacolname]):
            titles = 'e11,e22,e33,e23,e13,e12'.split(',')
            main_title = 'strain_components_histogram'
            if 'voigt' in datacolname:
                data_array = data_array * np.array([1,1,1,1/2,1/2,1/2])
        elif any(['stress' in datacolname, 'sigma' in  datacolname]):
            titles = 's11,s22,s33,s23,s13,s12'.split(',')
            main_title = 'stress_components_histogram'
        else:
            print('data name not recognized. Should contain either "strain"/"eps" or "stress"/"sigma"')
            return
          
        for i, (a,t) in enumerate(zip(ax, titles)):
            x = data_array[:,i]
            x_c = x[x != float('inf')]
            low, up = np.percentile(x_c, (percentile_cut[0],percentile_cut[1]))
            bins = np.linspace(low, up, nbins)
            h = a.hist(x_c, label=t, bins=bins, **kwargs)
            a.vlines(x=np.median(x_c), ymin=0, ymax=h[0].max(), colors='r', label='median')
            a.set_title(t)
            a.set_xlim(low,up)
            a.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            a.legend(loc='upper left', fontsize=7)
                
        fig.tight_layout()
        fig.suptitle(f'{main_title} – {self.dsname}', y=1.0)
        if save:
            self.saveplot(fig, main_title)
        if out:
            return fig
            

    def get_ipf_orientations(self, phase, datacolname='U', ipf_directions = [(0,0,1)], ellipsoid=False, smooth = False, smooth_kernel_size = 3, smooth_threshold = 10):
        """get IPF maps in selected sample directions, entered as a list of (x,y,z) tuples. Returns dictionnary of rgb maps for each direction.
        If ipf_direction is set to 'xyz' (str), returns ipf colors in orthogonal xyz sample basis."""

        assert phase in self.phases.pnames, 'phase name not recognized'
        
        #map grid
        nx, ny = self.grid.shape
        xb, yb = self.grid.xbins, self.grid.ybins
    
        # phase symmetry
        cs =  self.phases.get(phase)
        if ellipsoid:
            cell = np.array([1,1,1,90,90,90])
            uc = ImageD11.unitcell.unitcell(cell, symmetry=16)
            sym = oq.symmetry.D2h
        else:
            uc = ImageD11.unitcell.unitcell(cs.cell, cs.spg_no)
            sym = cs.orix_phase.point_group

        # get pixel orientation
        Umats = self.get(datacolname).copy()
        
        m_selec = (self.phase_ids == cs.phase_id) & (self.nindx > 0)  #  mask for unindexed / bad pixels
        
        if smooth:
            Umats_smooth = orientation.local_orientation_smooth_orix(Umats.reshape(nx,ny,3,3), sym, kernel_size=smooth_kernel_size,
                                                                     global_mask = m_selec.reshape(nx,ny), local_mask=None,
                                                                     threshold_deg=smooth_threshold)
            Umats = Umats_smooth.reshape(nx*ny,3,3)
        ori = uc.get_orix_orien_fast(Umats)
       
        # get rgb map from ipf orientations
        if ipf_directions == 'xyz':
            rgb_maps = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1])}
            
        else:
            rgb_maps = {tuple(axis): np.asarray(axis) for axis in ipf_directions}

        for key, axis in rgb_maps.items():
            rgb = uc.get_ipf_colour_from_orix_orien(ori, axis=axis)
            rgb[~m_selec, :] = 0  # set non-selected pixels to black
            rgb_maps[key] = rgb

        return rgb_maps
    
    
    def plot_ipf_orientation(self, phase, datacolname='U', ipf_directions = [(0,0,1)], ellipsoid = False, show_color_key = True,
                             smooth=False, smooth_kernel_size = 3, smooth_threshold = 10,
                             show_gb=False, gb_res_fact=1, gb_color='w', save=False, hide_cbar=False, out=False, **kwargs):
        
        """ Plot IPF orientation color maps 
        
        Args:
        --------
        phase (str)    : name of the phase to plot. must be in self.phases
        datacolname (str)    : name of orientation data array. Default is 'U'. Must be a ndarray with shape (N,3)
        ipf_directions (array) : direction for the ipf colorkey in the laboratory reference frame.
                         must be a 3x1 vector [x,y,z]. Default: z-vector [0,0,1]
        smooth (bool)  : apply orientation smoothing before computing rgb colors
        smooth_kernel_size, smooth_threshold : see orientation.local_orientation_smooth_orix
        ellipsoid (bool) : set symmetry to one of a triaxial ellipsoid (orthorombic, mmm).
                           For ipf color map of strain-stress principal components orientation
        show_color_key (bool): plot ipf color key is separate figure. default is True
        show_gb (bool)   : overlap grain boundaries. default is False. 
        gb_res_fact / gb_color: controls grain boundaries aspect. see self.add_grain_boundaries
        save (bool)    : save plot (default is False)
        hide_cbar (bool) : hide colorbar from plot (delault is False)
        out (bool)     : return fig output (default is False)
        kwargs (dict)  : keyword arguments passed to matplotlib"""
        
        #map grid
        nx, ny = self.grid.nx, self.grid.ny
        xb, yb = self.grid.xbins, self.grid.ybins
    
        # phase symmetry + color key
        cs =  self.phases.get(phase)
        if ellipsoid:
            sym = oq.symmetry.D2h
            ipf_key = opl.IPFColorKeyTSL(sym, direction=ovec.Vector3d.zvector)
        else:
            sym = cs.orix_phase.point_group.laue
            cs.get_ipfkey(direction = ovec.Vector3d.zvector)
            ipf_key = cs.ipfkey
        # get rgb maps
        rgb_maps = self.get_ipf_orientations(phase, datacolname, ipf_directions, ellipsoid, smooth, smooth_kernel_size, smooth_threshold)

        # Figure ---- Layout: max 3 columns ----
        n_maps = len(rgb_maps)
        n_cols = min(3, n_maps)
        n_rows = np.ceil( (n_maps)/n_cols).astype(int)

        fig, axes = pl.subplots(n_rows, n_cols, figsize=(5*n_cols,5*n_rows))
        if n_maps == 1:
            axes = np.array([axes])
            
        axes = axes.ravel()
        for ax in axes:
            ax.set_axis_off()

        for ax, (ipfdir, rgb) in zip(axes, rgb_maps.items()):

            rgb = np.asarray(rgb)
            # Normalize if RGB is 0–255
            if rgb.dtype != float or rgb.max() > 1.0:
                rgb = rgb.astype(float) / 255.0

            im = ax.pcolormesh(xb, yb, rgb.reshape(nx,ny,3), rasterized=True, **kwargs)
            ax.set_aspect("equal")
            ax.set_title(f"{phase} IPF {str(ipfdir)}")
            
            if not hide_cbar:
                ax.add_artist(self.grid.scalebar())
            if show_gb:
             self.add_grain_boundaries(phase, ax=ax, resolution_factor = gb_res_fact, gb_color = gb_color) 

        # add color key
        if show_color_key:
            pl.matplotlib.rcParams.update({'font.size': 8})
            fig_key = pl.figure(figsize=(3,3))
            ax_key = fig_key.add_subplot(111, projection="ipf", symmetry=sym)
            ax_key.plot_ipf_color_key(ipf_key)
            #ipf_key.plot(ax_key)
            pl.matplotlib.rcParams.update({'font.size': 10})

        fig.suptitle(self.dsname, y=1.0)    
        fig.tight_layout()
        if save:
            self.saveplot(fig, f'{phase}_ipf_maps')
            if show_color_key:
                fname = os.path.basename(self.h5name).replace('.h5', f'{phase}_ipfkey.svg')
                fig_key.savefig(fname)
        if out:
            return fig
            

    def saveplot(self, fig, title):
        """ save xmap plot to default location within the dataset directory """
        savedir = os.path.join(os.path.dirname(self.h5name), 'XMAP_PLOTS')    # sub-folder to save plots: create it if does not exist
        subprocess.run(f'mkdir -p {savedir}'.split(' '), check=True)         
        
        fname = os.path.basename(self.h5name).replace('.h5', f'_{title}.svg')
        fig.savefig('/'.join([savedir,fname]), format='svg', dpi=250)
        
        print(f'{title} saved to {savedir}')
        
        
    def add_grain_boundaries(self, pname, ax=None, resolution_factor=1, gb_color='k', **kwargs):
        """
        add grain boundary overlay for selected phase on existing plot
        
        Args:
        -------
        pname  : phase name
        ax     : figure axis on which grain boundaries shall be added
        resolution_factor (int) : increases the grid size for the grain boundary mask -> thinner grain boundaries
        gb_color : grain boundary color (default = black)
        **kwargs : other options to pass for plotting. 
        """
        #  create new figure if no axis is entered
        if ax is None:
            fig = pl.figure(figsize=(6,6))
            ax = fig.add_subplot(111, aspect='equal')
            ax.set_axis_off()
        
        # compute grain boundaries if not already done
        if f'gb_{pname}' not in self.titles():
            print(f'no grain boundaries for phase {pname}.Computing them now, using the default orientation column (U) and angle threshold (10°)')
            self.calcGrains(pname, update_grain_ids=False) 
        
        # get 2D grain boundary mask, rescale it and skeletonize to get skinny boundaries
        # aliases
        nx, ny = self.grid.nx, self.grid.ny
        xb, yb = self.grid.xbins, self.grid.ybins
        # 2D masks
        gb2D = self.get(f'gb_{pname}').reshape(nx,ny)
        gb2D_resized = skimage.transform.resize(gb2D, (resolution_factor*nx, resolution_factor*ny), order=0,
                                                anti_aliasing=False, preserve_range=True).astype(np.uint8)

        gb2D_skel = skimage.morphology.skeletonize(gb2D_resized)
        
        # define resized grid for the resized grain boundary mask
        xgb = np.linspace(xb.min()-1, xb.max(),gb2D_skel.shape[0])
        ygb = np.linspace(yb.min()-1, yb.max(),gb2D_skel.shape[1])
        
        # colormap for grain boundaries
        kwargs['vmin'] = 0.1
        if 'cmap' not in kwargs.keys():
            cmap = mcolors.ListedColormap([gb_color])
            cmap.set_under('none')
            kwargs['cmap'] = cmap
        ax.pcolormesh(xgb, ygb, gb2D_skel, rasterized=True, **kwargs)
            
            
    def save_to_hdf5(self, h5name=None, save_mode='minimal',  save_mode_grains_dict = 'minimal', debug=0):
        """ 
        save pixelmap to hdf5 format
        
        Args:
        --------
        h5name : hdf5 file name. If None, name in self.h5name will be used. default is None
        save_mode : minimal / full. If minimal, drops all columns computed from grains and columns related to strain and stress
        save_mode_grains_dict: minimal / full. If minimal, drop hkl and etasigns properties (if present), which take a lot of space
        """
        # save path
        if h5name is None:
            try:
                h5name = self.h5name
                h5name[0]
            except:
                print("please enter a path for the h5 file")
        
        with h5py.File(h5name, 'w') as f:
            
            f.attrs['h5path'] = h5name
            
            # 1 - Save grid ('grid' group)
            #############
            grid_group = f.create_group('grid')
            
            attr = 'pixel_size', 'pixel_unit'
            for item in self.grid.__dict__.keys():
                if item in attr:
                    grid_group.attrs[item] = self.grid.__getattribute__(item)
                else:
                    data = self.grid.__getattribute__(item)
                    grid_group.create_dataset(item, data = data, dtype = int) 
            
            # 2 - Save phases ('phases' group)
            ##############
            phases_group = f.create_group('phases')
            for pname, pid in zip(self.phases.pnames, self.phases.pids):
                # create a new group for each phase
                phase = phases_group.create_group(pname)
                cs = self.phases.get(pname)
                phase.attrs.update({ 'pid':pid})
                phase.attrs.update({'cif_path':str(cs.cif_path)})
                try:
                    phase.create_dataset('cif_file', data=cs.cif_file, dtype=h5py.string_dtype())
                except:
                    print('error in saving cif file for phase', pname)
                
            # 3 - save grains
            ###############
            if save_mode_grains_dict == 'minimal':
                skip =  ['hkl', 'etasigns']
            else:
                skip = None
            save_grains_dict(self.grains.dict, h5name, skip = skip)

            # 4 - Save other data
            skip = ['grid', 'xi', 'yi', 'phases', 'pksind', 'h5name', 'dsname', 'grains']  # things to skip
            
            for item in self.__dict__.keys():
                if item in skip:
                    continue
                if save_mode == 'minimal' and any([s in item for s in ['_g','strain','stress','eps','sigma']]):
                    continue
                data = self.__getattribute__(item)
                if debug:
                    print(item) 
                f.create_dataset(item, data = data, dtype = type(data.flatten()[0]))

        print("Pixelmap saved to:", h5name)
        


#############################
def get_lambda_expression(func):
    """
    help function to extract the expression from a lambda function as a string, removing the object reference prefix (e.g., 'xmap.').
    """
    if callable(func):
        # Get the source code of the lambda function
        source = inspect.getsource(func).strip()
        # Extract the part after `lambda` parameters and the colon
        match = re.search(r'lambda\s+[\w\s,]*:\s*(.*)', source)
        if match:
            expression = match.group(1).strip()
            # Remove any object reference like 'xmap.' from the expression
            # This removes patterns like 'xmap.attr1' -> 'attr1'
            expression = re.sub(r'\b\w+\.', '', expression)
            return expression
    return None


# Save / load functions
##########################    
def load_from_hdf5(h5name, debug=0):
    """ load pixelmap from hdf5 file"""
    with h5py.File(h5name, 'r') as f:
        # Load grid information 
        xbins  = f['grid/xbins'][()]
        ybins  = f['grid/xbins'][()]
        pxsize = f['grid'].attrs['pixel_size']
        pxunit = f['grid'].attrs['pixel_unit']

        # Load phases information : names ids, cif_path, cif_file (stored as list of strings)
        pnames, pids, cif_paths, cif_files = [], [], [], []
        for pname, phase in f['phases'].items():
            pid = phase.attrs['pid']
            cif_path = phase.attrs['cif_path']
            
            try:    
                cif_file_bytes = phase['cif_file'][()]
                cif_file = [c.decode('utf-8') for c in cif_file_bytes]   # strings encoded, need to be decoded
            except:
                cif_file = '_'
                
            pnames.append(pname)
            pids.append(pid)
            cif_paths.append(cif_path)
            cif_files.append(cif_file)
        
        if debug:
            print(f'cif paths {cif_paths} \n files {cif_files} \n pnames {pnames} \n pids {pids}')
            
        # Load grains
        if 'grains' in list(f.keys()):
            grainsdict = load_grains_dict(h5name)
        else:
            grainsdict = {}
    
        # Load other data
        skip = ['grid', 'phases', 'grains']
        data = {}
        for item in f.keys():
            if item in skip:
                continue
            data[item] = f[item][()]

    # Create a new Pixelmap object 
    pixelmap = Pixelmap(xbins, ybins, h5name=h5name)
    
    # update grid
    pixelmap.grid.pixel_size = pxsize
    pixelmap.grid.pixel_unit = pxunit
    
    # Add phases to Pixelmap
    for pname, pid, cpath, cfile in zip(pnames, pids, cif_paths, cif_files):
        if pname == "notIndexed":
            continue
        
        # if cif_path is valid, load crystal structure from there; otherwise, try to load it from saved cif file
        try:
            cs = crystal_structure.CS(pname,pid,cpath)
        except Exception as e:
            print(f'cif path for phase {pname} invalid. Loading from saved file')
            crystal_structure.list_to_cif(cfile, 'tmp')
            cs = crystal_structure.CS(pname,pid,'tmp')
            os.remove('tmp')
    
        if cs is not None:
            pixelmap.phases.add_phase(pname, cs)
                
    # Add data
    for d in data.keys():
        pixelmap.add_data(data[d], d)
    # Add grainsdict
    pixelmap.grains.dict = grainsdict
    pixelmap.grains.gids = list(grainsdict.keys())
    pixelmap.grains.glist = list(grainsdict.values())
    
    # remove tmp files possibly created when loading phases
    if os.path.exists('tmp') and not os.path.isdir('tmp'):
            os.remove('tmp')

    return pixelmap




def save_grains_dict(grainsdict, h5name, skip = None, debug=0):
    """ save grain dictionnary to hdf5."""
    
    with h5py.File( h5name, 'a') as hout:
        # Delete the existing 'grains' group if it already exists
        if 'grains' in hout:
            del hout['grains']
            
        grains_grp = hout.create_group('grains')

        for i,g in grainsdict.items():
            gr = grains_grp.create_group(str(i))    
            gprops = [p for p in list(g.__dict__.keys()) if not p.startswith('_')]  # list grain properties, skip _U, _UB etc. (dependent)
            
            if debug:
                print(gprops)
            
            for p in gprops:
                if p in skip:
                    continue
                attr = g.__getattribute__(p)
                if attr is None:   # skip empty attributes
                    continue
                # find data type + shape
                if np.issubdtype(type(attr), np.integer):
                    dtype = 'int'
                    shape = None
                elif np.issubdtype(type(attr), np.floating):
                    dtype = 'float'
                    shape = None
                elif isinstance(attr, str):
                    dtype = str
                    shape = None
                else:
                    attr = np.array(attr)
                    shape = attr.shape
                    try:
                        dtype = type(attr.flatten()[0])
                    except:    # occurs if attr is empty
                        dtype = float
                
                if debug:
                    print(p,dtype)
                # save arrays as datasets and othr data as attributes
                if shape is not None: 
                    gr.create_dataset(p, data = attr, dtype = dtype) 
                else:
                    gr.attrs.update({ p : attr})


def load_grains_dict(h5name):
    grainsdict = {}
    with h5py.File(h5name,'r') as f:
        if 'grains' in list(f.keys()):
            grains = f['grains']
        else:
            grains = f
            
        gids = list(grains.keys())
        gids.sort(key = lambda i: int(i))
        
        # loop through grain ids and load data
        for gi in gids:
            gr = grains[gi]
            # create grain from ubi
            try:
                g = ImageD11.grain.grain(gr['ubi'])
            except Exception as e:
                print(f'loading grain {gi} failed')
                continue
            # load other properties
            for prop, vals in gr.items():
                if prop == 'ubi':
                    continue
                ary = vals[()]
                setattr(g, prop, ary)
            # add grain attributes
            for attr, val_attr in gr.attrs.items():
                setattr(g, attr, val_attr)
                        
            # add grain to grainsdict
            grainsdict[int(gi)] = g
            
    return grainsdict
    
    
    
def create_from_dataset(ds, h5name = None, pixel_unit=None):
    """ create new pixelmap and initialize the grid from dataset information. pixel unit in µm by default"""
    
    # bins
    xb = yb = np.arange(len(ds.ybinedges))
    
    # initialize pixelmap
    if h5name is None:
        h5name = os.path.join(os.getcwd(), ds.dsname+'_xmap.h5')
    xmap = Pixelmap(xb, yb, h5name = h5name)
    
    # pixel size and unit
    if pixel_unit is not None:
        xmap.grid.pixel_size = ds.ystep
        xmap.grid.pixel_unit = pixel_unit
    
    elif 'frelon' in ds.detector:
        xmap.grid.pixel_size = ds.ystep * 1000
        xmap.grid.pixel_unit = 'µm'
    else:
        xmap.grid.pixel_size = ds.ystep
        xmap.grid.pixel_unit = 'µm'

    return xmap
        
        
