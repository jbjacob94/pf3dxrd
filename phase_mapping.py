import os, sys, copy
from tqdm import tqdm
import concurrent.futures, multiprocessing
import numpy as np, pylab as pl

import ImageD11.refinegrains, ImageD11.columnfile, ImageD11.parameters

from pf_3dxrd import utils, peak_mapping, crystal_structure, pixelmap


class PhaseMapper:
    """
    A class to perform phase mapping on a 2D pixel grid. See corresponding publication [add doi when published]
    """
    # Init
    ##########################
    def __init__(self, cf, ds, xmap,  CS_list=None):
        
        # data
        self.peakfile = cf
        self.dataset = ds
        self.pixelmap = xmap
        self.npks = cf.nrows
        self.wl = self.peakfile.parameters.get('wavelength')
        self.sortkey = None
        self.check_peakfile()
        
        # phase structures
        self.phases = self.PHASES()
        if CS_list is not None:
            for cs in CS_list:
                self.phases.add_phase(cs.name, cs) 
        
        # mapping parameters
        self.minpks = 50
        self.min_confidence = 0.1
        self.kernel_size = 1
        self.ncpu = len(os.sched_getaffinity( os.getpid() ))
        self.chunksize = 500
        
        # results and stats
        self.res = {}
        self.stats_phase_masks = {}       # peak fraction in each phase masks, including non-assigned peaks and proportion of overlaps)
        self.stats_labeled_peaks = {}     # peak fraction for each phase after phase mapping
        
        
    def __str__(self):
        return f"PhaseMapper:\n peaks to map:{self.npks}\n {self.phases}\n phase_ids:{self.phases.pids}\n {self.pixelmap.grid}\n minpks: {self.minpks}\n min_confidence: {self.min_confidence}\n kernel_size: {self.kernel_size}"     
    
    
    def get(self,prop):
        return self.__getattribute__(prop)
    
    
    def copy(self):
        return copy.deepcopy(self)
    
    
    def check_peakfile(self):
        """ 
        make sure peakfile if ok: contains (xs,ys) + (xi,yi,xyi) columns, sorted by tthc. Also add rescaled intensity (Lorentz factor)
        """
        
        cf = self.peakfile
        
        assert all(['xs' in cf.titles, 'ys' in cf.titles]), 'peakfile has no coordinates in sample space. Please run friedel_pairs.update_geometry_s3dxrd'
        
        if 'xyi' not in self.peakfile.titles:
            print('no xyi column in peakfile. Adding pixel index...')
            peak_mapping.add_pixel_labels(cf, self.dataset)
        
        print('sorting by two-theta...')
        cf.sortby('tthc')
        self.sortkey = 'tthc'
            
        print('rescaling intensity...')        
        lf = ImageD11.refinegrains.lf(cf.tthc, cf.eta)  # lorentz factor for intensity scaling
        cf.addcolumn(cf.sum_intensity * lf, 'sumI')
       
    
        
    # PHASE subclass
    ##########################
    class PHASES:
        """ sub-class to store information on crystal structures. Modified from pixelmap"""
        def __init__(self):
            self.pnames = []
            self.pids = []
           
        
        def __str__(self):
            return f"phases: {self.pnames}"
        
        
        def get(self,attr):
            return self.__getattribute__(attr)
            
            
        def add_phase(self, pname, cs):
            """ add phase to pixelmap.phases. 
            
            Parameters
            ----------
            pname (str) : phase name
            cs          : crystal_structure.CS object
            """
            # if this phase name already exists, delete it
            if pname in self.pnames:
                print(pname, ': There is already a phase with this name in self.phases. Will overwrite it.')
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
            
        
            
    
    # Methods
    ##########################################################################################
    ##########################################################################################
        
    def add_phases_from_pixelmap(self):
        """ Inherit phases from self.pixelmap if they have already been defined there """
        for pname in self.pixelmap.phases.pnames:
            if pname == 'notIndexed':
                continue
            self.phases.add_phase(pname, self.pixelmap.phases.get(pname))
            
    
    def find_strongest_bragg_peaks(self, pname, min_tth=0, max_tth=25, Nmax = 30, Imin=0.1, prominence = 0.1, doplot=False):
        """ 
        Simulate powder spectrum (Intensity vs two-theta) and find the N-stongest Bragg peaks for phase "pname". 
        Args:
        -------
        pname (str): phase name
        min_tth, max_tth (floats) : two-theta range over which diffraction spectrum is computed
        Imin       : Minimum intensity threshold to consider a local maximum as a peak
        prominence : Minimum prominence to consider a local maximum as a peak
        Nmax       : If the result of peaksearch contains more than Nmax peak, keep only the Nmax strongest
        doplot : if True, plot two-theta powder spectrum with peak positiosn for the selected phase
        
        See also: crystal_structure.compute_powder_spec, crystal_structure.find_strongest_peaks
        """
        cs = self.phases.get(pname)
        cs.compute_powder_spec(self.wl, min_tth, max_tth, doplot=False)
        cs.find_strongest_peaks(Imin, Nmax, prominence, doplot=doplot)
        
        
    def find_strongest_bragg_peaks_all(self, min_tth=0, max_tth=25, Nmax = 30, Imin=0.1, prominence = 0.1, doplot=False):
        """
        run find_strongest_bragg_peaks for all phases using the same parameters for peak search
        
        See also: find_strongest_bragg_peaks
        """
        for pname in self.phases.pnames:
            self.find_strongest_bragg_peaks(pname, min_tth, max_tth, Nmax, Imin, prominence, doplot)
            
            
    def compute_phase_mask(self, pname, tth_max, tth_tol):
        """
        compute boolean selection mask for phase self.phases.pname.
        This mask select peaks in self.peakfile based on a two-theta distance threshold with computed Bragg peaks of the selected phase. 
        All peaks at a distance less than tth_tol from any theoretical Bragg peak of the selected phase are retained. 
        
        Args:
        -------
        pname (str): phase name
        tth_max : maximum two-theta for peak selection in mask
        tth_tol : tolerance threshold around theoretical tth peaks to add a peak to the selection mask
        """
        # select cs object  check Bragg peaks have been computed
        cs = self.phases.get(pname)
        assert hasattr(cs, 'strong_peaks'), 'No Bragg peaks found for this phase. Please run self.compute_phase_mask'
        
        # check peakfile is sorted by tthc (required for select_tth_rings)
        mask = utils.select_tth_rings(self.peakfile, cs.strong_peaks[0], tth_tol, tth_max, is_sorted = self.sortkey == 'tthc')
        pks_frac = np.count_nonzero(mask) / self.npks
        
        print(f'pks fraction {cs.name}: {pks_frac:.4f}')  # prop of peaks assigned to this phase
        self.peakfile.addcolumn(mask, pname)
        self.stats_phase_masks[pname] = pks_frac
        self.sortkey = 'tthc'  # reset sortkey
        
    
    def compute_mask_overlaps(self):
        """
        add masks corresponding to overlapping peaks (selected in multiple phase masks) and non-assigned peaks (not selected by any phase)
        """
        
        allphases = np.concatenate([self.peakfile.getcolumn(p) for p in self.phases.pnames])
        allphases = allphases.reshape(len(self.phases.pnames),self.npks)
        sum_allphases = np.sum(allphases, axis=0)
        
        self.peakfile.addcolumn(sum_allphases > 1, 'overlap')
        self.peakfile.addcolumn(sum_allphases > 0, 'assigned')

        print(f'Total pks fraction assigned: {np.count_nonzero(self.peakfile.assigned)/self.npks:.4f}')
        print(f'fraction overlapping: {np.count_nonzero(self.peakfile.overlap)/self.npks:.4f}')
            
    
    def compute_phase_mask_all(self, tth_max, tth_tol):
        """ 
        compute boolean selection mask for all phase in self.phases, using the same tth_tol and tth_max
        
        See also: compute_phase_mask
        """
    
        for pname in self.phases.pnames:
            self.compute_phase_mask(pname, tth_max, tth_tol)
        print('\n================')
        self.compute_mask_overlaps()
    
    
    def prepare_for_phase_labeling(self):
        """ 
        make peakfile ready for phase mapping:
        - check phase masks have been computed
        - initialize phase_id column
        - sort by xyi and convert peakfile.xyi to int (faster index search)
        - create list of unique xyi indexes to search through
        """
        # check phases masks are here
        assert all([t in self.peakfile.titles for t in self.phases.pnames]), 'some phase masks are missing'
        
        self.peakfile.addcolumn(np.full(self.npks, -1), 'phase_id')
        
        # prepare xyi array
        self.peakfile.xyi = self.peakfile.xyi.astype(int)
        if self.sortkey != 'xyi':
            self.peakfile.sortby('xyi')
            self.sortkey = 'xyi'
        
        self.xyi_uniq = np.unique(self.peakfile.xyi)
        
        np.seterr(divide = 'ignore', invalid = 'ignore')
        
        print('Ready for phase labeling!')

        
    def best_phase_match(self, px):
        """ 
        Find the best-matching phase on pixel px, among the list of phases in self.phases.
        The best-matching phase is the one gathering the highest total cumulated intensity over pixel px
        based on pre-computed boolean masks. Also returns confidence criteria to evaluate how good the match is.
        Better description of the method and definition of confidence criteria in the associated publication.
        
        Args:
        --------
        px : pixel index (xyi)
        
        Returns:
        --------
        phase_id     : integer value corresponding to the index of the best phase in self.phases.pids. -1 if no phase assigned
        completeness : Proportion of total intensity (between 0 and 1) matched by the best phase on pixel px. 0 if no match found
        uniqueness   : Proportion of intensity (between 0 and 1) uniquely matched by the best phase (ie. no match found with any other
                       phase for this subset). 0 if no match found
        confidence   : normalized confidence index. product of completeness x uniqueness normalized to the number of phases.
        pksinds      : list of peak indexes corresponding to the phase assigned on pixel px. needed to update phase_id column in peakfile
        """
        
        # preliminary checks + initialization
        ##########################################
        
        # check peakfile has been sorted 
        assert all([self.sortkey == 'xyi', 'phase_id' in self.peakfile.titles]), 'peakfile is not ready. Run prepare_for_phase_labeling'
        assert len(self.phases.pnames) > 0, 'no phases to match in self.phases'
        assert all([t in self.peakfile.titles for t in self.phases.pnames]), 'some phase masks are missing'
        
        default_output = -1, 0, 0, 0, []  # default output returned if no best match can be found
        # some aliases
        cf = self.peakfile
        pnames = self.phases.pnames
        pids = self.phases.pids
        n = len(pnames)
        
        # peak selection for pixel px. 
        ##########################################
        s = peak_mapping.pks_from_px(cf.xyi, px, kernel_size=self.kernel_size)    # peak selection for phase matching
        s_px = peak_mapping.pks_from_px(cf.xyi, px, kernel_size=1)                # peak selection for central px
        
        if len(s) == 0:
            return default_output
        
        # minpks filter: if not enough peaks associated with at least one phase, return default output
        npk = np.array([np.count_nonzero(cf.getcolumn(p)[s]) for p in pnames])
        if max(npk) < self.minpks:
            return default_output
        
        # Find best match + compute confidence criteria
        ##################################
        # cumulated intensity for each phase ki
        sum_I_ki = np.array([sum(cf.getcolumn(p)[s] * cf.sumI[s]) for p in pnames])   
        # cumulated intensity of non-overlapping peaks for each phase ki
        sum_I_ki_no_overlap = np.array([sum(cf.getcolumn(p)[s] * np.invert(cf.overlap[s]) * cf.sumI[s] ) for p in pnames])  
        # total cumulated intensity over pixel px
        sum_I_px = sum(cf.sumI[s])  
        sum_I_px_assigned = sum(cf.sumI[s] * cf.assigned[s])    

        # completeness and uniqueness for each phase
        completeness = sum_I_ki / sum_I_px
        uniqueness   = sum_I_ki_no_overlap / sum_I_ki
        
        # nan filter: return default output if all phases yield 'nan' for completeness
        if np.all(np.isnan(completeness)):
            return default_output
    
        # best match
        best_i = np.nanargmax(completeness)    # index of best-matching phase
        pid = pids[best_i]                    # phase_id for best-matching phase
    
        u_best = uniqueness[best_i]
        c_best = completeness[best_i]
        
        if n == 1:
            conf_ind_best = c_best * u_best
        else:
            conf_ind_best = 1/(n-1) * ( n * c_best -1) * u_best
        
        # min confidence filter: return default output if confidence is too low
        if conf_ind_best < self.min_confidence:  
            return default_output
    

        # find pksinds
        pks = cf.getcolumn(pnames[pids.index(pid)])[s_px]   # bool array to select only peaks from the indexed phase over pixel px
        pksinds = s_px[pks.astype(bool)]                        # pks index of selected peaks 
        
        return pid, c_best, u_best, conf_ind_best, pksinds
        
    
    
    def results_to_xmap(self):
        """ extract results in self.res dict and add them to pixelmap"""
    
        # initialize new data columns
        xmap = self.pixelmap
        xmap.add_data(np.full(xmap.xyi.shape, 0, dtype=np.uint16), 'Npks')
        xmap.add_data(np.full(xmap.xyi.shape, 0, dtype=np.float64), 'phase_label_confidence')
        xmap.add_data(np.full(xmap.xyi.shape, 0, dtype=np.float64), 'completeness')
        xmap.add_data(np.full(xmap.xyi.shape, 0, dtype=np.float64), 'uniqueness')
        xmap.phase_id = np.full(xmap.xyi.shape, -1)

        # update xmap with results
        for i,col in enumerate(['phase_id','completeness', 'uniqueness', 'phase_label_confidence']):
            dat = np.array([self.res[px][i] for px in self.xyi_uniq])
            xmap.update_pixels(self.xyi_uniq, col, dat)
            
        npks = np.array([len(self.res[px][4]) for px in self.xyi_uniq])
        xmap.update_pixels(self.xyi_uniq, 'Npks', npks)
        
        # loop through results, update xmap and phase label in peakfile
        self.peakfile.phase_id = np.full(self.peakfile.nrows, -1)
        for i,px in enumerate(tqdm(self.xyi_uniq)):
            if npks[i] == 0:
                continue
            pks = self.res[px][4]
            self.peakfile.phase_id[pks] = self.res[px][0]
    
        print(xmap) 
    
    
    def get_stats_raw_masks(self):
        """ print fraction of intensity / nb of peaks in each phase mask """
        titles = self.phases.pnames+ ['overlap','assigned']
        stats_dict = {t:[] for t in titles}
    
        print('peak fractions in raw selection masks \n----------') 
        for t in titles:
            mask = self.peakfile.getcolumn(t)
            pks_frac = np.count_nonzero(mask) / self.npks
            print(f'{t}: {pks_frac:.4f}')  # prop of peaks assigned to this phase
            stats_dict[t].append(pks_frac)
    
        print('====================')
        print('fraction of total intensity in masks \n----------')
    
        sumItot = np.sum(self.peakfile.sumI)
        for t in titles:
            mask = self.peakfile.getcolumn(t)
            Ints_frac = np.sum(self.peakfile.sumI[mask]) / sumItot
            print(f'{t}: {Ints_frac:.4f}')  # prop of peaks assigned to this phase
            stats_dict[t].append(Ints_frac)
                           
        self.stats_phase_masks = stats_dict
        
        
    def get_stats_labeled_peaks(self):
        """ print fraction of intensity / nb of peaks for each phase after phase mapping has run"""
        titles = self.phases.pnames+ ['assigned']
        stats_dict = {t:[] for t in titles}
    
        print('fraction of labeled peaks \n----------') 
        for t in titles:
            if t == 'assigned':
                m = self.peakfile.phase_id != -1
            else:
                pid = self.phases.pids[self.phases.pnames.index(t)]
                m = self.peakfile.phase_id == pid

            pks_frac = np.count_nonzero(m) / self.npks
            print(f'{t}: {pks_frac:.4f}')  # prop of peaks assigned to this phase
            stats_dict[t].append(pks_frac)
    
        print('====================')
        print('fraction of total intensity \n----------')
    
        sumItot = np.sum(self.peakfile.sumI)
        for t in titles:
            if t == 'assigned':
                m = self.peakfile.phase_id != -1
            else:
                pid = self.phases.pids[self.phases.pnames.index(t)]
                m = self.peakfile.phase_id == pid
                
            Ints_frac = np.sum(self.peakfile.sumI[m]) / sumItot
            print(f'{t}: {Ints_frac:.4f}')  # prop of peaks assigned to this phase
            stats_dict[t].append(Ints_frac)
                           
        self.stats_labeled_peaks = stats_dict
    
    
        
    def plot_tth_histogram(self, min_tth, max_tth, tth_step = 0.001,
                            show_non_corrected=False,
                            show_theorytth = True,
                            mask=None):
            
        """
        plot histogram of corrected 2-theta over the selected range 
        
        Args:
        -------
        min_tth, max_tth   : 2-theta range over which histogram is computed
        tth_step           : increment fro 2-theta bins
        show_non_corrected : if True, alqo plots histogram of non-corrected 2-theta
        show_theorytth     : if True, add ticks marking the position of computed Bragg peaks for each phase
        mask (bool)        : custom boolean mask to select a subset of peaks in peakfile. must be the same length as columns in peakfile 
        """ 
            
        # sort peakfile by tth
        if self.sortkey != 'tthc':
            self.peakfile.sortby('tthc')
            self.sortkey = 'tthc'
            
        fig = pl.figure(figsize=(10,5))
            
        # raw tth plot
        if show_non_corrected:
            h_raw, b_raw, _ = utils.compute_tth_histogram(self.peakfile, use_tthc=False,
                                                       tthmin = min_tth, tthmax = max_tth, tth_step = tth_step,
                                                       mask=mask, density=True)
            
            pl.plot(b_raw, h_raw, '--', lw=.8, label='non-corrected peaks')
            
        # corr tth plot
        h, b,_ = utils.compute_tth_histogram(self.peakfile, use_tthc=True,
                                                       tthmin = min_tth, tthmax = max_tth, tth_step = tth_step,
                                                       mask=mask, density=True)
        
        pl.plot(b, h, '-', lw=.8, label = 'corrected peaks')
            
            
        # add theoretical Bragg peaks from stored phases
        if show_theorytth:
            colors = pl.matplotlib.cm.tab10.colors
            
            for i, pname in enumerate(self.phases.pnames):
                cs = self.phases.get(pname)
                pl.vlines(cs.strong_peaks[0], ymin=-0.1 * h.max(), ymax=0, colors = colors[i+2], label=cs.name)
                
        pl.legend(loc='upper left')
        pl.xlabel('2-theta deg')
        pl.ylabel('probability density')
        pl.xlim(min_tth, max_tth)
        
        
        
    def plot_tth_eta(self, min_tth=0, max_tth=25,
                            show_theorytth = True,
                            phase_colors = None):
            
        """ 
        plot eta vs. 2-theta over the selected range
        
        Args:
        ------
        min_tth, max_tth : 2-theta range over which histogram is computed
        show_theorytth   : if True, add ticks marking the position of computed Bragg peaks for each phase
        phase_colors     : use a mask to color each peak by phase. 
                           - 'from_phase_mask': pre-computed phase mask (in self.peakfile.phase_name) is used
                           - 'from_phase_id'  : final phase_id (obtaind from pixel-by-pixel phase mapping is used
                           - None : single color for all peaks
        """ 
        
        
        # sort peakfile by tth
        if self.sortkey != 'tthc':
            self.peakfile.sortby('tthc')
            self.sortkey = 'tthc'
         
        # limit number of peaks to plot to 1e6; Useful for large peakfiles
        p = min(1e6/self.peakfile.nrows,1)
        m = np.random.choice([True, False], self.peakfile.nrows, p = [p, 1-p])
        m2 = np.all([self.peakfile.tthc <= max_tth, self.peakfile.tthc >= min_tth], axis=0)
        
        # plot
        fig = pl.figure(figsize=(10,5))    
        pl.xlim(min_tth, max_tth)
        pl.plot(self.peakfile.tthc[m*m2], self.peakfile.eta[m*m2], ',', label = 'all peaks')
        
        colors = pl.matplotlib.cm.tab10.colors
        
        for i, pname in enumerate(self.phases.pnames):
            if phase_colors not in ['from_mask','from_phase_id']:
                continue
            cs = self.phases.get(pname)
            color = colors[i+2]
            
            if phase_colors == 'from_mask':
                pm = self.peakfile.getcolumn(pname)
            elif phase_colors == 'from_phase_id':
                pm = self.peakfile.phase_id == cs.phase_id
            
            pl.plot(self.peakfile.tthc[m*m2*pm], self.peakfile.eta[m*m2*pm], ',', color = color)
            
            
        # add theoretical Bragg peaks from stored phases
        if show_theorytth:
            for i, pname in enumerate(self.phases.pnames):
                cs = self.phases.get(pname)
                pl.vlines(cs.strong_peaks[0], -100,100, colors = colors[i+2], label=cs.name)
                
        pl.legend(loc='upper left')
        pl.xlabel('2-theta deg')
        pl.ylabel('eta deg')
                
                    