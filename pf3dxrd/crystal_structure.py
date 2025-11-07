import os, sys
import numpy as np, pylab as pl
import scipy.spatial, scipy.signal
import copy

import diffpy.structure
import Dans_Diffraction as dif

from orix import plot as opl, crystal_map as ocm, vector as ovec
from ImageD11 import stress



class CS:
    """ 
    class to store crystal structure information imported from cif files. Wrapper built upon modules
    diffpy.structure ( https://www.diffpy.org/diffpy.structure/),
    orix (https://orix.readthedocs.io/en/stable/) and 
    Dans_diffraction ( https://pypi.org/project/Dans-Diffraction/1.5.0/).
    
    Data import from the cif file relies on diffpy.strucure and Dans_diffraction modules.
    """
    
    def __init__(self, name, pid=-1, cif_path=None):
        self.name = name
        self.cif_path = cif_path
        self.cif_file = None
        self.phase_id = pid
        self.spg = None
        self.spg_no = None
        self.lattice_type = None
        self.cell = []
        self.str_dans = None
        self.str_diffpy = None
        self.orix_phase = None
        
        if pid < 0:
            self.color = 'k'
        else:
            self.color = pl.matplotlib.cm.tab10.colors[pid]        
        
        if self.cif_path is not None:
            self.add_data_from_cif(self.cif_path)
            self.cif_file = cif_to_list(self.cif_path)
            

    def __str__(self):
        return f"CS: {self.name}, phase_id: {self.phase_id}, spg: {self.spg}, spg_no: {self.spg_no}, lattice: {self.cell}"
    
    
    def get(self,prop):
        return self.__getattribute__(prop)
    
    
    def copy(self):
        return copy.deepcopy(self)
    
    
    class ElasticConstants:
        """ sub-class to store elastic constants for the stiffness tensor Cij """
        def __init__(self,
                     symmetry = None,
                     unitcell = None,
                     c11=None,c12=None,c13=None,c14=None,c15=None,c16=None,
                     c22=None,c23=None,c24=None,c25=None,c26=None,
                     c33=None,c34=None,c35=None,c36=None,
                     c44=None,c45=None,c46=None,
                     c55=None,c56=None,
                     c66=None):
            self.symmetry = str(symmetry)
            self.unitcell = unitcell
            self.c11 = c11
            self.c12 = c12
            self.c13 = c13
            self.c14 = c14
            self.c15 = c15
            self.c16 = c16
            self.c22 = c22
            self.c23 = c23
            self.c24 = c24
            self.c25 = c25
            self.c26 = c26
            self.c33 = c33
            self.c34 = c34
            self.c35 = c35
            self.c36 = c36
            self.c44 = c44
            self.c45 = c45
            self.c46 = c46
            self.c55 = c55
            self.c56 = c56
            self.c66 = c66
            
        
        def get(self,attr):
            return self.__getattribute__(attr)
        
        def as_dict(self):
            """ return all elastic constants as a dict """
            return {cij:self.get(cij) for cij in dir(self) if cij.startswith('c') and self.get(cij) is not None}
        
        
    # METHODS
    ##############
            
    def add_elastic_constants(self,
                             symmetry = None,  
                             c11=None,c12=None,c13=None,c14=None,c15=None,c16=None,
                             c22=None,c23=None,c24=None,c25=None,c26=None,
                             c33=None,c34=None,c35=None,c36=None,
                             c44=None,c45=None,c46=None,
                             c55=None,c56=None,
                             c66=None):
        
        self.elastic_pars = self.ElasticConstants(symmetry = symmetry,
                                                  unitcell = self.cell,
                                                  c11=c11,c12=c12,c13=c13,c14=c14,c15=c15,c16=c16,
                                                  c22=c22,c23=c23,c24=c24,c25=c25,c26=c26,
                                                  c33=c33,c34=c34,c35=c35,c36=c36,
                                                  c44=c44,c45=c45,c46=c46,
                                                  c55=c55,c56=c56,
                                                  c66=c66)
     
    
    
    def add_data_from_cif(self, cif_path):
        """ 
        Import crystal structure information from cif file and extract different properties.
        
        Following attributes added to CS / updated:
        ---------------
        cif_path (str) : path to the cif file
        str_dans       : dif.Crystal object (from Dans_diffraction)
        str_diffpy     : diffpy.structure object
        orix_phase     : orix.Phase object
        spg (str)      : name of space group
        spg_no (int)   : space group number 
        lattice type (str) : type of Bravais lattice (P,I,F,R, etc.)   
        cell (array)   : unit cell array [a,b,c,alpha,beta,gamma]
        """
        
        assert os.path.exists(cif_path), 'incorrect path for crystal structure file'
        self.cif_path = cif_path
        
        # load with Dans_diffraction and diffpy.structure 
        try:
            self.str_dans = dif.Crystal(self.cif_path)
            self.str_diffpy = diffpy.structure.loadStructure(self.cif_path, fmt='cif')
        except:
            print(self.name, 'No cif file found, or maybe it is corrupted')
    
        # symmetry info (space group nb + name) + unit cell
        self.spg = self.str_dans.Symmetry.spacegroup_name().split(' ')[0]
        try:
            self.spg_no = int(self.str_dans.Symmetry.spacegroup_number)
        except:
            print('No space group number in cif file')
    
        self.lattice_type = self.str_dans.Symmetry.spacegroup_name()[0]
        cell = self.str_dans.Cell
        self.cell = np.array([cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma])
        self.orix_phase = ocm.Phase(name=self.name, space_group=self.spg_no, structure=self.str_diffpy, color=self.color)  
        
   
    def to_EpsSigSolver(self):
        """
        creates strain stress solver (from ImageD11.stress) using unitcell and elastic constants in self.elastic_pars
        See also: ImageD11.stress.EpsSigsolver
        """
        
        assert 'elastic_pars' in dir(self), 'No elastic constants found in CS!'
        e = self.elastic_pars
        
        ess = stress.EpsSigSolver(name = self.name,
                                              symmetry = self.elastic_pars.symmetry,
                                              unitcell = self.cell,
                                              c11=e.c11, c12=e.c12, c13=e.c13, c14=e.c14, c15=e.c15, c16=e.c16,
                                              c22=e.c22, c23=e.c23, c24=e.c24, c25=e.c25, c26=e.c26,
                                              c33=e.c33, c34=e.c34, c35=e.c35, c36=e.c36,
                                              c44=e.c44, c45=e.c45, c46=e.c46,
                                              c55=e.c55, c56=e.c56,
                                              c66=e.c66)
        return ess

    
    def get_ipfkey(self, direction = ovec.Vector3d.zvector()):
        """ 
        get inverse pole figure color key from crystal point group (using orix)
        
        See also: https://orix.readthedocs.io/en/stable/tutorials/inverse_pole_figures.html
        """
        self.ipfkey = opl.IPFColorKeyTSL(self.orix_phase.point_group, direction = direction)
        
        
        
    def compute_powder_spec(self, wl, min_tth=0, max_tth=25, doplot=False):
        """
        simulate powder diffraction spectrum I(two-theta) from cif data (using Dans_dif module). 
        
        Args:
        -------
        wl : X-ray wavelength (Angstrom)
        min_tth, max_tth : two-theta range over which spectrum is computed
        doplot (bool) : plot the powder spectrum
        
        New attributes: 
        -----------------------
        powder_spec (list) : [two-theta, Intensity].
        Intensity is normalized to [0,1], where 1 is the intensity of the strongest diffraction peak
        
        See also: Dans_diffraction.Scatter
        """
    
        E_kev = X_ray_energy(wl)
        self.str_dans.Scatter.setup_scatter(scattering_type='x-ray', energy_kev=E_kev, min_twotheta=min_tth, max_twotheta=max_tth)
    
        # simulate powder pattern
        tth, ints, _ = self.str_dans.Scatter.powder(units='twotheta')  # tth, Intensity coordinates of powder pattern
        ints = ints/ints.max()  # normalize intensity
        
        self.powder_spec = [tth, ints]
        
        if doplot:
            pl.figure()
            pl.plot(tth, ints,'-')
            pl.xlabel('tth deg')
            pl.ylabel('normalized Intensity')
            pl.title('xray powder spectrum - ' + self.name)
    

    
    def find_strongest_peaks(self, Imin=0.1, Nmax=30, prominence = 0.1, doplot=False):
        """ 
        Do peaksearch on powder spectrum (simulated with Dans_diffraction) and return the N-strongest peaks
        sorted by decreasing intensity. 
        
        Args:
        ---------
        Imin       : Minimum intensity threshold to consider a local maximum as a peak
        prominence : Minimum prominence to consider a local maximum as a peak
        Nmax       : If the result of peaksearch contains more than Nmax peak, keep only the Nmax strongest
        
        See also: scipy.signal.find_peaks
        """
        
        try:
            tth, I = self.powder_spec[0], self.powder_spec[1]
        except:
            print('No spectrum data. Compute powder spectrum first')
        
        pksindx, pksI = scipy.signal.find_peaks(I, height=Imin, prominence=prominence)
        pks = tth[pksindx].tolist()
        pksI = list(pksI.values())[0].tolist()  # pksI returned in a dict. convert it to an array
    
        # sort peaks by intensity
        pks_sorted = [l1 for (l2, l1) in sorted(zip(pksI,pks), key=lambda x: x[0], reverse=True)]
        pksI_sorted = [l2 for (l2, l1) in sorted(zip(pksI,pks), key=lambda x: x[0], reverse=True)]
    
        # take only the most intense peaks
        if len(pks_sorted) > Nmax:
            pks_sorted = pks_sorted[:Nmax]
            pksI_sorted = pks_sorted[:Nmax]
    
        self.strong_peaks = [pks_sorted, pksI_sorted]
        
        if doplot:
            pl.figure()
            pl.plot(tth, I,'-')
            pl.vlines(x=pks_sorted, ymin=0, ymax=1, colors='red', lw=.5)
            pl.xlabel('tth deg')
            pl.ylabel('normalized Intensity')
            pl.title('strongest diffraction peaks - ' + self.name)
           
        
        
# UTILS
###########################

def load_CS_from_cif(cif_path, name='', pid = -1):
    """
    create a CS object directly from a cif file
    
    Args:
    --------
    cif_path (str) : path to cif file
    name (str)     : phase name
    pid (int)      : phase id. integer label unique to the phase 
                     (useful for defining phase_id in pixelmap and ImageD11.columnfile)
    
    """
    cs = CS(name, pid, cif_path)
    
    if name == '':
        cs.name = cs.str_dans.name
        try:
            cs.name[0]
        except:
            print('no phase name found in cif file. Please enter a name for the structure')
    return cs



def cif_to_str(file_path):
    """ Reads a cif file line by line and returns the content as a string """
    try:
        with open(file_path, 'r') as file:
            cif_str = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    return cif_str



def str_to_cif(file_content, output_file_path):
    """ Writes a cif string to a file, respecting line breaks. """
    try:
        with open(output_file_path, 'w') as output_file:
            output_file.write(file_content)
        print(f"File '{output_file_path}' created successfully.")
    except Exception as e:
        print(f"Error: {e}")
    

    
def cif_to_list(file_path):
    """ Reads a cif file line by line and returns the content as a list of strings. """
    file_text_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                file_text_list.append(line.strip())
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return file_text_list




def list_to_cif(file_text_list, output_file_path):
    """ Writes a list of strings (representing a cif file) to a file, with each string representing a line. """
    try:
        with open(output_file_path, 'w') as output_file:
            for line in file_text_list:
                output_file.write(line + '\n')
        print(f"File '{output_file_path}' created successfully.")
    except Exception as e:
        print(f"Error: {e}")


def X_ray_energy(wl):
    """ return x-ray energy (kev) from wavelength """
    E_kev = 6.62607015e-34*2.99792e8/(wl*1e-10) / 1.60218e-19 / 1e3
    return E_kev


