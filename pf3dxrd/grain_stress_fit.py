# functions to compute strain and stress from a list of grains or a pixel map
import numpy as np
from tqdm import tqdm

import ImageD11.grain, ImageD11.stress

from pf_3dxrd import pixelmap, crystal_structure



# Bulk grain strain/stress (directly computed from a set of ubis)
###############################################################
def solve_strain_stress(EpsSigSolver, ubis, pname, **kwargs):
    """
    routine for EpsSigSolver to compute strain and stress properties for a list of ubis. Keyword arguments control which properties 
    are computed (deviatoric, invariant, grain/lab reference frame, etc.)
    
    Parameters
    ----------
    EpsSigSolver : EpsSigSolver object from ImageD11.stress, which should contain a set of elastic constants and a reference unit cell.
    ubis (list)  : lsit of grain ubis
    pname (str)  : phase name
    
    **kwargs (dict) : keyword arguments to control what should be computed. variables set up by kwargs are:
                      m_exponent (float)    : exponent for the Seth-Hill finite strain tensors E = 1/2m (U^2m - I)
                      reference_frame (str) : reference frame in which strain and stress are computed. 'Lab' or 'Grain'
                      output_format (str)   : output format for strain / stress. One of 'tensor','voigt','mandel','xfab','default'
                      deviatoric (bool)     : compute deviatoric tensors
                      invariants (bool)     : compute invariants
                      principal_components  : compute principal components
                      deviatoric_pc (bool)  : return principal components of the deviatoric tensors if it is selected. Default is Faulse
    
    Outputs are not directly returned but added as new instances to EpsSigSolver
    """
    # extract keyword arguments
    m_exponent = kwargs.get('m_exponent', None)
    reference_frame = kwargs.get('reference_frame', None)
    output_format = kwargs.get('output_format', None)
    deviatoric = kwargs.get('deviatoric', None)
    invariants = kwargs.get('invariants', None)
    principal_components = kwargs.get('principal_components', None)
    deviatoric_pc = kwargs.get('deviatoric_pc', None)
    
    assert reference_frame in ['Lab','Grain'], 'reference_frame must be either "Lab" or "Grain" '
    assert output_format in ['tensor','default','voigt','mandel','xfab'], 'unrecognized output format'
    
    # initialize strain stress solver
    ess = EpsSigSolver
    ess.UBIs = ubis
    
    # compute strain & stress
    print('computing deformation gradient tensor...')
    ess.compute_Deformation_Gradient_Tensors()
    
    print('computing strain and stress...')
    if reference_frame == 'Lab':
        ess.strain2stress_Lab(m = m_exponent)
    else:
        ess.strain2stress_Ref(m = m_exponent)
        
    # compute additional properties : deviatoric tensor, eigen decomposition etc.
    if any([deviatoric, invariants, principal_components]):
        print('computing additional properties')
    
    if deviatoric:
        ess.deviatoric_tensors()
        
    if invariants:
        ess.invariant_props(f'eps_{reference_frame}')
        ess.invariant_props(f'sigma_{reference_frame}')
        
    if principal_components:
        if deviatoric_pc and deviatoric:
            ess.compute_principal_components(f'eps_{reference_frame}_d')
            ess.compute_principal_components(f'sigma_{reference_frame}_d')
        else:
            ess.compute_principal_components(f'eps_{reference_frame}')
            ess.compute_principal_components(f'sigma_{reference_frame}')        
        
    # convert tensors to 6-component vecs
    if output_format != 'tensor':
        print('converting tensors to vecs...')
        ess.convert_tensors_to_vecs(output_format)
    
    

    
def xmap_strain_stress_px(xmap, pname, stress_unit='MPa', B0 = None, overwrite_xmap=False, verbose=False, **kwargs):
    """ 
    compute strain and stress properties for pixel ubis in xmap, and update xmap data columns
    
    Parameters
    ----------
    xmap        : pixelmap object. Must contain a UBI column (ie, indexing must have been done previously)
    pname (str) : name of the phase to select in xmap. mist be in xmap.phases
    stress_unit : str, give unit of stress defined with the elastic parameters
    B0 (array)  : reference unit cell shp (6,1). If not provided, reference cell stored in xmap.phase.pname is used 
    overwrite_xmap (bool) : overwrite previous strain/stress data columns
    kwargs : keyword arguments ot pass to solve_strain_stress (see docstring of solve_strain_stress for more details)
    """
    # initialize strain stress solver
    cs = xmap.phases.get(pname).copy()
    pxsel = xmap.phase_id == cs.phase_id
    ubis = xmap.UBI[pxsel]
    
    if B0 is not None:
        assert B0.shape == (6,), 'wrong shape for B0. Must be an 6-component 1D array [a,b,c,alpha,beta,gamma]'
        cs.cell = B0
    
    ess = cs.to_EpsSigSolver()  
    ess.stress_unit=stress_unit

    # compute strain and stress for pixels 
    solve_strain_stress(ess, ubis, pname, **kwargs)
    
    # select data columns to add to xmap
    alldata = [d for d in dir(ess) if any(['eps' in d, 'sigma' in d])]  # all strain / stress related columns
    to_keep = [d for d in alldata if ess.__getattribute__(d)[0].shape != (3,3) or d.endswith('eigvecs') ]
    
    for dname in to_keep:
        data = np.array(ess.__getattribute__(dname))
        
        dname = dname.replace('eps','strain').replace('sigma','stress')
        
        if isinstance(data[0], (float, int)):
            init_array = np.full(xmap.xyi.shape, np.inf)
            
        else: init_array = np.full(xmap.xyi.shape + data[0].shape, np.inf)
        
        if verbose:
            print(f'addding {dname} {init_array.shape} to xmap')
                
        if overwrite_xmap or dname not in xmap.titles():
            xmap.add_data(init_array, dname)

        xmap.update_pixels(xmap.xyi[pxsel], dname, data)
        
        

def xmap_strain_stress_grains(xmap, pname, stress_unit='MPa', B0 = None, overwrite_xmap=False, **kwargs):
    """ 
    compute strain and stress properties for pixel ubis in xmap, and update xmap data columns
    
    Parameters
    ----------
    xmap        : pixelmap object. Must contain a UBI column (ie, indexing must have been done previously)
    pname (str) : name of the phase to select in xmap. mist be in xmap.phases
    stress_unit : str, give unit of stress defined with the elastic parameters
    B0 (array)  : reference unit cell shp (6,1). If not provided, reference cell stored in xmap.phase.pname is use 
    overwrite_xmap (bool) : overwrite previous strain/stress data columns
    kwargs : keyword arguments ot pass to solve_strain_stress (see docstring of solve_strain_stress for more details)
    """
    # initialize strain stress solver
    cs = xmap.phases.get(pname)
    glist = xmap.grains.select_by_phase(pname)
    ubis = xmap.grains.get_all('ubi', pname)
    
    if B0 is not None:
        assert B0.shape == (6,), 'wrong shape for B0. Must be an 6-component 1D array [a,b,c,alpha,beta,gamma]'
        cs.cell = B0
    
    ess = cs.to_EpsSigSolver()  
    ess.stress_unit=stress_unit
    
    # compute strain and stress for selected grains
    solve_strain_stress(ess, ubis, pname, **kwargs)
    
    # select data columns to add to xmap
    alldata = [d for d in dir(ess) if any(['eps' in d, 'sigma' in d])]  # all strain / stress related columns
    to_keep = [d for d in alldata if ess.__getattribute__(d)[0].shape != (3,3) or d.endswith('eigvecs') ]
    
    for dname in to_keep:
        data = np.array(ess.__getattribute__(dname))
        dname = dname.replace('eps','strain').replace('sigma','stress')
        # add data as new grain attribute
        for g,val in tqdm( zip(glist,data) ):
            setattr(g, dname, val)
        
        # map grain prop to xmap
        xmap.map_grain_prop(dname, pname)
