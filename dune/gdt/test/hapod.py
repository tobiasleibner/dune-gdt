from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.algorithms.pod import pod as pymor_pod
from pymor.basic import gram_schmidt
import numpy as np
from scipy.linalg import eigh
from boltzmannutility import create_listvectorarray

class HapodParameters:
    '''Stores the HAPOD parameters :math:`\omega`, :math:`\epsilon^\ast` and :math:`L_\mathcal{T}` for easier passing
       and provides the local error tolerance :math:`\varepsilon_\mathcal{T}(\alpha)` '''
    def __init__(self, rooted_tree_depth, epsilon_ast=1e-4, omega=0.95):
        self.epsilon_ast = epsilon_ast
        self.omega = omega
        self.rooted_tree_depth = rooted_tree_depth
        
    def get_epsilon_alpha(self, num_snaps_in_leafs, root_of_tree=False):
        if not root_of_tree:
            epsilon_alpha = self.epsilon_ast * np.sqrt(1. - self.omega**2) * \
                            np.sqrt(num_snaps_in_leafs) / np.sqrt(self.rooted_tree_depth - 1)
        else:
            epsilon_alpha = self.epsilon_ast * self.omega * np.sqrt(num_snaps_in_leafs)
        return epsilon_alpha


def pod(inputs, num_snaps_in_leafs, parameters, root_of_tree=False, orthonormalize=True, incremental=True):
    '''Calculates a POD in the HAPOD tree. The input is a list where each element is either a vectorarray or 
       a pair of (orthogonal) vectorarray and singular values from an earlier POD. If incremental is True, the 
       algorithm avoids the recalculation of the diagonal blocks where possible by using the singular values.
       :param inputs: list of input vectors (and svals)
       :type inputs: list where each element is either a vectorarray or [vectorarray, numpy.ndarray]
       :param num_snaps_in_leafs: The number of snapshots below the current node (:math:`\widetilde{\mathcal{S}}_\alpha`)
       :param parameters: An object of type HapodParameters
       :param root_of_tree: Whether this is the root of the HAPOD tree
       :param orthonormalize: Whether to reorthonormalize the resulting modes
       :param incremental: Whether to build the gramian incrementally using information from the singular values'''
    # calculate offsets and check whether svals are provided in input
    offsets = [0]
    svals_provided = []
    vector_length = 0
    epsilon_alpha = parameters.get_epsilon_alpha(num_snaps_in_leafs, root_of_tree=root_of_tree)
    for i, modes in enumerate(inputs):
        if type(modes) is list:
           assert(issubclass(type(modes[0]), VectorArrayInterface))
           assert(issubclass(type(modes[1]), np.ndarray) and modes[1].ndim == 1)
           modes[0].scal(modes[1])
           svals_provided.append(True)
        elif issubclass(type(modes), VectorArrayInterface):
           inputs[i] = [modes]
           svals_provided.append(False)
        else:
           raise ValueError("")
        offsets.append(offsets[-1]+len(inputs[i][0]))
        vector_length = max(vector_length, modes[0][0].dim)

    if incremental:
        # calculate gramian avoiding recalculations
        gramian = np.empty((offsets[-1],) * 2)
        all_modes = create_listvectorarray(0, vector_length)
        for i in range(len(inputs)):
            modes_i, svals_i = [inputs[i][0], inputs[i][1] if svals_provided[i] else None]
            gramian[offsets[i]:offsets[i+1], offsets[i]:offsets[i+1]] = np.diag(svals_i)**2 if svals_provided[i] else modes_i.gramian()
            for j in range(i,len(inputs)):
                modes_j = inputs[j][0]
                cross_gramian = modes_i.dot(modes_j)
                gramian[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = cross_gramian
                gramian[offsets[j]:offsets[j+1], offsets[i]:offsets[i+1]] = cross_gramian.T
            all_modes.append(modes_i) 
        modes_i._list=None
        
        EVALS, EVECS = eigh(gramian, overwrite_a=True, turbo=True, eigvals=None)
        del gramian

        EVALS = EVALS[::-1]
        EVECS = EVECS.T[::-1, :]  # is this a view? yes it is!

        errs = np.concatenate((np.cumsum(EVALS[::-1])[::-1], [0.]))

        below_err = np.where(errs <= epsilon_alpha**2)[0]
        first_below_err = below_err[0]

        svals = np.sqrt(EVALS[:first_below_err])
        EVECS = EVECS[:first_below_err]

        final_modes = all_modes.lincomb(EVECS / svals[:, np.newaxis])
        all_modes._list = None
        del modes
        del EVECS

        if orthonormalize:
            final_modes = gram_schmidt(final_modes, copy=False)
        
        return final_modes, svals
    else:
        modes = create_listvectorarray(0, vector_length)
        for i in range(len(inputs)):
            if svals_provided[i]:
                inputs[i][0].scal(inputs[i][1])
            modes.append(inputs[i][0])
        return pymor_pod(modes, atol=0., rtol=0., l2_err=epsilon_alpha, orthonormalize=orthonormalize, check=False)

