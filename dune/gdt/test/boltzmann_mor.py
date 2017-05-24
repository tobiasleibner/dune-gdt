import sys
import random
import time
import numpy as np
from timeit import default_timer as timer
from mpi4py import MPI

from pymor.basic import *
from boltzmann.wrapper import DuneDiscretization
from boltzmann_binary_tree_hapod import boltzmann_binary_tree_hapod
from Hapod import convert_to_listvectorarray, solver_statistics


def calculate_l2_error_for_random_samples(basis, mpi, solver, grid_size, chunk_size, seed = MPI.COMM_WORLD.Get_rank()*time.clock(), 
                                          mean_error=False, with_half_steps=True):
    '''Calculates model reduction and projection error for random parameter'''

    random.seed(seed)
    mu = [random.uniform(0., 8.), random.uniform(0., 8.), 0., random.uniform(0., 8.)]

    basis = convert_to_listvectorarray(basis.data)
    _, num_time_steps = solver_statistics(solver, chunk_size, with_half_steps)

    nt = int(num_time_steps - 1) if not with_half_steps else int((num_time_steps - 1)/2)
    d = DuneDiscretization(nt,
                           solver.time_step_length(),
                           '',
                           2000000,
                           grid_size,
                           False,
                           True,
                           *mu)

    mu = d.parse_parameter(mu)

    # calculate high-dimensional solution
    start = timer()
    U = d.solve(mu, return_half_steps=False)
    elapsed_high_dim = timer() - start

    rd, rc, _ = reduce_generic_rb(d.as_generic_type(), basis)

    red_errs = []
    proj_errs = []

    start = timer()
    U_rb = rd.solve(mu)
    elapsed_red = timer() - start
    U_rb = rc.reconstruct(U_rb)
    if mean_error:
        red_errs = np.sqrt(np.sum((U - U_rb).l2_norm()**2) / len(U))
        proj_errs = np.sqrt(np.sum((U - basis.lincomb(U.dot(basis))).l2_norm()**2) / len(U))
    else:
        red_errs = np.sum((U - U_rb).l2_norm()**2)
        proj_errs = np.sum((U - basis.lincomb(U.dot(basis))).l2_norm()**2)

    l2_mean_errs = mpi.comm_world.gather(red_errs, root=0)
    l2_mean_projection_errors = mpi.comm_world.gather(proj_errs, root=0)
    elapsed_red = mpi.comm_world.gather(elapsed_red, root=0)
    elapsed_high_dim = mpi.comm_world.gather(elapsed_high_dim, root=0)
    return l2_mean_errs, l2_mean_projection_errors, elapsed_red, elapsed_high_dim

if __name__ == "__main__":
    ''' Solves a HAPOD to get a basis of the reduced space and then calculates projection and model reduction error for random samples'''
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    omega = float(sys.argv[4])
    basis, _, total_num_snaps, _, mpi, _, _, solver = boltzmann_binary_tree_hapod(grid_size, chunk_size, tol*grid_size, omega=omega)
    basis, _ = mpi.shared_memory_bcast_modes(basis)
    red_errs, proj_errs, elapsed_red, elapsed_high_dim = calculate_l2_error_for_random_samples(basis, mpi, solver, grid_size, chunk_size, mean_error=False)
    red_err = np.sqrt(np.sum(red_errs) / total_num_snaps) / grid_size if mpi.rank_world == 0 else None
    proj_err = np.sqrt(np.sum(proj_errs) / total_num_snaps) / grid_size if mpi.rank_world == 0 else None
    elapsed_red_mean = np.sum(elapsed_red) / len(elapsed_red) if mpi.rank_world == 0 else None
    elapsed_high_dim_mean = np.sum(elapsed_high_dim) / len(elapsed_high_dim) if mpi.rank_world == 0 else None
    if mpi.rank_world == 0:
        print("\n\n\nResults:\n")
        print('Solving the high-dimensional problem took %g seconds on average.' % elapsed_high_dim_mean)
        print('Solving the reduced problem took %g seconds on average.' % elapsed_red_mean)
        print('The mean l2 reduction error and mean l2 projection error were %g and %g, respectively.'
              % (red_err, proj_err))
        
