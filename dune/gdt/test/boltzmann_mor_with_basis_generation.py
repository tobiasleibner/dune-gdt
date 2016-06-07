from __future__ import print_function

import sys
import random
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
from mpi4py import MPI

from pymor.basic import *
from boltzmann.wrapper import DuneDiscretization
from boltzmann_RAPOD_timechunk_wise_stephans_pod import rapod_timechunk_wise


def calculate_mean_l2_error_for_random_samples(basis, b, seed = MPI.COMM_WORLD.Get_rank()*time.clock(), write_plot=True, mean_error=True):
    random.seed(seed)
    mu = [random.uniform(0., 8.), random.uniform(0., 8.), 0., random.uniform(0., 8.)]

    basis = b.convert_to_listvectorarray(basis.data)

    nt = int(b.num_time_steps - 1) if not b.with_half_steps else int((b.num_time_steps - 1)/2)
    d = DuneDiscretization(nt,
                           b.solver.time_step_length(),
                           1,
                           '',
                           2000000,
                           b.gridsize,
                           False,
                           True,
                           *mu)

    mu = d.parse_parameter(mu)

    # basis generation (POD on single trajectory)
    U = d.solve(mu, return_half_steps=False)

    rd, rc, _ = reduce_generic_rb(d.as_generic_type(), basis)
    #rd2, rc2, _ = reduce_generic_rb(d.as_generic_type(), V2)

    errs = []
    errs2 = []
    if write_plot:
        for dim in range(len(basis)):
            print('.', end=''); sys.stdout.flush()
            rrd, rrc, _ = reduce_to_subbasis(rd, dim, rc)
            U_rb = rrc.reconstruct(rrd.solve(mu))
            if mean_error:
                errs.append(np.sqrt(np.sum((U - U_rb).l2_norm()**2) / len(U)))
                errs2.append(np.sqrt(np.sum((U - basis.lincomb(U.dot(basis))).l2_norm()**2) / len(U)))
            else:
                errs.append(np.sum((U - U_rb).l2_norm()**2))
                errs2.append(np.sum((U - basis.lincomb(U.dot(basis))).l2_norm()**2))
    else:
        U_rb = rc.reconstruct(rd.solve(mu))
        if mean_error:
            errs = np.sqrt(np.sum((U - U_rb).l2_norm()**2) / len(U))
            errs2 = np.sqrt(np.sum((U - basis.lincomb(U.dot(basis))).l2_norm()**2) / len(U))
        else:
            errs = np.sum((U - U_rb).l2_norm()**2)
            errs2 = np.sum((U - basis.lincomb(U.dot(basis))).l2_norm()**2)

    l2_mean_errs = b.comm_world.gather(errs, root=0)
    l2_mean_projection_errors = b.comm_world.gather(errs2, root=0)
    if b.rank_world == 0:
#        with open("pickled_errors_gridsize_" + str(b.gridsize) + "_tol_" + str(b.epsilon_ast), 'w') as f:
#            pickle.dump(errs, f)
#        with open("pickled_errors2_gridsize_" + str(b.gridsize) + "_tol_" + str(b.epsilon_ast), 'w') as f:
#            pickle.dump(errs2, f)
        if write_plot:
            for err in l2_mean_errs:
                plt.semilogy(err)
            plt.savefig("gridsize_" + str(b.gridsize) + "_tol_" + str(b.epsilon_ast) + ".png")
            plt.gca().set_color_cycle(None) # restart color cycle
            for err2 in l2_mean_projection_errors:
                plt.semilogy(err2, linestyle=':')
            plt.savefig("gridsize_" + str(b.gridsize) + "_tol_" + str(b.epsilon_ast) + "_with_proj.png")
    return l2_mean_errs, l2_mean_projection_errors

if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    basis, _, _, b = rapod_timechunk_wise(grid_size, chunk_size, tol)
    calculate_mean_l2_error_for_random_samples(basis, b)
