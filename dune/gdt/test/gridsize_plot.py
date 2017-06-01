from __future__ import print_function

import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer
from mpi4py import MPI
import resource

from pymor.basic import *
from boltzmann.wrapper import DuneDiscretization
from boltzmann_incremental_hapod import boltzmann_incremental_hapod
from boltzmann_binary_tree_hapod import boltzmann_binary_tree_hapod
from boltzmann_all_binary_tree_hapod import boltzmann_all_binary_tree_hapod
from boltzmann_pod import boltzmann_pod
from boltzmann_mor import calculate_l2_error_for_random_samples
from mpiwrapper import MPIWrapper
from boltzmannutility import calculate_total_projection_error

chunk_size = int(sys.argv[1])
initial_tol = float(sys.argv[2])
omega = float(sys.argv[3])

# inc_inc = incremental HAPOD with incremental gramian POD algorithm, inc = incremental HAPOD with standard POD
hapod_types = ["inc_inc", "inc", "bt_inc", "bt", "all_bt_inc", "all_bt"]
incremental_gramian = [True, False, True, False, True, False]

# create lists of empty lists to store the values
num_modes_hapod, max_num_local_modes_hapod, max_vecs_before_pod_hapod, time_hapod, l2_proj_errs_snaps_hapod, l2_proj_errs_random_hapod, l2_red_errs_hapod, memory_hapod, svals_hapod = [[[] for i in hapod_types] for i in range(9)]
num_modes_pod, time_pod, time_data_gen_pod, l2_proj_errs_snaps_pod, l2_proj_errs_random_pod, l2_red_errs_pod, memory_pod, svals_pod = [[] for i in range(8)]

filenames_hapod = ["/scratch/tmp/l_tobi01/pickled_bases/hapod_" + hapod_type  for hapod_type in hapod_types]
filename_pod = "/scratch/tmp/l_tobi01/pickled_bases/pod"
x_axis = []
logfilename = "logfile_gridsize_plot" + "_chunk_" + str(chunk_size) + "_tol_" + str(initial_tol) +  "_omega_" + str(omega)
logfile = open(logfilename, "w", 0)

for grid_size in (5, 10, 20, 40, 60, 80, 100):
    tol = initial_tol * grid_size
    x_axis.append(grid_size)
    mpi = MPIWrapper()
    if mpi.rank_world == 0:
        print("Current grid size:", grid_size)
        logfile.write("Current grid size:" + str(grid_size) + "\n")

    for num, hapod_type in enumerate(hapod_types):
        basis = []
        try:
            f = open(filenames_hapod[num] + "_" + str(grid_size) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega)+ "_tol_" + str(tol), "rb")
            if mpi.rank_world == 0:
                basis, svals, total_num_snapshots, elapsed, max_vecs_before_pod, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(f) 
            else:
                basis, svals, total_num_snapshots, elapsed, max_vecs_before_pod, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*9
            f.close()    
        except (OSError, IOError) as e:
    	    start = timer()
            basis, svals, total_num_snapshots, mu, mpi, max_vecs_before_pod, max_local_modes, solver \
                     = boltzmann_incremental_hapod(grid_size, chunk_size, tol, omega=omega, incremental_gramian=incremental_gramian[num]) if num < 2 else \
                       (boltzmann_binary_tree_hapod(grid_size, chunk_size, tol, omega=omega, incremental_gramian=incremental_gramian[num]) if num < 4 else \
                        boltzmann_all_binary_tree_hapod(grid_size, chunk_size, tol, omega=omega, incremental_gramian=incremental_gramian[num]))

            elapsed = timer() - start
            #basis = mpi.shared_memory_bcast_modes(basis, returnlistvectorarray=True)
            #l2_red_err_list_random, l2_proj_err_list_random, _, _ = calculate_l2_error_for_random_samples(basis, mpi, solver, grid_size, chunk_size)
            basis, win = mpi.shared_memory_bcast_modes(basis)
            l2_proj_err_snaps = calculate_total_projection_error(basis, grid_size, mu, total_num_snapshots, mpi)
            if mpi.rank_world == 0:
                l2_proj_err_snaps /= grid_size
            l2_proj_err_random = 0 #np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots) / grid_size if mpi.rank_world == 0 else None
            l2_red_err_random = 0 #np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots) / grid_size if mpi.rank_world == 0 else None
            mpi.comm_world.Barrier()
            if mpi.rank_world == 0:
                f = open(filenames_hapod[num] + "_" + str(grid_size) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega) + "_tol_" + str(tol), "wb")
                tmp = basis, svals, total_num_snapshots, elapsed, max_vecs_before_pod, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
                pickle.dump(tmp, f)
                f.close()
            win.Free()
            del solver
        time_hapod[num].append(elapsed)
        num_modes_hapod[num].append(len(basis))
        max_num_local_modes_hapod[num].append(max_local_modes)
        max_vecs_before_pod_hapod[num].append(max_vecs_before_pod)
        l2_proj_errs_snaps_hapod[num].append(l2_proj_err_snaps)
        l2_proj_errs_random_hapod[num].append(l2_proj_err_random)
        l2_red_errs_hapod[num].append(l2_red_err_random)
        memory_hapod[num].append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
        svals_hapod[num].append(svals)
        del basis
        if mpi.rank_world == 0:
            print(hapod_type + "done \n")

#    try:
#        g = open(filename_pod + "_tol_" + str(tol), "rb")
#        if mpi.rank_world == 0:
#            basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(g) 
#        else:
#            basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*8
#        g.close()    
#    except (OSError, IOError) as e:
#        start = timer()
#        basis, svals, total_num_snapshots, mu, mpi, elapsed_data_gen, elapsed_pod, solver = boltzmann_pod(grid_size, tol)
#        elapsed = timer() - start
#        basis = mpi.shared_memory_bcast_modes(basis, returnlistvectorarray=True)
#        l2_red_err_list_random, l2_proj_err_list_random, _, _ = calculate_l2_error_for_random_samples(basis, mpi, solver, grid_size, chunk_size)
#        basis, win = mpi.shared_memory_bcast_modes(basis)
#        l2_proj_err_snaps = calculate_total_projection_error(basis, grid_size, mu, total_num_snapshots, mpi)
#        if mpi.rank_world == 0:
#            l2_proj_err_snaps /= grid_size
#        l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots)/grid_size if mpi.rank_world == 0 else None
#        l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots)/grid_size if mpi.rank_world == 0 else None
#        mpi.comm_world.Barrier()
#        if mpi.rank_world == 0:
#            g = open(filename_pod + "_tol_" + str(tol), "wb")
#            tmp = basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
#            pickle.dump(tmp, g)
#            g.close()
#        win.Free()
#    time_pod.append(elapsed_pod + elapsed_data_gen)
#    time_data_gen_pod.append(elapsed_data_gen)
#    num_modes_pod.append(len(basis))
#    l2_proj_errs_snaps_pod.append(l2_proj_err_snaps)
#    l2_proj_errs_random_pod.append(l2_proj_err_random)
#    l2_red_errs_pod.append(l2_red_err_random)
#    memory_pod.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
#    svals_pod.append(svals)
#    del basis
#    if mpi.rank_world == 0:
#        print("pod done\n")


from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')

if mpi.rank_world == 0:
    plt.figure()

    for num, hapod_type in enumerate(hapod_types):
        plt.semilogx(x_axis, num_modes_hapod[num], label=hapod_type)
    #plt.semilogx(x_axis, num_modes_pod, label="pod")
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("POD basis size")
    plt.legend(loc='upper left')

    filename = "grid_size_num_modes_chunk_%d_tol_%g_omega_%g" % (chunk_size, initial_tol, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = [x_axis] 
    for num in range(len(hapod_types)):
        data += [num_modes_hapod[num], max_num_local_modes_hapod[num], max_vecs_before_pod_hapod[num]]
    data = np.array(data) 
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%f']*(3*len(hapod_types)+1))


    plt.figure()

    for num, hapod_type in enumerate(hapod_types):
        plt.loglog(x_axis, l2_proj_errs_snaps_hapod[num], label="proj_snaps_" + hapod_type)
        plt.loglog(x_axis, l2_proj_errs_random_hapod[num], label="proj_random_" + hapod_type)
        plt.loglog(x_axis, l2_red_errs_hapod[num], label="reduced_" + hapod_type)
    plt.gca().set_color_cycle(None) # restart color cycle
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("L2 mean error")
    plt.legend(loc='lower left', prop=fontP)

    filename = "grid_size_l2_mean_errs_chunk_%d_tol_%g_omega_%g" % (chunk_size, initial_tol, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = [x_axis]
    for num in range(len(hapod_types)):
        data += [l2_proj_errs_snaps_hapod[num], l2_proj_errs_random_hapod[num], l2_red_errs_hapod[num]]
    data = np.array(data)
                    
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%f']*(3*len(hapod_types)+1))


    plt.figure()

    for num, hapod_type in enumerate(hapod_types):
        plt.semilogx(x_axis, time_hapod[num], label=hapod_type)
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Elapsed time (seconds)")
    plt.legend(loc='upper right')

    filename = "grid_size_time_chunk_%d_tol_%g_omega_%g" % (chunk_size, initial_tol, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_axis] + [time_hapod[num] for num in range(len(hapod_types))])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%f']*(len(hapod_types)+1))

    plt.figure()

    for num, hapod_type in enumerate(hapod_types):
        plt.semilogx(x_axis, memory_hapod[num], label=hapod_type)
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Memory usage (GB)")
    plt.legend(loc='upper right')

    plt.savefig("memory_gridsize_" + str(grid_size) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega) + ".png")
    plt.clf()


    plt.figure()

    x_end = min([len(svals_hapod[num][-1]) for num in range(len(hapod_types))])
    x_counts = range(1,x_end+1)
    for num, hapod_type in enumerate(hapod_types):
        plt.semilogy(x_counts, svals_hapod[num][-1][0:x_end], label=hapod_type)

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Value")
    plt.legend(loc='upper right')

    filename = "grid_size_svals_chunk_%d_tol_%g_omega_%g" % (chunk_size, initial_tol, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_counts] + [svals_hapod[num][-1][0:x_end] for num in range(len(hapod_types))])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%d'] + ['%.15g']*(len(hapod_types)))

    logfile.close()
