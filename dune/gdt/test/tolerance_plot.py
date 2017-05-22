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
from boltzmann_Hapod_timechunk_wise import hapod_timechunk_wise
from boltzmann_POD import boltzmann_standard_pod
from boltzmann_mor import calculate_mean_l2_error_for_random_samples
from boltzmann_live_HAPOD_on_trajectory import live_hapod_on_trajectory
from Hapod import HapodBasics

grid_size = int(sys.argv[1])
chunk_size = int(sys.argv[2])
omega = float(sys.argv[3])

# tcw = timechunkwise, live = live HAPOD with compute nodes, bt = binary tree of compute nodes, inc=incremental_POD_algorithm
hapod_types = ["tcw_live", "tcw_live_inc", "tcw_bt", "tcw_bt_inc"]
is_incremental = [False, True, False, True]
binary_tree = [False, False, True, True]

# create lists of empty lists to store the values
num_modes_hapod, max_num_local_modes_hapod, max_vecs_before_pod_hapod, time_hapod, l2_proj_errs_snaps_hapod, l2_proj_errs_random_hapod, l2_red_errs_hapod, memory_hapod, svals_hapod = [[[] for i in hapod_types] for i in range(9)]
num_modes_pod, time_pod, time_data_gen_pod, l2_proj_errs_snaps_pod, l2_proj_errs_random_pod, l2_red_errs_pod, memory_pod, svals_pod = [[] for i in range(8)]

filenames_hapod = ["/scratch/tmp/l_tobi01/pickled_bases/hapod_" + hapod_type + "_" + str(grid_size) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega) for hapod_type in hapod_types]
filename_pod = "/scratch/tmp/l_tobi01/pickled_bases/pod_gridsize_" + str(grid_size)
x_axis = []
logfilename = "logfile_tolerance_plot" + str(grid_size) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega)
logfile = open(logfilename, "w", 0)

for exponent in range(2, 6):
    tol1 = 0.1 ** exponent
    tol = tol1 * grid_size
    x_axis.append(tol1)
    b = HapodBasics(grid_size, chunk_size, epsilon_ast=tol, omega=omega)
    if b.rank_world == 0:
        print("Current tolerance:", tol)
        logfile.write("Current tolerance:" + str(tol) + "\n")

    for num, hapod_type in enumerate(hapod_types):
        try:
            f = open(filenames_hapod[num] + "_tol_" + str(tol), "rb")
            if b.rank_world == 0:
                basis, svals, total_num_snapshots, elapsed, max_vecs_before_pod, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(f) 
            else:
                basis, svals, total_num_snapshots, elapsed, max_vecs_before_pod, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*9
            f.close()    
        except (OSError, IOError) as e:
    	    start = timer()
            basis, svals, total_num_snapshots, b, max_vecs_before_pod, max_local_modes = hapod_timechunk_wise(grid_size, chunk_size, tol, log=False, bcast_modes=False, 
                                                                                                              calculate_max_local_modes=True, omega=omega, 
                                                                                                              incremental_pod=is_incremental[num],
                                                                                                              nodes_binary_tree=binary_tree[num])
            elapsed = timer() - start
            basis = b.shared_memory_bcast_modes(basis)
            l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
            if b.rank_world == 0:
                l2_proj_err_snaps /= grid_size
            l2_red_err_list_random, l2_proj_err_list_random, _, _ = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
            l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            b.comm_world.Barrier()
            if b.rank_world == 0:
                f = open(filenames_hapod[num] + "_tol_" + str(tol), "wb")
                tmp = basis, svals, total_num_snapshots, elapsed, max_vecs_before_pod, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
                pickle.dump(tmp, f)
                f.close()
        time_hapod[num].append(elapsed)
        num_modes_hapod[num].append(len(basis))
        max_num_local_modes_hapod[num].append(max_local_modes)
        max_vecs_before_pod_hapod[num].append(max_vecs_before_pod)
        l2_proj_errs_snaps_hapod[num].append(l2_proj_err_snaps)
        l2_proj_errs_random_hapod[num].append(l2_proj_err_random)
        l2_red_errs_hapod[num].append(l2_red_err_random)
        memory_hapod[num].append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
        svals_hapod[num].append(svals)
        for win in b.allocated_wins:
            win.Free()
        b.allocated_wins = []
        if b.rank_world == 0:
            logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                           str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        if b.rank_world == 0:
            print(hapod_type + "done \n")

    try:
        g = open(filename_pod + "_tol_" + str(tol), "rb")
        if b.rank_world == 0:
            basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(g) 
        else:
            basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*8
        g.close()    
    except (OSError, IOError) as e:
        start = timer()
        basis, svals, total_num_snapshots, b, elapsed_data_gen, elapsed_pod = boltzmann_standard_pod(grid_size, tol, log=False, bcast_modes=False)
        elapsed = timer() - start
        basis = b.shared_memory_bcast_modes(basis)
        l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
        if b.rank_world == 0:
            l2_proj_err_snaps /= grid_size
        l2_red_err_list_random, l2_proj_err_list_random, _, _ = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
        l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
        l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
        b.comm_world.Barrier()
        if b.rank_world == 0:
            g = open(filename_pod + "_tol_" + str(tol), "wb")
            tmp = basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
            pickle.dump(tmp, g)
            g.close()
    time_pod.append(elapsed_pod + elapsed_data_gen)
    time_data_gen_pod.append(elapsed_data_gen)
    num_modes_pod.append(len(basis))
    l2_proj_errs_snaps_pod.append(l2_proj_err_snaps)
    l2_proj_errs_random_pod.append(l2_proj_err_random)
    l2_red_errs_pod.append(l2_red_err_random)
    memory_pod.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
    svals_pod.append(svals)
    for win in b.allocated_wins:
        win.Free()
    b.allocated_wins = []
    del basis
    if b.rank_world == 0:
        logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
    if b.rank_world == 0:
            print("pod done\n")


from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')

if b.rank_world == 0:
    plt.figure()

    for num, hapod_type in enumerate(hapod_types):
        plt.semilogx(x_axis, num_modes_hapod[num], label=hapod_type)
    plt.semilogx(x_axis, num_modes_pod, label="pod")
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("POD basis size")
    plt.legend(loc='upper left')

    filename = "num_modes_gridsize_%d_chunk_%d_omega_%g" % (b.gridsize, chunk_size, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = [x_axis, num_modes_pod] 
    for num in range(len(hapod_types)):
        data += [num_modes_hapod[num], max_num_local_modes_hapod[num], max_vecs_before_pod_hapod[num]]
    data = np.array(data) 
    data2 = np.array([[0 for _ in range(len(x_axis))]] + [num_modes_pod]*(3*len(hapod_types)+1))
    data_diff = data - data2
    data = data.T
    data_diff = data_diff.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%f']*(3*len(hapod_types)+2))
    with open(filename + "_diff.dat", 'w') as f:
        np.savetxt(f, data_diff, fmt=['%f']*(3*len(hapod_types)+2))


    plt.figure()

    for num, hapod_type in enumerate(hapod_types):
        plt.loglog(x_axis, l2_proj_errs_snaps_hapod[num], label="proj_snaps_" + hapod_type)
        plt.loglog(x_axis, l2_proj_errs_random_hapod[num], label="proj_random_" + hapod_type)
        plt.loglog(x_axis, l2_red_errs_hapod[num], label="reduced_" + hapod_type)
    plt.gca().set_color_cycle(None) # restart color cycle
    plt.loglog(x_axis, l2_proj_errs_snaps_pod, label="proj_snaps_pod", linestyle=':')
    plt.loglog(x_axis, l2_proj_errs_random_pod, label="proj_random_pod", linestyle=':')
    plt.loglog(x_axis, l2_red_errs_pod, label="reduced_pod", linestyle=':')
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("L2 mean error")
    plt.legend(loc='lower left', prop=fontP)

    filename = "l2_mean_errs_gridsize_%d_chunk_%d_omega_%g" % (b.gridsize, chunk_size, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = [x_axis, l2_proj_errs_snaps_pod, l2_proj_errs_random_pod, l2_red_errs_pod]
    for num in range(len(hapod_types)):
        data += [l2_proj_errs_snaps_hapod[num], l2_proj_errs_random_hapod[num], l2_red_errs_hapod[num]]
    data = np.array(data)
                    
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%f']*(3*len(hapod_types)+4))


    plt.figure()

    for num, hapod_type in enumerate(hapod_types):
        plt.semilogx(x_axis, time_hapod[num], label=hapod_type)
    plt.semilogx(x_axis, time_pod, label="pod")
    plt.semilogx(x_axis, time_data_gen_pod, label="pod data")
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Elapsed time (seconds)")
    plt.legend(loc='upper right')

    filename = "time_gridsize_%d_chunk_%d_omega_%g" % (b.gridsize, chunk_size, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_axis, time_pod, time_data_gen_pod] + [time_hapod[num] for num in range(len(hapod_types))])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%f']*(len(hapod_types)+3))

    plt.figure()

    for num, hapod_type in enumerate(hapod_types):
        plt.semilogx(x_axis, memory_hapod[num], label=hapod_type)
    plt.semilogx(x_axis, memory_pod, label="pod")
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Memory usage (GB)")
    plt.legend(loc='upper right')

    plt.savefig("memory_gridsize_" + str(b.gridsize) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega) + ".png")
    plt.clf()


    plt.figure()

    x_end = min(min([len(svals_hapod[num][-1]) for num in range(len(hapod_types))]), len(svals_pod[-1]))
    x_counts = range(1,x_end+1)
    for num, hapod_type in enumerate(hapod_types):
        plt.semilogy(x_counts, svals_hapod[num][-1][0:x_end], label=hapod_type)
    plt.semilogy(x_counts, svals_pod[-1][0:x_end], label="pod")

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Value")
    plt.legend(loc='upper right')

    filename = "svals_gridsize_%d_chunk_%d_omega_%g_tol_%g_norm" % (b.gridsize, chunk_size, omega, x_axis[-1])

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_counts, svals_pod[-1][0:x_end]] + [svals_hapod[num][-1][0:x_end] for num in range(len(hapod_types))])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%d'] + ['%.15g']*(len(hapod_types)+1))

    logfile.close()
