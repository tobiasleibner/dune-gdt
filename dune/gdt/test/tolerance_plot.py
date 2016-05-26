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
from boltzmann_RAPOD_timechunk_wise_stephans_pod import rapod_timechunk_wise
from boltzmann_standard_pod import boltzmann_standard_pod
from boltzmann_mor_with_basis_generation import calculate_mean_l2_error_for_random_samples
from boltzmann_rapod_only_on_trajectory import rapod_only_on_trajectory
from Hapod import HapodBasics

grid_size = int(sys.argv[1])
chunk_size = int(sys.argv[2])

num_modes_hapod_tcw = []
time_hapod_tcw = []
l2_proj_errs_snaps_hapod_tcw = []
l2_proj_errs_random_hapod_tcw = []
l2_red_errs_hapod_tcw = []
timings_1_hapod_tcw = []
timings_2_hapod_tcw = []
timings_3_hapod_tcw = []
timings_4_hapod_tcw = []
timings_5_hapod_tcw = []
memory_hapod_tcw = []

num_modes_rapod_traj = []
time_rapod_traj = []
l2_proj_errs_snaps_rapod_traj = []
l2_proj_errs_random_rapod_traj = []
l2_red_errs_rapod_traj = []
timings_1_rapod_traj = []
timings_2_rapod_traj = []
timings_3_rapod_traj = []
timings_4_rapod_traj = []
timings_5_rapod_traj = []
memory_rapod_traj = []

num_modes_pod = []
time_pod = []
l2_proj_errs_snaps_pod = []
l2_proj_errs_random_pod = []
l2_red_errs_pod = []
timings_1_pod = []
timings_2_pod = []
timings_3_pod = []
timings_4_pod = []
timings_5_pod = []
memory_pod = []


filename_tcw = "/scratch/tmp/l_tobi01/pickled_bases/tcw_gridsize_" + str(grid_size) + "_chunk_" + str(chunk_size)
filename_traj = "/scratch/tmp/l_tobi01/pickled_bases/traj_gridsize_" + str(grid_size) + "_chunk_" + str(chunk_size)
filename_pod = "/scratch/tmp/l_tobi01/pickled_bases/pod_gridsize_" + str(grid_size)

x_axis = []

logfilename = "logfile_tolerance_plot" + str(grid_size) + "_chunk_" + str(chunk_size)
logfile = open(logfilename, "w", 0)

for exponent in range(1, 9):
    tol = 0.1 ** exponent
    x_axis.append(tol)
    b = HapodBasics(grid_size, chunk_size, epsilon_ast=tol)
    if b.rank_world == 0:
        print("Current tolerance:", tol)
        logfile.write("Current tolerance:" + str(tol) + "\n")

    try:
        f = open(filename_tcw + "_tol_" + str(tol), "rb")
        if b.rank_world == 0:
            basis, svals, total_num_snapshots, elapsed, timings, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(f) 
        else:
            basis, svals, total_num_snapshots, elapsed, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*7
            timings = b.zero_timings_dict()
        f.close()    
    except (OSError, IOError) as e:
	start = timer()
        basis, svals, total_num_snapshots, b, timings = rapod_timechunk_wise(grid_size, chunk_size, tol, log=False, scatter_modes=False)
        elapsed = timer() - start
        basis = b.shared_memory_scatter_modes(basis)
        l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
        l2_red_err_list_random, l2_proj_err_list_random = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
        l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots) if b.rank_world == 0 else None
        l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots) if b.rank_world == 0 else None
        b.comm_world.Barrier()
        if b.rank_world == 0:
            f = open(filename_tcw + "_tol_" + str(tol), "wb")
            tmp = basis, svals, total_num_snapshots, elapsed, timings, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
            pickle.dump(tmp, f)
            f.close()
    time_hapod_tcw.append(elapsed)
    num_modes_hapod_tcw.append(len(basis))
    l2_proj_errs_snaps_hapod_tcw.append(l2_proj_err_snaps)
    l2_proj_errs_random_hapod_tcw.append(l2_proj_err_random)
    l2_red_errs_hapod_tcw.append(l2_red_err_random)
    timings_1_hapod_tcw.append(timings["gramian"])
    timings_2_hapod_tcw.append(timings["EV decomp"])
    timings_3_hapod_tcw.append(timings["left-sing vecs"])
    timings_4_hapod_tcw.append(timings["reorthonormalizing"])
    timings_5_hapod_tcw.append(timings["check"])
    memory_hapod_tcw.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
    del basis

    if b.rank_world == 0:
        logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")

    if b.rank_world == 0:
        print("rapod_tcw_done\n")

    try:
        h = open(filename_traj + "_tol_" + str(tol), "rb")
        if b.rank_world == 0:
            basis, svals, total_num_snapshots, elapsed, timings, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(h) 
        else:
            basis, svals, total_num_snapshots, elapsed, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*7
            timings = b.zero_timings_dict()
        h.close()    
    except (OSError, IOError) as e:
	start = timer()
        basis, svals, total_num_snapshots, b, timings = rapod_only_on_trajectory(grid_size, chunk_size, tol, log=False, scatter_modes=False)
        elapsed = timer() - start
        basis = b.shared_memory_scatter_modes(basis)
        l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
        l2_red_err_list_random, l2_proj_err_list_random = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
        l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots) if b.rank_world == 0 else None
        l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots) if b.rank_world == 0 else None
        b.comm_world.Barrier()
        if b.rank_world == 0:
            h = open(filename_traj + "_tol_" + str(tol), "wb")
            tmp = basis, svals, total_num_snapshots, elapsed, timings, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
            pickle.dump(tmp, h)
            h.close()
    time_rapod_traj.append(elapsed)
    num_modes_rapod_traj.append(len(basis))
    l2_proj_errs_snaps_rapod_traj.append(l2_proj_err_snaps)
    l2_proj_errs_random_rapod_traj.append(l2_proj_err_random)
    l2_red_errs_rapod_traj.append(l2_red_err_random)
    timings_1_rapod_traj.append(timings["gramian"])
    timings_2_rapod_traj.append(timings["EV decomp"])
    timings_3_rapod_traj.append(timings["left-sing vecs"])
    timings_4_rapod_traj.append(timings["reorthonormalizing"])
    timings_5_rapod_traj.append(timings["check"])
    memory_rapod_traj.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
    del basis

    if b.rank_world == 0:
        logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")

    if b.rank_world == 0:
        print("rapod_traj_done\n")

    try:
        g = open(filename_pod + "_tol_" + str(tol), "rb")
        if b.rank_world == 0:
            basis, svals, total_num_snapshots, elapsed, timings, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(g) 
        else:
            basis, svals, total_num_snapshots, elapsed, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*7
            timings = b.zero_timings_dict()
        g.close()    
    except (OSError, IOError) as e:
	start = timer()
        basis, svals, total_num_snapshots, b, timings = boltzmann_standard_pod(grid_size, tol, log=False, scatter_modes=False)
        elapsed = timer() - start
        basis = b.shared_memory_scatter_modes(basis)
        l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
        l2_red_err_list_random, l2_proj_err_list_random = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
        l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots) if b.rank_world == 0 else None
        l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots) if b.rank_world == 0 else None
        b.comm_world.Barrier()
        if b.rank_world == 0:
            g = open(filename_pod + "_tol_" + str(tol), "wb")
            tmp = basis, svals, total_num_snapshots, elapsed, timings, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
            pickle.dump(tmp, g)
            g.close()
    time_pod.append(elapsed)
    num_modes_pod.append(len(basis))
    l2_proj_errs_snaps_pod.append(l2_proj_err_snaps)
    l2_proj_errs_random_pod.append(l2_proj_err_random)
    l2_red_errs_pod.append(l2_red_err_random)
    timings_1_pod.append(timings["gramian"])
    timings_2_pod.append(timings["EV decomp"])
    timings_3_pod.append(timings["left-sing vecs"])
    timings_4_pod.append(timings["reorthonormalizing"])
    timings_5_pod.append(timings["check"])
    memory_pod.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
    del basis
    
    if b.rank_world == 0:
        logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
    

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')

if b.rank_world == 0:
    plt.figure()

    plt.semilogx(x_axis, num_modes_hapod_tcw, label="tcw")
    plt.semilogx(x_axis, num_modes_rapod_traj, label="traj")
    plt.semilogx(x_axis, num_modes_pod, label="pod")
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("POD basis size")
    plt.legend(loc='lower right')

    plt.savefig("num_modes_gridsize_" + str(b.gridsize) + "_chunk_" + str(chunk_size) + ".png")
    plt.clf()


    plt.figure()

    plt.loglog(x_axis, l2_proj_errs_snaps_hapod_tcw, label="proj_snaps_tcw")
    plt.loglog(x_axis, l2_proj_errs_random_hapod_tcw, label="proj_random_tcw")
    plt.loglog(x_axis, l2_red_errs_hapod_tcw, label="reduced_tcw")
    plt.gca().set_color_cycle(None) # restart color cycle
    plt.loglog(x_axis, l2_proj_errs_snaps_rapod_traj, label="proj_snaps_traj", linestyle='--')
    plt.loglog(x_axis, l2_proj_errs_random_rapod_traj, label="proj_random_traj", linestyle='--')
    plt.loglog(x_axis, l2_red_errs_rapod_traj, label="reduced_traj", linestyle='--')
    plt.gca().set_color_cycle(None) # restart color cycle
    plt.loglog(x_axis, l2_proj_errs_snaps_pod, label="proj_snaps_pod", linestyle=':')
    plt.loglog(x_axis, l2_proj_errs_random_pod, label="proj_random_pod", linestyle=':')
    plt.loglog(x_axis, l2_red_errs_pod, label="reduced_pod", linestyle=':')
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("L2 mean error")
    plt.legend(loc='lower left', prop=fontP)

    plt.savefig("l2_mean_errs_gridsize_" + str(b.gridsize) + "_chunk_" + str(chunk_size) + ".png")
    plt.clf()


    plt.figure()

    plt.semilogx(x_axis, time_hapod_tcw, label="tcw")
    plt.semilogx(x_axis, time_rapod_traj, label="traj")
    plt.semilogx(x_axis, time_pod, label="pod")
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Elapsed time (seconds)")
    plt.legend(loc='upper right')

    plt.savefig("time_gridsize_" + str(b.gridsize) + "_chunk_" + str(chunk_size) + ".png")
    plt.clf()


    plt.figure()

    plt.semilogx(x_axis, time_hapod_tcw, label="tcw_total")
    plt.semilogx(x_axis, timings_1_hapod_tcw, label="tcw_gramian")
    plt.semilogx(x_axis, timings_2_hapod_tcw, label="tcw_ev_decomp")
    plt.semilogx(x_axis, timings_3_hapod_tcw, label="tcw_leftsing")
    plt.semilogx(x_axis, timings_4_hapod_tcw, label="tcw_reortho")
    plt.semilogx(x_axis, timings_5_hapod_tcw, label="tcw_check")
    plt.gca().set_color_cycle(None) # restart color cycle
    plt.semilogx(x_axis, time_rapod_traj, label="traj_total", linestyle='--')
    plt.semilogx(x_axis, timings_1_rapod_traj, label="traj_gramian", linestyle='--')
    plt.semilogx(x_axis, timings_2_rapod_traj, label="traj_ev_decomp", linestyle='--')
    plt.semilogx(x_axis, timings_3_rapod_traj, label="traj_leftsing", linestyle='--')
    plt.semilogx(x_axis, timings_4_rapod_traj, label="traj_reortho", linestyle='--')
    plt.semilogx(x_axis, timings_5_rapod_traj, label="traj_check", linestyle='--') 
    plt.gca().set_color_cycle(None) # restart color cycle
    plt.semilogx(x_axis, time_pod, label="pod_total", linestyle=':')
    plt.semilogx(x_axis, timings_1_pod, label="pod_gramian", linestyle=':')
    plt.semilogx(x_axis, timings_2_pod, label="pod_ev_decomp", linestyle=':')
    plt.semilogx(x_axis, timings_3_pod, label="pod_leftsing", linestyle=':')
    plt.semilogx(x_axis, timings_4_pod, label="pod_reortho", linestyle=':')
    plt.semilogx(x_axis, timings_5_pod, label="pod_check", linestyle=':') 

    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Elapsed time (seconds)")
    plt.legend(loc=(0., 0.), prop=fontP)

    plt.savefig("timings_gridsize_" + str(b.gridsize) + "_chunk_" + str(chunk_size) + ".png")
    plt.clf()

    
    plt.figure()

    plt.semilogx(x_axis, memory_hapod_tcw, label="tcw")
    plt.semilogx(x_axis, memory_rapod_traj, label="traj")
    plt.semilogx(x_axis, memory_pod, label="pod")
    plt.gca().invert_xaxis()

    plt.xlabel("L2 mean error bound for POD")
    plt.ylabel("Memory usage (GB)")
    plt.legend(loc='upper right')

    plt.savefig("memory_gridsize_" + str(b.gridsize) + "_chunk_" + str(chunk_size) + ".png")
    plt.clf()


    logfile.close()
