import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer
from mpi4py import MPI
import resource
import math

from pymor.basic import *
from boltzmann.wrapper import DuneDiscretization
from boltzmann_RAPOD_timechunk_wise_stephans_pod import rapod_timechunk_wise
from boltzmann_standard_pod import boltzmann_standard_pod
from boltzmann_mor_with_basis_generation import calculate_mean_l2_error_for_random_samples
from boltzmann_rapod_only_on_trajectory import rapod_only_on_trajectory
from modred_test import modred_pod
from modred_in_memory import modred_pod_in_memory
from Hapod import HapodBasics
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')


initial_grid_size = int(sys.argv[1])
chunk_size = int(sys.argv[2])
omega = float(sys.argv[3])
initial_tol = float(sys.argv[4])

num_modes_hapod_tcw = []
max_num_local_modes_hapod_tcw = []
time_hapod_tcw = []
l2_proj_errs_snaps_hapod_tcw = []
l2_proj_errs_random_hapod_tcw = []
l2_red_errs_hapod_tcw = []
memory_hapod_tcw = []
svals_hapod_tcw = []


num_modes_rapod_traj = []
max_num_local_modes_rapod_traj = []
time_rapod_traj = []
l2_proj_errs_snaps_rapod_traj = []
l2_proj_errs_random_rapod_traj = []
l2_red_errs_rapod_traj = []
memory_rapod_traj = []
svals_rapod_traj = []

num_modes_pod = []
time_pod = []
data_gen_pod = []
l2_proj_errs_snaps_pod = []
l2_proj_errs_random_pod = []
l2_red_errs_pod = []
memory_pod = []
svals_pod = []

num_modes_modred = []
time_modred = []
data_gen_modred = []
l2_proj_errs_snaps_modred = []
l2_proj_errs_random_modred = []
l2_red_errs_modred = []
memory_modred = []
eigvals_modred = []

num_modes_modred_in_mem = []
time_modred_in_mem = []
data_gen_modred_in_mem = []
l2_proj_errs_snaps_modred_in_mem = []
l2_proj_errs_random_modred_in_mem = []
l2_red_errs_modred_in_mem = []
memory_modred_in_mem = []
eigvals_modred_in_mem = []

x_axis = []

logfilename = "logfile_gridsize_plot" + str(initial_grid_size) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega) + "_initial_tol_" + str(initial_tol)
logfile = open(logfilename, "w")

grid_size = initial_grid_size
for grid_size in (5, 10, 20):
    tol = initial_tol * grid_size
    filename_tcw = "/scratch/tmp/l_tobi01/pickled_bases/tcw_py3_gridsize_" + str(grid_size) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega) + "_tol_" + str(tol)
    filename_traj = "/scratch/tmp/l_tobi01/pickled_bases/traj_py3_gridsize_" + str(grid_size) + "_chunk_" + str(chunk_size) + "_omega_" + str(omega) + "_tol_" + str(tol)
    filename_pod = "/scratch/tmp/l_tobi01/pickled_bases/pod_py3_gridsize_" + str(grid_size) + "_tol_" + str(tol)
    filename_modred = "/scratch/tmp/l_tobi01/pickled_bases/modred_py3_gridsize_" + str(grid_size) + "_tol_" + str(tol)
    filename_modred_in_mem = "/scratch/tmp/l_tobi01/pickled_bases/modred_py3_in_mem_gridsize_" + str(grid_size) + "_tol_" + str(tol)

    x_axis.append(grid_size)
    b = HapodBasics(grid_size, chunk_size, epsilon_ast=tol, omega=omega)
    if b.rank_world == 0:
        print("Current grid size:", grid_size)
        logfile.write("Current grid_size:" + str(tol) + "\n")

    try:
        f = open(filename_tcw, "rb")
        if b.rank_world == 0:
            basis, svals, total_num_snapshots, elapsed, max_vecs, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(f)
        else:
            basis, svals, total_num_snapshots, elapsed, max_vecs, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*9
        f.close()
    except (FileNotFoundError, OSError, IOError) as e:
        start = timer()
        basis, svals, total_num_snapshots, b, max_vecs, max_local_modes = rapod_timechunk_wise(grid_size, chunk_size, tol, log=False, scatter_modes=False,
                                                                                                        calculate_max_local_modes= True, omega=omega)
        elapsed = timer() - start
        basis = b.shared_memory_scatter_modes(basis)
        l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
        if b.rank_world == 0:
            l2_proj_err_snaps /= grid_size
        l2_red_err_list_random, l2_proj_err_list_random = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
        l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
        l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
        b.comm_world.Barrier()
        if b.rank_world == 0:
            f = open(filename_tcw, "wb")
            tmp = basis, svals, total_num_snapshots, elapsed, max_vecs, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
            pickle.dump(tmp, f)
            f.close()
    time_hapod_tcw.append(elapsed)
    num_modes_hapod_tcw.append(len(basis))
    max_num_local_modes_hapod_tcw.append(max_local_modes)
    l2_proj_errs_snaps_hapod_tcw.append(l2_proj_err_snaps)
    l2_proj_errs_random_hapod_tcw.append(l2_proj_err_random)
    l2_red_errs_hapod_tcw.append(l2_red_err_random)
    memory_hapod_tcw.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
    svals_hapod_tcw.append(svals)
    del basis

    if b.rank_world == 0:
        logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                      str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")

    if b.rank_world == 0:
        print("rapod_tcw_done\n")

    if grid_size <= 100:
        try:
            h = open(filename_traj, "rb")
            if b.rank_world == 0:
                basis, svals, total_num_snapshots, elapsed, max_vecs, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(h)
            else:
                basis, svals, total_num_snapshots, elapsed, max_vecs, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*9
            h.close()
        except (FileNotFoundError, OSError, IOError) as e:
            start = timer()
            basis, svals, total_num_snapshots, b, max_vecs, max_local_modes = rapod_only_on_trajectory(grid_size, chunk_size, tol, log=False, scatter_modes=False,
                                                                                                  calculate_max_local_modes=True, omega=omega)
            elapsed = timer() - start
            basis = b.shared_memory_scatter_modes(basis)
            l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
            if b.rank_world == 0:
                l2_proj_err_snaps /= grid_size
            l2_red_err_list_random, l2_proj_err_list_random = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
            l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            b.comm_world.Barrier()
            if b.rank_world == 0:
                h = open(filename_traj, "wb")
                tmp = basis, svals, total_num_snapshots, elapsed, max_vecs, max_local_modes, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
                pickle.dump(tmp, h)
                h.close()
        time_rapod_traj.append(elapsed)
        num_modes_rapod_traj.append(len(basis))
        max_num_local_modes_rapod_traj.append(max_local_modes)
        l2_proj_errs_snaps_rapod_traj.append(l2_proj_err_snaps)
        l2_proj_errs_random_rapod_traj.append(l2_proj_err_random)
        l2_red_errs_rapod_traj.append(l2_red_err_random)
        memory_rapod_traj.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
        svals_rapod_traj.append(svals)
        del basis

        if b.rank_world == 0:
            logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                           str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")

        if b.rank_world == 0:
            print("rapod_traj_done\n")

    if grid_size <= 40:
        try:
            g = open(filename_pod, "rb")
            if b.rank_world == 0:
                basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(g)
            else:
                basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*8
            g.close()
        except (FileNotFoundError, OSError, IOError) as e:
            basis, svals, total_num_snapshots, b, elapsed_data_gen, elapsed_pod = boltzmann_standard_pod(grid_size, tol, log=False, scatter_modes=False)
            basis = b.shared_memory_scatter_modes(basis)
            l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
            if b.rank_world == 0:
                l2_proj_err_snaps /= grid_size
            l2_red_err_list_random, l2_proj_err_list_random = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
            l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            b.comm_world.Barrier()
            if b.rank_world == 0:
                g = open(filename_pod, "wb")
                tmp = basis, svals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
                pickle.dump(tmp, g)
                g.close()
        time_pod.append(elapsed_data_gen + elapsed_pod)
        data_gen_pod.append(elapsed_data_gen)
        num_modes_pod.append(len(basis))
        l2_proj_errs_snaps_pod.append(l2_proj_err_snaps)
        l2_proj_errs_random_pod.append(l2_proj_err_random)
        l2_red_errs_pod.append(l2_red_err_random)
        memory_pod.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
        svals_pod.append(svals)
        del basis

        if b.rank_world == 0:
            logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                           str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")\

        if b.rank_world == 0:
            print("pod_done\n")

        svals_pod = b.comm_world.bcast(svals_pod, root=0)
        num_modes_pod = b.comm_world.bcast(num_modes_pod, root=0)
    else:
        num_modes_hapod_tcw = b.comm_world.bcast(num_modes_hapod_tcw, root=0)

    if grid_size <= 40:
        try:
            f = open(filename_modred, "rb")
            if b.rank_world == 0:
                basis, eigvals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(f)
            else:
                basis, eigvals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*8
            f.close()
        except (FileNotFoundError, OSError, IOError) as e:
            basis, eigvals, total_num_snapshots, b, elapsed_data_gen, elapsed_pod = modred_pod(grid_size, chunk_size, num_modes_pod[-1] if grid_size <= 40 else num_modes_hapod_tcw[-1], log=False, scatter_modes=False)
            basis = b.shared_memory_scatter_modes(basis)
            l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
            if b.rank_world == 0:
                l2_proj_err_snaps /= grid_size
            l2_red_err_list_random, l2_proj_err_list_random = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
            l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            b.comm_world.Barrier()
            if b.rank_world == 0:
                f = open(filename_modred, "wb")
                tmp = basis, eigvals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
                pickle.dump(tmp, f)
                f.close()
        time_modred.append(elapsed_data_gen + elapsed_pod)
        data_gen_modred.append(elapsed_data_gen)
        num_modes_modred.append(len(basis))
        l2_proj_errs_snaps_modred.append(l2_proj_err_snaps)
        l2_proj_errs_random_modred.append(l2_proj_err_random)
        l2_red_errs_modred.append(l2_red_err_random)
        memory_modred.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
        eigvals_modred.append(eigvals)
        del basis

    if b.rank_world == 0:
        logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")


    if b.rank_world == 0:
        print("modred_done\n")

    if grid_size <= 20:
        try:
            f = open(filename_modred_in_mem, "rb")
            if b.rank_world == 0:
                basis, eigvals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = pickle.load(f)
            else:
                basis, eigvals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random = ([None],)*8
            f.close()
        except (FileNotFoundError, OSError, IOError) as e:
            basis, eigvals, total_num_snapshots, b, elapsed_data_gen, elapsed_pod = modred_pod_in_memory(grid_size, chunk_size, num_modes_pod[-1], log=False, scatter_modes=False)
            basis = b.shared_memory_scatter_modes(basis)
            l2_proj_err_snaps = b.calculate_total_projection_error(basis, total_num_snapshots)
            if b.rank_world == 0:
                l2_proj_err_snaps /= grid_size
            l2_red_err_list_random, l2_proj_err_list_random = calculate_mean_l2_error_for_random_samples(basis, b, seed=b.rank_world, write_plot=False, mean_error=False)
            l2_proj_err_random = np.sqrt(np.sum(l2_proj_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            l2_red_err_random = np.sqrt(np.sum(l2_red_err_list_random) / total_num_snapshots)/grid_size if b.rank_world == 0 else None
            b.comm_world.Barrier()
            if b.rank_world == 0:
                f = open(filename_modred_in_mem, "wb")
                tmp = basis, eigvals, total_num_snapshots, elapsed_data_gen, elapsed_pod, l2_proj_err_snaps, l2_proj_err_random, l2_red_err_random
                pickle.dump(tmp, f)
                f.close()
        time_modred_in_mem.append(elapsed_pod)
        data_gen_modred_in_mem.append(elapsed_data_gen)
        num_modes_modred_in_mem.append(len(basis))
        l2_proj_errs_snaps_modred_in_mem.append(l2_proj_err_snaps)
        l2_proj_errs_random_modred_in_mem.append(l2_proj_err_random)
        l2_red_errs_modred_in_mem.append(l2_red_err_random)
        memory_modred_in_mem.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2)
        eigvals_modred_in_mem.append(eigvals)
        del basis

        if b.rank_world == 0:
            logfile.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                           str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")


if b.rank_world == 0:
    plt.figure()

    if len(num_modes_pod) < len(num_modes_hapod_tcw):
        num_modes_pod.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_pod)))
    if len(num_modes_modred) < len(num_modes_hapod_tcw):
        num_modes_modred.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_modred)))

    plt.plot(x_axis, num_modes_hapod_tcw, label="tcw")
    plt.plot(x_axis, num_modes_rapod_traj, label="traj")
    plt.plot(x_axis, max_num_local_modes_hapod_tcw, label="tcw_local")
    plt.plot(x_axis, max_num_local_modes_rapod_traj, label="traj_local")
    plt.plot(x_axis, num_modes_pod, label="pod")

    plt.xlabel("Grid size")
    plt.ylabel("POD basis size")
    plt.legend(loc='upper left')

    filename = "num_modes_initialgridsize_%d_tol_%g_chunk_%d_omega_%g_norm" % (initial_grid_size, tol, chunk_size, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_axis, num_modes_pod, num_modes_rapod_traj, num_modes_hapod_tcw, max_num_local_modes_rapod_traj, max_num_local_modes_hapod_tcw])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%d','%d','%d','%d','%d','%d'])


    plt.figure()

    if len(num_modes_pod) < len(num_modes_hapod_tcw):
        l2_proj_errs_snaps_pod.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_pod)))
        l2_proj_errs_random_pod.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_pod)))
        l2_red_errs_pod.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_pod)))
    if len(num_modes_modred) < len(num_modes_hapod_tcw):
        l2_proj_errs_snaps_modred.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_modred)))
        l2_proj_errs_random_modred.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_modred)))
        l2_red_errs_modred.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_modred)))

    plt.semilogy(x_axis, l2_proj_errs_snaps_hapod_tcw, label="proj_snaps_tcw")
    plt.semilogy(x_axis, l2_proj_errs_random_hapod_tcw, label="proj_random_tcw")
    plt.semilogy(x_axis, l2_red_errs_hapod_tcw, label="reduced_tcw")
#    plt.gca().set_color_cycle(None) # restart color cycle
    plt.semilogy(x_axis, l2_proj_errs_snaps_rapod_traj, label="proj_snaps_traj", linestyle='--')
    plt.semilogy(x_axis, l2_proj_errs_random_rapod_traj, label="proj_random_traj", linestyle='--')
    plt.semilogy(x_axis, l2_red_errs_rapod_traj, label="reduced_traj", linestyle='--')
    plt.gca().set_color_cycle(None) # restart color cycle
    plt.semilogy(x_axis, l2_proj_errs_snaps_pod, label="proj_snaps_pod", linestyle=':')
    plt.semilogy(x_axis, l2_proj_errs_random_pod, label="proj_random_pod", linestyle=':')
    plt.semilogy(x_axis, l2_red_errs_pod, label="reduced_pod", linestyle=':')
#    plt.gca().set_color_cycle(None) # restart color cycle
    plt.semilogy(x_axis, l2_proj_errs_snaps_modred, label="proj_snaps_modred", linestyle='-.')
    plt.semilogy(x_axis, l2_proj_errs_random_modred, label="proj_random_modred", linestyle='-.')
    plt.semilogy(x_axis, l2_red_errs_modred, label="reduced_modred", linestyle='-.')

    plt.xlabel("Gridsize")
    plt.ylabel("L2 mean error")
    plt.legend(loc='lower left', prop=fontP)

    filename = "l2_mean_errs_initialgridsize_%d_tol_%g_chunk_%d_omega_%g_norm" % (initial_grid_size, tol, chunk_size, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_axis, l2_proj_errs_snaps_hapod_tcw, l2_proj_errs_random_hapod_tcw, l2_red_errs_hapod_tcw,
                     l2_proj_errs_snaps_rapod_traj, l2_proj_errs_random_rapod_traj, l2_red_errs_rapod_traj,
                     l2_proj_errs_snaps_pod, l2_proj_errs_random_pod, l2_red_errs_pod,
                     l2_proj_errs_snaps_modred, l2_proj_errs_random_modred, l2_red_errs_modred])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f'])


    plt.figure()

    if len(num_modes_pod) < len(num_modes_hapod_tcw):
        time_pod.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_pod)))
    if len(time_modred_in_mem) < len(time_hapod_tcw):
        time_modred_in_mem.extend([0.] * (len(time_hapod_tcw) - len(time_modred_in_mem)))
    if len(time_modred) < len(time_hapod_tcw):
        time_modred.extend([0.] * (len(time_hapod_tcw) - len(time_modred)))

    plt.plot(x_axis, time_hapod_tcw, label="tcw")
    plt.plot(x_axis, time_rapod_traj, label="traj")
    plt.plot(x_axis, time_pod, label="pod")
    plt.plot(x_axis, time_modred, label="modred")
    plt.plot(x_axis, time_modred_in_mem, label="modred_in_mem")

    plt.xlabel("Grid size")
    plt.ylabel("Elapsed time (seconds)")
    plt.legend(loc='upper left')

    filename = "time_initialgridsize_%d_tol_%g_chunk_%d_omega_%g_norm" % (initial_grid_size, tol, chunk_size, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_axis, time_pod, time_modred, time_modred_in_mem, time_rapod_traj, time_hapod_tcw])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%d','%f','%f','%f','%f','%f'])

    plt.figure()

    if len(num_modes_pod) < len(num_modes_hapod_tcw):
        data_gen_pod.extend([0.] * (len(num_modes_hapod_tcw) - len(num_modes_pod)))
    if len(time_modred_in_mem) < len(time_hapod_tcw):
        data_gen_modred_in_mem.extend([0.] * (len(time_hapod_tcw) - len(time_modred_in_mem)))
    if len(time_modred) < len(time_hapod_tcw):
        data_gen_modred.extend([0.] * (len(time_hapod_tcw) - len(time_modred)))

    plt.plot(x_axis, data_gen_pod, label="pod")
    plt.plot(x_axis, data_gen_modred, label="modred")
    plt.plot(x_axis, data_gen_modred_in_mem, label="modred_in_mem")

    plt.xlabel("Grid size")
    plt.ylabel("Elapsed time for data gen (seconds)")
    plt.legend(loc='upper right')

    filename = "time_data_gen_initialgridsize_%d_tol_%g_chunk_%d_omega_%g_norm" % (initial_grid_size, tol, chunk_size, omega)

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_axis, data_gen_pod, data_gen_modred, data_gen_modred_in_mem])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%d','%f','%f','%f'])

    plt.figure()

    x_end = min(len(svals_hapod_tcw[-1]), len(svals_rapod_traj[-1]), len(svals_pod[-1]), len(eigvals_modred[-1]))
    x_counts = range(1,x_end+1)
    plt.semilogy(x_counts, svals_hapod_tcw[-1][0:x_end], label="tcw")
    plt.semilogy(x_counts, svals_rapod_traj[-1][0:x_end], label="traj")
    plt.semilogy(x_counts, svals_pod[-1][0:x_end], label="pod")
    svals_modred = [math.sqrt(val) for val in eigvals_modred[-1]]
    plt.semilogy(x_counts, svals_modred[0:x_end], label="modred")

    plt.xlabel("")
    plt.ylabel("Singular Value")
    plt.legend(loc='upper right')

    filename = "svals_gridsize_%d_chunk_%d_omega_%g_tol_%g_norm" % (x_axis[-1], chunk_size, omega, tol)

    plt.savefig(filename + ".png")
    plt.clf()

    data = np.array([x_counts, svals_pod[-1][0:x_end], svals_rapod_traj[-1][0:x_end], svals_hapod_tcw[-1][0:x_end], svals_modred[0:x_end]])
    data = data.T

    with open(filename + ".dat", 'w') as f:
        np.savetxt(f, data, fmt=['%d','%f','%f','%f','%f'])

    logfile.close()
