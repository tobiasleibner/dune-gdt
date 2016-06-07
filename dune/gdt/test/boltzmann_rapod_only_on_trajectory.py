import numpy as np
import resource
from timeit import default_timer as timer
import sys
from pymor.basic import pod
from Hapod import HapodBasics
from boltzmann_RAPOD_timechunk_wise_stephans_pod import calculate_error


def rapod_only_on_trajectory(grid_size, chunk_size, tol, log=True, scatter_modes=True, omega=0.5, calculate_max_local_modes=False):
    start = timer()
    b = HapodBasics(grid_size, chunk_size, epsilon_ast=tol, omega=omega)
    b.rooted_tree_depth = b.num_chunks + 2
    max_local_modes = max_vecs_before_pod = 0

    filename = "trajectory_rapod"
    if log and b.rank_world == 0:
        log_file = b.get_log_file(filename)

    total_timings = b.zero_timings_dict()

    # calculate Boltzmann problem trajectory
    modes, svals, total_num_snapshots, timings = b.rapod_on_trajectory()
    if log and b.rank_world == 0:
        log_file.write("After the rapod, there are " + str(len(modes)) + " of " + str(total_num_snapshots) + " left!\n")
    modes.scal(svals)

    # gather snapshots on rank 0 of node and perform pod
    modes, total_num_snapshots = b.gather_on_rank_0(b.comm_proc, modes, total_num_snapshots, uniform_num_modes=False)
        
    if b.rank_proc == 0:
        max_vecs_before_pod = max(max_vecs_before_pod, len(modes))
        modes, timings = b.pod_and_scal(modes, total_num_snapshots)
        max_local_modes = max(max_local_modes, len(modes))
        if log and b.rank_world == 0:
            log_file.write("After the rapod, there are " + str(len(modes)) + " of " + str(total_num_snapshots) + " left!\n")
        for key in timings:
            total_timings[key] += timings[key]

    # gather snapshots on rank 0 of world
    if b.rank_proc == 0:
        modes, total_num_snapshots = b.gather_on_rank_0(b.comm_rank_0_group, modes, total_num_snapshots, uniform_num_modes=False)
        svals = None
        if b.rank_world == 0:
            max_vecs_before_pod = max(max_vecs_before_pod, len(modes))
            modes, svals, timings = b.pod(modes, total_num_snapshots, root_of_tree=True)
            max_local_modes = max(max_local_modes, len(modes))
            for key in timings:
                total_timings[key] += timings[key]
            if log:
                log_file.write("After the pod, there are " + str(len(modes)) + " of " + str(total_num_snapshots) + " left!\n")
                log_file.write("The maximum amount of memory used on rank 0 was: " +
                           str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
                elapsed = timer() - start
                log_file.write("time elapsed: " + str(elapsed) + "\n")

    if calculate_max_local_modes:
        max_local_modes = b.comm_world.gather(max_local_modes, root=0)
        max_vecs_before_pod = b.comm_world.gather(max_vecs_before_pod, root=0)
        if b.rank_world == 0:
            max_local_modes = max(max_local_modes)
            max_vecs_before_pod = max(max_vecs_before_pod)
            if log:
                log_file.write("The maximal number of local modes was: " + str(max_local_modes) + "\n")
                log_file.write("The maximal number of vecs before pod was: " + str(max_vecs_before_pod) + "\n")

    if scatter_modes:
        modes = b.shared_memory_scatter_modes(modes)

    return modes, svals, total_num_snapshots, b, total_timings, max_vecs_before_pod, max_local_modes


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    omega = float(sys.argv[4])
    final_modes, svals, total_num_snapshots, b, _, _, _ = rapod_only_on_trajectory(grid_size, chunk_size, tol * grid_size, omega=omega, calculate_max_local_modes=True)
    filename = "trajectory_rapod_svals"
    if b.rank_world == 0:
        with open(filename, "w") as f:
            f.write(str(svals))
    err = calculate_error(final_modes, total_num_snapshots, b)
    if b.rank_world == 0:
        print(err)






