import numpy as np
import resource
from timeit import default_timer as timer
import sys
from pymor.basic import pod
from Hapod import HapodBasics
from Hapod import calculate_error


def live_hapod_on_trajectory(grid_size, chunk_size, tol, log=True, bcast_modes=True, omega=0.5,
                             calculate_max_local_modes=False):
    start = timer()
    b = HapodBasics(grid_size, chunk_size, epsilon_ast=tol, omega=omega)
    b.rooted_tree_depth = b.num_chunks + 2
    max_local_modes = max_vecs_before_pod = 0

    filename = "HAPOD_trajectory"
    if log and b.rank_world == 0:
        log_file = b.get_log_file(filename)


    # calculate Boltzmann problem trajectory
    modes, svals, total_num_snapshots = b.live_hapod_on_trajectory()
    if log and b.rank_world == 0:
        log_file.write("After the live HAPOD on the trajectory, there are " + str(len(modes)) + " modes of " + str(total_num_snapshots) + " snapshots left on rank 0!\n")
    modes.scal(svals)

    # gather snapshots on rank 0 of node and perform pod
    modes, total_num_snapshots = b.gather_on_rank_0(b.comm_proc, modes, total_num_snapshots, uniform_num_modes=False)
        
    if b.rank_proc == 0:
        max_vecs_before_pod = max(max_vecs_before_pod, len(modes))
        modes = b.pod_and_scal(modes, total_num_snapshots)
        max_local_modes = max(max_local_modes, len(modes))
        if log and b.rank_world == 0:
            log_file.write("After the POD on node level, there are " + str(len(modes)) + " of " + str(total_num_snapshots) +
                           " left!\n")

    # gather snapshots on rank 0 of world
    if b.rank_proc == 0:
        modes, total_num_snapshots = b.gather_on_rank_0(b.comm_rank_0_group, modes, total_num_snapshots,
                                                        uniform_num_modes=False)
        svals = None
        if b.rank_world == 0:
            max_vecs_before_pod = max(max_vecs_before_pod, len(modes))
            modes, svals = b.pod(modes, total_num_snapshots, root_of_tree=True)
            max_local_modes = max(max_local_modes, len(modes))
            if log:
                log_file.write("After the POD on rank 0, there are " + str(len(modes)) + " of " + str(total_num_snapshots) +
                               " left!\n")
                log_file.write("The HAPOD resulted in %d final modes taken from a total of %d snapshots!\n" % (len(modes), 
                                                                                              total_num_snapshots))
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
                log_file.write("The maximal number of input vectors to a local POD was: " + str(max_vecs_before_pod) + "\n")

    if bcast_modes:
        modes = b.shared_memory_bcast_modes(modes)

    return modes, svals, total_num_snapshots, b, max_vecs_before_pod, max_local_modes


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    omega = float(sys.argv[4])
    final_modes, svals, total_num_snapshots, b, _, _ = live_hapod_on_trajectory(grid_size, chunk_size,
                                                                                tol * grid_size, omega=omega,
                                                                                calculate_max_local_modes=True)
    filename = "HAPOD_trajectory"
    filename_errors = "HAPOD_trajectory_error"  
    calculate_error(filename_errors, final_modes, total_num_snapshots, b, grid_size)
    b.comm_world.Barrier()
    if b.rank_world == 0:
        log_file = b.get_log_file(filename, "r")
        log_file_errors = b.get_log_file(filename_errors, "r")
        print("\n\n\nResults:\n")
        print(log_file.read())
        print(log_file_errors.read())
        log_file.close()
        log_file_errors.close()




