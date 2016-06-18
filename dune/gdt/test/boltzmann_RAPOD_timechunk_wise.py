import numpy as np
import resource
from timeit import default_timer as timer
import sys
from Hapod import HapodBasics
from Hapod import calculate_error


def rapod_timechunk_wise(grid_size, chunk_size, tol, log=True, scatter_modes=True, omega=0.5,
                         calculate_max_local_modes=False):
    start = timer()

    max_vectors_before_pod = 0
    max_local_modes = 0

    b = HapodBasics(grid_size, chunk_size, epsilon_ast=tol, omega=omega)
    b.rooted_tree_depth = b.num_chunks + b.size_rank_0_group

    filename = "RAPOD_timechunk_wise"
    log_file = None
    if log and b.rank_world == 0:
        log_file = b.get_log_file(filename)

    modes = None
    total_num_snapshots = 0
    for i in range(0, int(b.num_chunks)):
        timestep_vectors = b.solver.next_n_time_steps(b.chunk_size, b.with_half_steps)
        num_snapshots = len(timestep_vectors)
        timestep_vectors = b.pod_and_scal(timestep_vectors, num_snapshots)
        gathered_vectors, num_snapshots_in_this_chunk = b.gather_on_rank_0(b.comm_proc,
                                                                           timestep_vectors,
                                                                           num_snapshots,
                                                                           uniform_num_modes=False)
        del timestep_vectors
        if b.rank_proc == 0:
            total_num_snapshots += num_snapshots_in_this_chunk
            if i == 0:
                modes, svals = b.pod(gathered_vectors, num_snapshots_in_this_chunk)
                max_local_modes = max(max_local_modes, len(modes))
            else:
                max_vectors_before_pod = max(max_vectors_before_pod, len(modes) + len(gathered_vectors))
                modes, svals = b.scal_and_pod_for_rapod(modes, svals, gathered_vectors, total_num_snapshots)
                max_local_modes = max(max_local_modes, len(modes))
            del gathered_vectors
            if log and b.rank_world == 0:
                log_file.write("In the first pod, in step " + str(i) + " there are " + str(len(modes)) +
                               " of " + str(total_num_snapshots) + " left!\n")

    if b.rank_proc == 0:
        final_modes, svals, total_num_snapshots, max_vectors_before_pod_in_rapod, max_local_modes_in_rapod \
            = b.rapod_over_ranks(b.comm_rank_0_group,
                                 modes,
                                 svals,
                                 total_num_snapshots,
                                 last_rapod=True,
                                 logfile=log_file)
        max_vectors_before_pod = max(max_vectors_before_pod, max_vectors_before_pod_in_rapod)
        max_local_modes = max(max_local_modes, max_local_modes_in_rapod)
        del modes
    else:
        final_modes, svals, total_num_snapshots = (np.empty(shape=(0, 0)), None, None)

    if calculate_max_local_modes:
        max_vectors_before_pod = b.comm_world.gather(max_vectors_before_pod, root=0)
        max_local_modes = b.comm_world.gather(max_local_modes, root=0)
        if b.rank_world == 0:
            max_vectors_before_pod = max(max_vectors_before_pod)
            max_local_modes = max(max_local_modes)

    # write statistics to file
    if log and b.rank_world == 0:
        log_file.write("There were %d basis vectors taken from a total of %d snapshots!\n" % (len(final_modes), 
                                                                                              total_num_snapshots))
        if calculate_max_local_modes:
            log_file.write("The maximal number of local modes was: " + str(max_local_modes) + "\n")
            log_file.write("The maximal number of vectors before pod was: " + str(max_vectors_before_pod) + "\n")
        log_file.write("The maximum amount of memory used on rank 0 was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        elapsed = timer() - start
        log_file.write("time elapsed: " + str(elapsed) + "\n")
        log_file.close()

    if scatter_modes:
        final_modes = b.shared_memory_bcast_modes(final_modes)

    return final_modes, svals, total_num_snapshots, b, max_vectors_before_pod, max_local_modes


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    omega = float(sys.argv[4])
    final_modes, _, total_num_snapshots, b, _, _ = rapod_timechunk_wise(grid_size, chunk_size, tol * grid_size,
                                                                        omega=omega, calculate_max_local_modes=True)
    filename = "RAPOD_timechunk_wise_error"
    calculate_error(filename, final_modes, total_num_snapshots, b)


