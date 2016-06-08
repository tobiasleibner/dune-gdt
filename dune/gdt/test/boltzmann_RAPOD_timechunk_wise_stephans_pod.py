import numpy as np
import resource
from timeit import default_timer as timer
import sys
from Hapod import HapodBasics


def rapod_timechunk_wise(grid_size, chunk_size, tol, log=True, scatter_modes=True, omega=0.5, calculate_max_local_modes=False):
    start = timer()

    max_vectors_before_pod = 0
    max_local_modes = 0

    b = HapodBasics(grid_size, chunk_size, epsilon_ast=tol, omega=omega)
    b.rooted_tree_depth = b.num_chunks + b.size_rank_0_group

    filename = "RAPOD_timechunk_wise_stephans_pod"
    if log and (b.rank_world == 0 or b.rank_world == 1):
        log_file = b.get_log_file(filename)

    modes = None
    total_num_snapshots = 0
    pod_timings = b.zero_timings_dict()
    for i in range(0, int(b.num_chunks)):
        timestep_vectors = b.solver.next_n_time_steps(b.chunk_size, b.with_half_steps)
        num_snapshots = len(timestep_vectors)
        timestep_vectors, timings = b.pod_and_scal(timestep_vectors, num_snapshots)
        gathered_vectors, num_snapshots_in_this_chunk = b.gather_on_rank_0(b.comm_proc,
                                                                           timestep_vectors,
                                                                           num_snapshots,
                                                                           uniform_num_modes=False)
        del timestep_vectors
        if b.rank_proc == 0:
            total_num_snapshots += num_snapshots_in_this_chunk
            if i == 0:
                modes, svals, timings2 = b.pod(gathered_vectors, num_snapshots_in_this_chunk)
                max_local_modes = max(max_local_modes, len(modes))
	    else:
                max_vectors_before_pod = max(max_vectors_before_pod, len(modes) + len(gathered_vectors))
                modes, svals, timings2 = b.scal_and_pod_for_rapod(modes, svals, gathered_vectors, total_num_snapshots)
                max_local_modes = max(max_local_modes, len(modes))
            del gathered_vectors
            if log and b.rank_world == 0:
                log_file.write("In the third pod, in step " + str(i) + " there are " + str(len(modes)) +
                               " of " + str(total_num_snapshots) + " left!\n")
            for key in pod_timings:
                pod_timings[key] += timings[key] + timings2[key]
#    if b.rank_proc == 0:
#        modes.scal(svals)

#    if b.rank_world != 0:
#        b.solver = None

    if log and b.rank_world == 1:
        log_file.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")

    if b.rank_proc == 0:
        final_modes, svals, total_num_snapshots, timings, max_vectors_before_pod_in_rapod, max_local_modes_in_rapod = b.rapod_over_ranks(b.comm_rank_0_group,
                                                                                                                                         modes,
                                                                                                                                         svals,
                                                                                                                                         total_num_snapshots,
                                                                                                                                         last_rapod=True)
        max_vectors_before_pod = max(max_vectors_before_pod, max_vectors_before_pod_in_rapod)
        max_local_modes = max(max_local_modes, max_local_modes_in_rapod)
        for key in pod_timings:
            pod_timings[key] += timings[key]
        del modes
    else:
        final_modes, svals, total_num_snapshots = (np.empty(shape=(0,0)), None, None)

    if calculate_max_local_modes:
        max_vectors_before_pod = b.comm_world.gather(max_vectors_before_pod, root=0)
        max_local_modes = b.comm_world.gather(max_local_modes, root=0)
        if b.rank_world == 0:
            max_vectors_before_pod = max(max_vectors_before_pod)
            max_local_modes = max(max_local_modes)

    # write statistics to file
    if log and b.rank_world == 0:
        log_file.write("There were " + str(len(final_modes)) + " basis vectors taken from a total of " + str(total_num_snapshots) + " snapshots!\n")
        if calculate_max_local_modes:
            log_file.write("The maximal number of local modes was: " + str(max_vectors_before_pod) + "\n")
        log_file.write("The maximum amount of memory used on rank 0 was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        elapsed = timer() - start
        log_file.write("time elapsed: " + str(elapsed) + "\n")
        log_file.close()

    if scatter_modes:
        final_modes = b.shared_memory_scatter_modes(final_modes)

    return final_modes, svals, total_num_snapshots, b, pod_timings, max_vectors_before_pod, max_local_modes

def calculate_error(filename, final_modes, total_num_snapshots, b):
    # Test: solve problem again to calculate error
    # broadcast final modes to rank 0 on each processor and calculate trajectory error on rank 0
    if b.rank_world == 0 or b.rank_world == 1:
        log_file = b.get_log_file(filename)
    start = timer()
    err = b.calculate_total_projection_error(final_modes, total_num_snapshots)
    if b.rank_world == 0:
        elapsed = timer() - start
        log_file.write("time used for calculating error: " + str(elapsed) + "\n")
        log_file.write("l2_mean_error is: " + str(err) + "\n")
    if b.rank_world == 0 or b.rank_world == 1:
        log_file.write("The maximum amount of memory used calculating the error on rank " + str(b.rank_world) +
                       " was: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        log_file.close()
    return err

if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    omega = float(sys.argv[4])
    final_modes, _, total_num_snapshots, b, _, _, _ = rapod_timechunk_wise(grid_size, chunk_size, tol * grid_size, omega=omega, calculate_max_local_modes=True)
    filename = "RAPOD_timechunk_wise_stephans_pod_error"
    calculate_error(filename, final_modes, total_num_snapshots, b)


