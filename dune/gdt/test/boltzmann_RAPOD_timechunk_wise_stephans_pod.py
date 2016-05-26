import numpy as np
import resource
from timeit import default_timer as timer
import sys
from Hapod import HapodBasics


def rapod_timechunk_wise(grid_size, chunk_size, tol, log=True, scatter_modes=True):
    start = timer()

    b = HapodBasics(grid_size, chunk_size, 1, tol)
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
#            gathered_vectors = b.pod_and_scal(gathered_vectors, num_snapshots_in_this_chunk)
            if i == 0:
                modes, svals, timings2 = b.pod(gathered_vectors, num_snapshots_in_this_chunk)
	    else:
                modes, svals, timings2 = b.scal_and_pod_for_rapod(modes, svals, gathered_vectors, total_num_snapshots)
#            modes.append(gathered_vectors)
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
        final_modes, svals, total_num_snapshots, timings = b.rapod_over_ranks(b.comm_rank_0_group, modes, svals, total_num_snapshots,
                                                                     last_rapod=True)
        for key in pod_timings:
            pod_timings[key] += timings[key]
        del modes
    else:
        final_modes, svals, total_num_snapshots = (np.empty(shape=(0,0)), None, None)

    # write statistics to file
    if log and b.rank_world == 0:
        log_file.write("There were " + str(len(final_modes)) + " basis vectors taken from a total of " + str(total_num_snapshots) + " snapshots!\n")
        log_file.write("The maximum amount of memory used on rank 0 was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        elapsed = timer() - start
        log_file.write("time elapsed: " + str(elapsed) + "\n")
        log_file.close()

    if scatter_modes:
        final_modes = b.shared_memory_scatter_modes(final_modes)

    return final_modes, svals, total_num_snapshots, b, pod_timings

def calculate_error(final_modes, total_num_snapshots, b):
    # Test: solve problem again to calculate error
    # broadcast final modes to rank 0 on each processor and calculate trajectory error on rank 0
    filename = "RAPOD_timechunk_wise_stephans_pod_error"
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

if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    final_modes, _, total_num_snapshots, b = rapod_timechunk_wise(grid_size, chunk_size, tol * grid_size)
    calculate_error(final_modes, total_num_snapshots, b)


