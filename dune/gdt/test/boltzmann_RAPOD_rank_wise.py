import numpy as np
import resource
from timeit import default_timer as timer
import sys
from Hapod import HapodBasics


def rapod_rank_wise():
    start = timer()

    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    b = HapodBasics(grid_size, chunk_size, 1)
    b.rooted_tree_depth = b.num_chunks + (b.size_proc - 1) + (b.size_rank_0_group - 1)

    filename = "RAPOD_rank_wise"
    if b.rank_world == 0 or b.rank_world == 1:
        log_file = b.get_log_file(filename)

    # RAPOD 1 and 2: Perform a RAPOD on each node (processor)
    modes, total_num_snapshots = b.rapod_over_ranks(b.comm_proc, modes_creator=b.rapod_on_trajectory)
    b.solver = None

    if b.rank_world == 1:
        log_file.write("The maximum amount of memory used on rank " + str(b.rank_world) + " was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")

    if b.rank_proc == 0:
        final_modes, total_num_snapshots = b.rapod_over_ranks(b.comm_rank_0_group, modes, total_num_snapshots,
                                                              last_rapod=True)
    else:
        final_modes, total_num_snapshots = (np.empty(shape=(0,0)), None)

    # write statistics to file
    if b.rank_world == 0:
        log_file.write("There was a total of " + str(total_num_snapshots) + " snapshots!\n")
        log_file.write("The maximum amount of memory used on rank 0 was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        elapsed = timer() - start
        log_file.write("time elapsed: " + str(elapsed) + "\n")

    # Test: solve problem again to calculate error
    # broadcast final modes to rank 0 on each processor and calculate trajectory error on rank 0
    start = timer()
    err = b.calculate_total_projection_error(final_modes, total_num_snapshots)
    if b.rank_world == 0:
        elapsed = timer() - start
        log_file.write("time used for calculating error: " + str(elapsed) + "\n")
        log_file.write("l2_mean_error is: " + str(err))
    if b.rank_world == 0 or b.rank_world == 1:
        log_file.write("The maximum amount of memory used calculating the error on rank " + str(b.rank_world) +
                       " was: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        log_file.close()


rapod_rank_wise()




