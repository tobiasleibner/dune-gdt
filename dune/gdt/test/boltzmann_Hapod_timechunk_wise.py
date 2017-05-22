import numpy as np
import resource
from timeit import default_timer as timer
import sys
from Hapod import HapodBasics
from Hapod import calculate_error
import pickle


def hapod_timechunk_wise(grid_size, chunk_size, tol, log=True, bcast_modes=True, omega=0.5,
                         calculate_max_local_modes=False, incremental_pod=True, nodes_binary_tree=True):
    start = timer()

    max_vectors_before_pod = 0
    max_local_modes = 0

    b = HapodBasics(grid_size, chunk_size, epsilon_ast=tol, omega=omega)
    if not nodes_binary_tree:
        b.rooted_tree_depth = b.num_chunks + b.size_rank_0_group
    else:
        # calculate depth of node binary tree
        binary_tree_depth = 1;
        node_ranks = range(0, b.comm_rank_0_group.Get_size())
        while len(node_ranks) > 1:
            binary_tree_depth += 1
            remaining_ranks = list(node_ranks)
            for odd_index in range(1, len(node_ranks), 2):
                remaining_ranks.remove(node_ranks[odd_index])
            node_ranks = remaining_ranks
        binary_tree_depth = b.comm_proc.bcast(binary_tree_depth, root=0)
        b.rooted_tree_depth = b.num_chunks + binary_tree_depth

    filename = "HAPOD_timechunk_wise"
    log_file = None
    if log and b.rank_world == 0:
        log_file = b.get_log_file(filename)

    modes = b.empty_vectorarray.zeros(0)
    svals = []
    total_num_snapshots = 0
    for i in range(0, int(b.num_chunks)):
        timestep_vectors = b.solver.next_n_time_steps(b.chunk_size, b.with_half_steps)
        num_snapshots = len(timestep_vectors)
        if incremental_pod and i > 0:
            timestep_vectors, timestep_svals = b.pod(timestep_vectors, num_snapshots)
        else:
            timestep_vectors, timestep_svals = [b.pod_and_scal(timestep_vectors, num_snapshots), None]
        gathered_vectors, gathered_svals, num_snapshots_in_this_chunk = b.gather_on_rank_0(b.comm_proc,
                                                                                           timestep_vectors,
                                                                                           num_snapshots,
                                                                                           uniform_num_modes=False,
                                                                                           svals=timestep_svals)
        if b.rank_proc == 0:
            total_num_snapshots += num_snapshots_in_this_chunk
            if incremental_pod and i > 0:
                max_vectors_before_pod = max(max_vectors_before_pod, len(modes) + len(gathered_vectors))
                assert(gathered_svals is not None)
                modes, svals = b.scal_and_pod_for_hapod(modes, svals, gathered_vectors, total_num_snapshots, svals2=gathered_svals)
            else:
                if len(modes) > 0:
                    modes.scal(svals)
                print("     ssssssssssssssssssssssssss " + str(gathered_svals))
                assert(gathered_svals is None or gathered_svals == [None]*b.size_proc)
                modes.append(gathered_vectors)
                modes, svals = b.pod(modes, total_num_snapshots)
            max_local_modes = max(max_local_modes, len(modes))
            del gathered_vectors
            if log and b.rank_world == 0:
                log_file.write("After the POD with timechunks " + str(i) + " there are " + str(len(modes)) +
                               " modes of " + str(total_num_snapshots) + " snapshots left!\n")

    start2 = timer();
    print("Starting")
    if b.rank_proc == 0:
        final_modes, svals, total_num_snapshots, max_vectors_before_pod_in_hapod, max_local_modes_in_hapod \
            = b.hapod_over_node_binary_tree(b.comm_rank_0_group,
                                            modes,
                                            svals,
                                            total_num_snapshots,
                                            last_hapod=True,
                                            logfile=log_file,
                                            incremental_pod=incremental_pod) if nodes_binary_tree else \
              b.live_hapod_over_ranks(b.comm_rank_0_group,
                                      modes,
                                      svals,
                                      total_num_snapshots,
                                      last_hapod=True,
                                      logfile=log_file,
                                      incremental_pod=incremental_pod)
        max_vectors_before_pod = max(max_vectors_before_pod, max_vectors_before_pod_in_hapod)
        max_local_modes = max(max_local_modes, max_local_modes_in_hapod)
        del modes
    else:
        final_modes, svals, total_num_snapshots = (np.empty(shape=(0, 0)), None, None)
    if log and b.rank_world == 0:
        log_file.write("time for final live Hapod:" + str(timer()-start2) +"\n")
        log_file.write("time for all:" + str(timer()-start) +"\n")

    if calculate_max_local_modes:
        max_vectors_before_pod = b.comm_world.gather(max_vectors_before_pod, root=0)
        max_local_modes = b.comm_world.gather(max_local_modes, root=0)
        if b.rank_world == 0:
            max_vectors_before_pod = max(max_vectors_before_pod)
            max_local_modes = max(max_local_modes)

    # write statistics to file
    if log and b.rank_world == 0:
        log_file.write("The HAPOD resulted in %d final modes taken from a total of %d snapshots!\n" % (len(final_modes), 
                                                                                              total_num_snapshots))
        if calculate_max_local_modes:
            log_file.write("The maximal number of local modes was: " + str(max_local_modes) + "\n")
            log_file.write("The maximal number of input vectors to a local POD was: " + str(max_vectors_before_pod) + "\n")
        log_file.write("The maximum amount of memory used on rank 0 was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        elapsed = timer() - start
        log_file.write("Time elapsed: " + str(elapsed) + "\n")
        log_file.close()

    if bcast_modes:
        final_modes = b.shared_memory_bcast_modes(final_modes)

    return final_modes, svals, total_num_snapshots, b, max_vectors_before_pod, max_local_modes


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    omega = float(sys.argv[4])
    nodes_binary_tree = not (sys.argv[5] == "False" or sys.argv[5] == "0") if len(sys.argv) > 5 else True
    incremental_pod = not (sys.argv[6] == "False" or sys.argv[6] == "0") if len(sys.argv) > 6 else True
    final_modes, _, total_num_snapshots, b, _, _ = hapod_timechunk_wise(grid_size, chunk_size, tol * grid_size,
                                                                        omega=omega, calculate_max_local_modes=True, 
                                                                        incremental_pod=incremental_pod,
                                                                        nodes_binary_tree=nodes_binary_tree)
    filename = "HAPOD_timechunk_wise"
    filename_errors = "HAPOD_timechunk_wise_error"
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


