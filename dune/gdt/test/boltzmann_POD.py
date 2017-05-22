import numpy as np
import resource
from timeit import default_timer as timer
import sys
from pymor.basic import pod
from Hapod import HapodBasics
from Hapod import calculate_error


def boltzmann_standard_pod(grid_size, tol, log=True, bcast_modes=True):
    b = HapodBasics(grid_size, 10, epsilon_ast=tol)

    filename = "standard_pod"
    if log and b.rank_world == 0:
        log_file = b.get_log_file(filename)

    # calculate Boltzmann problem trajectory
    start = timer()
    result = b.solver.solve(b.with_half_steps)
    b.comm_world.Barrier()
    elapsed_data_gen = timer() - start
    num_snapshots=len(result)

    # gather snapshots on rank 0
    start = timer()
    result, total_num_snapshots = b.gather_on_rank_0(b.comm_world, result, num_snapshots)
    singularvalues = None
    elapsed_pod = 0
    if b.rank_world == 0:
        result, singularvalues = pod(result, atol=0., rtol=0., l2_err=b.epsilon_ast*np.sqrt(total_num_snapshots))
        elapsed_pod = timer() - start
        if log:
            log_file.write("After the POD, there are " + str(len(result)) + " modes of " + str(total_num_snapshots) + " snapshots left!\n")
            log_file.write("The maximum amount of memory used on rank 0 was: " +
                           str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
            elapsed = timer() - start
            log_file.write("Time elapsed: " + str(elapsed) + "\n")

    if bcast_modes:
        result = b.shared_memory_bcast_modes(result)

    return result, singularvalues, total_num_snapshots, b, elapsed_data_gen, elapsed_pod


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    tol = float(sys.argv[2])
    final_modes, _, total_num_snapshots, b, _, _ = boltzmann_standard_pod(grid_size, tol * grid_size)
    filename = "standard_pod"
    filename_errors = "standard_pod_error"  
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






