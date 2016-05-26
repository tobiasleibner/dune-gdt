import numpy as np
import resource
from timeit import default_timer as timer
import sys
from pymor.basic import pod
from Hapod import HapodBasics
from boltzmann_RAPOD_timechunk_wise_stephans_pod import calculate_error


def boltzmann_standard_pod(grid_size, tol, log=True, scatter_modes=True):
    start = timer()
    b = HapodBasics(grid_size, 1, epsilon_ast=tol)

    filename = "standard_pod"
    if log and b.rank_world == 0:
        log_file = b.get_log_file(filename)

    # calculate Boltzmann problem trajectory
    result = b.solver.solve(b.with_half_steps)
#    b.solver = None
    num_snapshots=len(result)

    # gather snapshots on rank 0
    result, total_num_snapshots = b.gather_on_rank_0(b.comm_world, result, num_snapshots)
    singularvalues = None
    timings = b.zero_timings_dict()
    if b.rank_world == 0:
        result, singularvalues, timings = pod(result, atol=0., rtol=0., l2_mean_err=b.epsilon_ast)
        if log:
            log_file.write("After the pod, there are " + str(len(result)) + " of " + str(total_num_snapshots) + " left!\n")
            log_file.write("The maximum amount of memory used on rank 0 was: " +
                           str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
            elapsed = timer() - start
            log_file.write("time elapsed: " + str(elapsed) + "\n")
    return result, singularvalues, total_num_snapshots, b, timings


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    tol = float(sys.argv[3])
    final_modes, _, total_num_snapshots, b, _ = boltzmann_standard_pod(grid_size, tol * grid_size)
    filename = "standard_pod_error"
    calculate_error(filename, final_modes, total_num_snapshots, b)






