import numpy as np
import resource
from timeit import default_timer as timer
import sys
from pymor.basic import pod
from Hapod import HapodBasics


def boltzmann_standard_pod():
    start = timer()
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    b = HapodBasics(grid_size, chunk_size, 1)

    filename = "standard_pod"
    if b.rank_world == 0:
        log_file = b.get_log_file(filename)

    # calculate Boltzmann problem trajectory
    result = b.solver.solve()
    b.solver = None
    num_snapshots=len(result)

    # gather snapshots on rank 0
    result, total_num_snapshots = b.gather_on_rank_0(b.comm_world, result, num_snapshots)

    if b.rank_world == 0:
        result, singularvalues = pod(result, atol=0., rtol=0., l2_mean_err=b.epsilon_ast)
        log_file.write("After the pod, there are " + str(len(result)) + " of " + str(total_num_snapshots) + " left!\n")
        log_file.write("The maximum amount of memory used on rank 0 was: " +
                       str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        elapsed = timer() - start
        log_file.write("time elapsed: " + str(elapsed) + "\n")

    # calculate error
    error = b.calculate_total_projection_error(result, total_num_snapshots)
    if b.rank_world == 0:
        log_file.write("l2_mean_error is: " + str(error))
        log_file.close()



boltzmann_standard_pod()



