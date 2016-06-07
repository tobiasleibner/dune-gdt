from future.builtins import range
import numpy as np
import modred as mr
import resource
from timeit import default_timer as timer
import sys
from Hapod import HapodBasics
from boltzmann_RAPOD_timechunk_wise_stephans_pod import calculate_error
from itertools import izip


def modred_pod(grid_size, chunk_size, num_modes, log=True, scatter_modes=True, omega=0.5, smallest_ev=1e-13):
    b = HapodBasics(grid_size, chunk_size, omega=omega)

    # get snapshots

    # Create the snapshots and store in handles
    snapshots = []
    count = 0
    start = timer()
    while not b.solver.finished():
        next_snapshots = b.solver.next_n_time_steps(b.chunk_size, b.with_half_steps)
        snapshots.extend([mr.VecHandleArrayText('/scratch/tmp/l_tobi01/modred_snaps/gridsize%d_chunk%d_nummodes%d_omega%g_rank%d_num%d.txt' % (grid_size, chunk_size, num_modes, omega, b.rank_world, i)) for i in range(count, count + len(next_snapshots))])
        for snap, handle in izip(next_snapshots._list, snapshots[count:(count + len(next_snapshots))]):
            handle.put(snap.data)
        count += len(next_snapshots)
    snapshots_gathered = b.comm_world.allgather(snapshots)
    snapshots = []
    for snapshotlist in snapshots_gathered:
        snapshots.extend(snapshotlist)
    del snapshots_gathered
    del snapshotlist
    total_num_snapshots = len(snapshots)
    b.comm_world.Barrier()
    data_generation_elapsed = timer() - start

    # Perform POD
    start = timer()
    vecs_per_node = int(40000000000. / (grid_size**2 * 136 * 8))
    my_POD = mr.PODHandles(np.vdot, max_vecs_per_node=vecs_per_node)
    eigvals, eigvecs = my_POD.compute_decomp(snapshots, atol=smallest_ev)

    mode_nums = list(range(num_modes))
    modes = [mr.VecHandleArrayText('/scratch/tmp/l_tobi01/modred_modes/gridsize%d_chunk%d_nummodes%d_omega%g_num%d' % (grid_size, chunk_size, num_modes, omega, i)) for i in mode_nums]
    my_POD.compute_modes(mode_nums, modes)
    b.comm_world.Barrier()
    pod_elapsed = timer() - start

    #filename = "modred_pod"
    #if log and b.rank_world == 0:
    #    log_file = b.get_log_file(filename)

    modes_lva = None
    if b.rank_world == 0:
        modes_lva = b.empty_vectorarray.zeros(num_modes)
        for v, handle in izip(modes_lva._list, modes):
            v.data[:] = handle.get()[:,0]

    if scatter_modes:
        modes_lva = b.shared_memory_scatter_modes(modes_lva)

    svals = [math.sqrt(val) for val in eigvals]

    return modes_lva, svals[0:num_modes], total_num_snapshots, b, b.zero_timings_dict(), data_generation_elapsed, pod_elapsed


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    num_modes = float(sys.argv[3])
    omega = float(sys.argv[4])
    smallest_ev = float(sys.argv[5])
    final_modes, _, total_num_snapshots, b, _, _, _ = modred_pod(grid_size, chunk_size, num_modes, omega=omega, smallest_ev=smallest_ev)
    calculate_error(final_modes, total_num_snapshots, b)










