from future.builtins import range
import numpy as np
import modred as mr
import resource
from timeit import default_timer as timer
import sys
from Hapod import HapodBasics
from boltzmann_RAPOD_timechunk_wise_stephans_pod import calculate_error
from itertools import izip
import pickle


def modred_pod_in_memory(grid_size, chunk_size, num_modes, log=True, scatter_modes=True, smallest_ev=1e-13):
    b = HapodBasics(grid_size, chunk_size)

    # Create the snapshots and store in handles
    start = timer()
    snapshots = []
    vecs = b.solver.solve(b.with_half_steps)
    vecs, _ = b.gather_on_rank_0(b.comm_world, vecs, len(vecs))
    vecs = b.shared_memory_scatter_modes(vecs)
    
    snapshots = [mr.VecHandleInMemory(vec) for vec in vecs._array]
    total_num_snapshots = len(snapshots)
    b.comm_world.Barrier()
    data_gen_elapsed = timer() - start

    # Perform POD
    start = timer()
    vecs_per_node = int(47000000000. / (grid_size**2 * 136 * 8))
    my_POD = mr.PODHandles(np.vdot, max_vecs_per_node=vecs_per_node)
    eigvals, eigvecs = my_POD.compute_decomp(snapshots, atol=smallest_ev)

    mode_nums = list(range(num_modes))
    modes = [mr.VecHandleInMemory() for _ in mode_nums]
    my_POD.compute_modes(mode_nums, modes)
    b.comm_world.Barrier()
    pod_elapsed = timer() - start

    non_none_modes = filter(lambda x: x[1].get() is not None, enumerate(modes))
    modes_gathered = b.comm_world.gather(non_none_modes, root=0)
    if b.rank_world == 0:
        for mode_tuple_list in modes_gathered:
            for mode_tuple in mode_tuple_list:
                if not mode_tuple == []:
                    print(mode_tuple)
                    modes[mode_tuple[0]] = mode_tuple[1]
    b.comm_world.Barrier()       
        
            
    modes_lva = None
    if b.rank_world == 0:
        modes_lva = b.empty_vectorarray.zeros(num_modes)
        for v, handle in izip(modes_lva._list, modes):
            v.data[:] = handle.get()[:]

    if scatter_modes:
        modes_lva = b.shared_memory_scatter_modes(modes_lva)

    return modes_lva, eigvals[0:num_modes], total_num_snapshots, b, b.zero_timings_dict(), data_gen_elapsed, pod_elapsed


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    num_modes = int(sys.argv[3])
    smallest_ev = float(sys.argv[4])
    final_modes, _, total_num_snapshots, b, _, _, _ = modred_pod_in_memory(grid_size, chunk_size, num_modes, smallest_ev=smallest_ev)
    error = calculate_error(final_modes, total_num_snapshots, b)
    b.comm_world.Barrier()
    if b.rank_world == 0:
        print(error)










