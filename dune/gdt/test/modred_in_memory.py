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
    vecs_per_node = int(40000000000. / (grid_size**2 * 136 * 8))
    my_POD = mr.PODHandles(np.vdot, max_vecs_per_node=vecs_per_node)
    eigvals, eigvecs = my_POD.compute_decomp(snapshots, atol=smallest_ev)

    mode_nums = list(range(num_modes))
    modes = [mr.VecHandleInMemory() for _ in mode_nums]
    my_POD.compute_modes(mode_nums, modes)
    b.comm_world.Barrier()
    pod_elapsed = timer() - start

    non_none_modes_enumerated = filter(lambda x: x[1].get() is not None, enumerate(modes))
    non_none_mode_numbers = [x[0] for x in non_none_modes_enumerated]
    non_none_modes = [x[1].get() for x in non_none_modes_enumerated]
    if non_none_modes == []:
        non_none_modes = np.empty(shape=(0, 0))
    modes_gathered, _, displacements = b.gather_on_rank_0(b.comm_world, b.convert_to_listvectorarray(non_none_modes), len(non_none_modes), return_displacements=True, uniform_num_modes=False)
    mode_numbers = b.comm_world.gather(non_none_mode_numbers, root=0)
    vec_displacements = [val / b.vector_length for val in displacements]

    # sort modes
    modes = None
    if b.rank_world == 0:
        modes = b.empty_vectorarray.zeros(num_modes)
        current_vec_index = 0
        displacement_index = next(x[0] for x in enumerate(vec_displacements) if x[1] > 0) - 1
        rank_vec_index = 0
        print(vec_displacements)
        for vec in modes_gathered._list:
            print(current_vec_index, displacement_index, rank_vec_index)
            modes._list[mode_numbers[displacement_index][rank_vec_index]].data[:] = vec.data[:]     
            current_vec_index += 1
            rank_vec_index += 1
            try:
                next_displacement_index_value = vec_displacements[next(x[0] for x in enumerate(vec_displacements) if x[1] > vec_displacements[displacement_index])]
                next_displacement_index = next(x[0] for x in enumerate(vec_displacements) if x[1] > next_displacement_index_value) - 1 if next_displacement_index_value != vec_displacements[-1] else 0
                print(next_displacement_index)
                if current_vec_index >= vec_displacements[next_displacement_index]:
                    displacement_index = next_displacement_index 
                    rank_vec_index = 0
            except StopIteration:
                pass
    b.comm_world.Barrier()       
        
    if scatter_modes:
        modes = b.shared_memory_scatter_modes(modes)

    return modes, eigvals[0:num_modes], total_num_snapshots, b, b.zero_timings_dict(), data_gen_elapsed, pod_elapsed


if __name__ == "__main__":
    grid_size = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    num_modes = int(sys.argv[3])
    smallest_ev = float(sys.argv[4])
    final_modes, _, total_num_snapshots, b, _, _, _ = modred_pod_in_memory(grid_size, chunk_size, num_modes, smallest_ev=smallest_ev)
    error = calculate_error("modred_in_memory_error", final_modes, total_num_snapshots, b)
    b.comm_world.Barrier()
    if b.rank_world == 0:
        print(error)










