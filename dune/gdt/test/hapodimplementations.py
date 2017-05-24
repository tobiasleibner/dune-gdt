from pymor.algorithms.pod import pod as pymor_pod
import numpy as np
from hapod import pod
from mpiwrapper import MPIWrapper
        
def live_hapod_on_trajectory(solver, chunk_size, parameters, with_half_steps=True):
    ''' A live HAPOD with chunks of solution vectors of a Boltzmann moment problem solution trajectory
        :param solver: A Solver object from boltzmann/wrapper.py'''
    modes = solver.next_n_time_steps(chunk_size, with_half_steps)
    total_num_snapshots = len(modes)
    svals = None
    while not solver.finished():
        next_vectors = solver.next_n_time_steps(chunk_size, with_half_steps)
        total_num_snapshots += len(next_vectors)
        modes, svals = hapod([[modes, svals], next_vectors], total_num_snapshots, parameters)
    return modes, svals, total_num_snapshots

def live_hapod_over_ranks(comm, modes, num_snaps_in_leafs, parameters, svals=None, last_hapod=False, 
                          logfile=None, incremental_pod=True):
    ''' A live HAPOD with modes and possibly svals stored on ranks of the MPI communicator comm.
        May be used as part of a larger HAPOD tree, in that case you need to specify whether this
        part of the tree contains the root node (last_hapod=True)'''
    rank = comm.Get_rank()
    total_num_snapshots = num_snaps_in_leafs
    max_vecs_before_pod = len(modes)
    max_local_modes = 0 

    if comm.Get_size() > 1:
        for current_rank in range(1, comm.Get_size()):
            # send modes and svals to rank 0
            if rank == current_rank:
                comm.send([len(modes), len(svals) if svals is not None else 0, num_snaps_in_leafs, modes[0].dim], 
                          dest=0, tag=current_rank+1000)
                comm.Send(modes.data, dest=0, tag=current_rank+2000)
                if svals is not None:
                    comm.Send(svals, dest=0, tag=current_rank+3000)
                modes._list = None
                del modes
            # receive modes and svals
            elif rank == 0:
                len_modes_on_source, len_svals_on_source, total_num_snapshots_on_source, vector_length \
                                                                = comm.recv(source=current_rank, tag=current_rank+1000)
                max_vecs_before_pod = max(max_vecs_before_pod, len(modes) + len_modes_on_source)
                total_num_snapshots += total_num_snapshots_on_source
                modes_on_source = MPIWrapper.recv_vectorarray(comm, len_modes_on_source, vector_length, 
                                                              source=current_rank, tag=current_rank+2000)
                svals_on_source = np.empty(shape=(len_modes_on_source,))
                if len_svals_on_source > 0:
                    comm.Recv(svals_on_source, source=current_rank, tag=current_rank+3000) 
                if incremental_pod:
                    modes, svals = pod([[modes, svals], 
                                        [modes_on_source, svals_on_source] if len_svals_on_source > 0 else modes_on_source], 
                                       total_num_snapshots,
                                       parameters,
                                       root_of_tree=(current_rank == size - 1 and last_hapod))
                else:
                    if svals is not None:
                        modes.scal(svals)
                    if len_svals_on_source > 0:
                        modes_on_source.scal(svals_on_source)
                    modes.append(modes_on_source)
                    modes, svals = pod([modes], total_num_snapshots, parameters, root_of_tree=(current_rank == size - 1 and last_hapod))
                max_local_modes = max(max_local_modes, len(modes))
                del modes_on_source
    return modes, svals, total_num_snapshots, max_vecs_before_pod, max_local_modes

def binary_tree_depth(comm):
    """Calculates depth of binary tree of MPI ranks"""
    binary_tree_depth = 1;
    ranks = range(0, comm.Get_size())
    while len(ranks) > 1:
        binary_tree_depth += 1
        remaining_ranks = list(ranks)
        for odd_index in range(1, len(ranks), 2):
            remaining_ranks.remove(ranks[odd_index])
        ranks = remaining_ranks
    return binary_tree_depth

def binary_tree_hapod_over_ranks(comm, modes, num_snaps_in_leafs, parameters, svals=None,
                                 last_hapod=True, incremental_pod=True):
    ''' A HAPOD with modes and possibly svals stored on ranks of the MPI communicator comm. A binary tree
        of MPI ranks is used as HAPOD tree.
        May be used as part of a larger HAPOD tree, in that case you need to specify whether this
        part of the tree contains the root node (last_hapod=True) '''
    rank = comm.Get_rank()
    size = comm.Get_size()
    total_num_snapshots = num_snaps_in_leafs
    max_vecs_before_pod = len(modes)
    max_local_modes = 0 
    if size > 1:
        ranks = range(0, size);
        while len(ranks) > 1:
            remaining_ranks = list(ranks)
            # nodes with odd index send data to the node with index-1 where the pod is performed
            # this ensures that the modes end up on rank 0 in the end
            for odd_index in range(1, len(ranks), 2):
                sending_rank = ranks[odd_index]
                receiving_rank = ranks[odd_index-1]
                remaining_ranks.remove(sending_rank)
                if rank == sending_rank:
                    comm.send([len(modes), len(svals) if svals is not None else 0, total_num_snapshots, modes[0].dim], 
                              dest=receiving_rank, tag=sending_rank+1000)
                    comm.Send(modes.data, dest=receiving_rank, tag=sending_rank+2000)
                    if svals is not None:
                        comm.Send(svals, dest=receiving_rank, tag=sending_rank+3000)
                    modes._list = None
                elif rank == receiving_rank: 
                    len_modes_on_source, len_svals_on_source, total_num_snapshots_on_source, vector_length \
                                                                = comm.recv(source=sending_rank, tag=sending_rank+1000)
                    max_vecs_before_pod = max(max_vecs_before_pod, len(modes) + len_modes_on_source)
                    total_num_snapshots += total_num_snapshots_on_source
                    modes_on_source = MPIWrapper.recv_vectorarray(comm, len_modes_on_source, vector_length, 
                                                                  source=sending_rank, tag=sending_rank+2000)
                    svals_on_source = np.empty(shape=(len_modes_on_source,))
                    if len_svals_on_source > 0:
                        comm.Recv(svals_on_source, source=sending_rank, tag=sending_rank+3000) 
                    if incremental_pod:
                        modes, svals = pod([[modes, svals],  
                                             [modes_on_source, svals_on_source] if len_svals_on_source > 0 else modes_on_source], 
                                           total_num_snapshots, 
                                           parameters,
                                           root_of_tree=((len(ranks) == 2) and last_hapod))
                    else:
                        if svals is not None:
                            modes.scal(svals)
                        if len_svals_on_source > 0:
                            modes_on_source.scal(svals_on_source)
                        modes.append(modes_on_source)
                        modes, svals = pod([modes], total_num_snapshots, parameters, 
                                           root_of_tree=((len(ranks) == 2) and last_hapod))
                    del modes_on_source
                    max_local_modes = max(max_local_modes, len(modes))
            ranks = list(remaining_ranks)
    return modes, svals, total_num_snapshots, max_vecs_before_pod, max_local_modes
