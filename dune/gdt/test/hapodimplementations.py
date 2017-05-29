from abc import abstractmethod

from hapod import local_pod


def live_hapod_on_trajectory(snapshot_provider, parameters):
    ''' A live HAPOD with chunks of solution vectors of a Boltzmann moment problem solution trajectory
        :param solver: A Solver object from boltzmann/wrapper.py'''
    modes = snapshot_provider()
    total_num_snapshots = len(modes)
    svals = None
    for next_vectors in snapshot_provider():
        total_num_snapshots += len(next_vectors)
        modes, svals = local_pod([[modes, svals], next_vectors], total_num_snapshots, parameters)
    return modes, svals, total_num_snapshots


class MPICommunicator(object):

    rank = None
    size = None

    @abstractmethod
    def send_modes(self, dest, modes, svals, num_snaps_in_leafs):
        pass

    @abstractmethod
    def recv_modes(self, source):
        pass


def live_hapod_over_ranks(comm, modes, num_snaps_in_leafs, parameters, svals=None, last_hapod=False,
                          logfile=None, incremental_pod=True):
    ''' A live HAPOD with modes and possibly svals stored on ranks of the MPI communicator comm.
        May be used as part of a larger HAPOD tree, in that case you need to specify whether this
        part of the tree contains the root node (last_hapod=True)'''
    total_num_snapshots = num_snaps_in_leafs
    max_vecs_before_pod = len(modes)
    max_local_modes = 0

    if comm.size > 1:
        for current_rank in range(1, comm.size):
            # send modes and svals to rank 0
            if comm.rank == current_rank:
                comm.send_modes(0, modes, svals, num_snaps_in_leafs)
                modes = None
            # receive modes and svals
            elif comm.rank == 0:
                modes_on_source, svals_on_source, total_num_snapshots_on_source = \
                    comm.recv_modes(current_rank)
                max_vecs_before_pod = max(max_vecs_before_pod, len(modes) + len(modes_on_source))
                total_num_snapshots += total_num_snapshots_on_source
                modes, svals = local_pod(
                    [[modes, svals], [modes_on_source, svals_on_source]
                     if len(svals_on_source) > 0 else modes_on_source],
                    total_num_snapshots,
                    parameters,
                    incremental=incremental_pod,
                    root_of_tree=(current_rank == comm.size - 1 and last_hapod)
                )
                max_local_modes = max(max_local_modes, len(modes))
                del modes_on_source
    return modes, svals, total_num_snapshots, max_vecs_before_pod, max_local_modes


def binary_tree_depth(comm):
    """Calculates depth of binary tree of MPI ranks"""
    binary_tree_depth = 1
    ranks = range(0, comm.size)
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
    total_num_snapshots = num_snaps_in_leafs
    max_vecs_before_pod = len(modes)
    max_local_modes = 0
    if comm.size > 1:
        ranks = range(0, comm.size)
        while len(ranks) > 1:
            remaining_ranks = list(ranks)
            # nodes with odd index send data to the node with index-1 where the pod is performed
            # this ensures that the modes end up on rank 0 in the end
            for odd_index in range(1, len(ranks), 2):
                sending_rank = ranks[odd_index]
                receiving_rank = ranks[odd_index-1]
                remaining_ranks.remove(sending_rank)
                if comm.rank == sending_rank:
                    comm.send_modes(receiving_rank, modes, svals, total_num_snapshots)
                    modes = None
                elif comm.rank == receiving_rank:
                    modes_on_source, svals_on_source, total_num_snapshots_on_source = \
                        comm.recv_modes(sending_rank)
                    max_vecs_before_pod = max(max_vecs_before_pod, len(modes) + len(modes_on_source))
                    total_num_snapshots += total_num_snapshots_on_source
                    modes, svals = local_pod(
                        [[modes, svals], [modes_on_source, svals_on_source]
                         if len(svals_on_source) > 0 else modes_on_source],
                        total_num_snapshots,
                        parameters,
                        incremental=incremental_pod,
                        root_of_tree=((len(ranks) == 2) and last_hapod)
                    )
                    max_local_modes = max(max_local_modes, len(modes))
            ranks = list(remaining_ranks)
    return modes, svals, total_num_snapshots, max_vecs_before_pod, max_local_modes
