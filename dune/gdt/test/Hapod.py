from boltzmann import wrapper
from pymor.basic import *
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace
from mpi4py import MPI
from itertools import izip
from timeit import default_timer as timer
import numpy as np
import resource
import sys
import math
import random
from scipy.linalg import eigh


def create_and_scatter_parameters(comm, min_param=0., max_param=8.):
    # Samples all 3 parameters uniformly with the same width and adds random parameter combinations until
    # comm.Get_size() parameters are created. After that, parameter combinations are scattered to ranks.
    num_samples_per_parameter = int(comm.Get_size()**(1./3.))
    sample_width = (max_param - min_param) / (num_samples_per_parameter - 1) if num_samples_per_parameter > 1 else 1e10
    sigma_s_scattering_range = sigma_s_absorbing_range = sigma_a_absorbing_range = np.arange(min_param,
                                                                                             max_param + 1e-13,
                                                                                             sample_width)
    sigma_a_scattering_range = [0.]
    parameters_list = []
    for sigma_s_scattering in sigma_s_scattering_range:
        for sigma_s_absorbing in sigma_s_absorbing_range:
            for sigma_a_scattering in sigma_a_scattering_range:
                for sigma_a_absorbing in sigma_a_absorbing_range:
                    parameters_list.append([sigma_s_scattering,
                                            sigma_s_absorbing,
                                            sigma_a_scattering,
                                            sigma_a_absorbing])
    while len(parameters_list) < comm.Get_size():
        parameters_list.append([random.uniform(min_param, max_param),
                                random.uniform(min_param, max_param),
                                0.,
                                random.uniform(min_param, max_param)])

    return comm.scatter(parameters_list, root=0)


class HapodBasics:
    def __init__(self, gridsize, chunk_size, epsilon_ast=1e-4, omega=0.5):
        self.gridsize = gridsize
        self.chunk_size = chunk_size
        self.with_half_steps = True
        # Preparation: setup MPI
        # create world communicator
        self.comm_world = MPI.COMM_WORLD
        self.rank_world = self.comm_world.Get_rank()
        self.size_world = self.comm_world.Get_size()

        # gather processor names and assign each processor name a unique positive number
        proc_name = MPI.Get_processor_name()
        self.proc_name = proc_name
        proc_names = self.comm_world.allgather(proc_name)
        proc_names = sorted(set(proc_names))
        proc_numbers = dict()
        for i, proc_key in enumerate(proc_names):
            proc_numbers[proc_key] = i

        # use processor numbers to create a communicator on each processor
        self.comm_proc = MPI.Intracomm.Split(self.comm_world, proc_numbers[proc_name], self.rank_world)
        self.size_proc = self.comm_proc.Get_size()
        self.rank_proc = self.comm_proc.Get_rank()

        # create communicator containing rank 0 processes on each processor
        self.contained_in_rank_0_group = 1 if self.rank_proc == 0 else 0
        self.comm_rank_0_group = MPI.Intracomm.Split(self.comm_world, self.contained_in_rank_0_group, self.rank_world)
        self.size_rank_0_group = self.comm_rank_0_group.Get_size()
        self.rank_rank_0_group = self.comm_rank_0_group.Get_rank()

        # Preparation: Parameter selection
        # Choose parameters and scatter to cores.
        self.parameters = create_and_scatter_parameters(self.comm_world)

        # Setup Solver
        self.t_end = 3.2
        self.solver = self.create_solver(self.parameters)
        self.vector_length = self.solver.get_initial_values().dim
        self.empty_vectorarray = wrapper.DuneStuffListVectorSpace(self.vector_length).zeros(0)
        self.num_time_steps = math.ceil(self.t_end / self.solver.time_step_length()) + 1.
        if self.with_half_steps:
            self.num_time_steps += self.num_time_steps - 1
        self.num_chunks = math.ceil(self.num_time_steps / chunk_size)
        self.last_chunk_size = self.num_time_steps - chunk_size * (self.num_chunks - 1.)
        assert self.num_chunks >= 2
        assert 1 <= self.last_chunk_size <= self.chunk_size
        self.epsilon_ast = epsilon_ast
        self.omega = omega
        self.rooted_tree_depth = None
        self.allocated_wins = []

    def create_solver(self, mu):
        return wrapper.Solver("boltzmann_sigma_s_s_" + str(mu[0]) + "_a_" + str(mu[1]) +
                              "sigma_t_s_" + str(mu[2]) + "_a_" + str(mu[3]),
                              2000000,
                              self.gridsize,
                              False,
                              False,
                              *mu)

    def get_log_file(self, file_name, mode="w"):
        return open(file_name + "_gridsize_%d_chunksize_%d_tol_%g_omega_%g_rank_%d"
                    % (self.gridsize, self.chunk_size, self.epsilon_ast, self.omega, self.rank_world), mode)

    def calculate_trajectory_error(self, finalmodes):
        error = 0
        solver = self.create_solver(self.parameters)
        while not solver.finished():
            next_vectors = solver.next_n_time_steps(1, self.with_half_steps)
            next_vectors_npvecarray = NumpyVectorArray(np.zeros(shape=(len(next_vectors), self.vector_length)), NumpyVectorSpace(self.vector_length))
            for vec, vec2 in izip(next_vectors_npvecarray._array, next_vectors._list):
                vec[:] = vec2.data[:]
            del next_vectors
            error += np.sum((next_vectors_npvecarray -
                             finalmodes.lincomb(next_vectors_npvecarray.dot(finalmodes))).l2_norm()**2)
        return error

    def recv_vectorarray(self, comm, len_vectorarray, source, tag):
        received_array = np.empty(shape=(len_vectorarray, self.vector_length))
        comm.Recv(received_array, source=source, tag=tag)
        return self.convert_to_listvectorarray(received_array)

    def get_epsilon_alpha(self, num_snapshots_in_associated_leafs, root_of_tree=False):
        if not root_of_tree:
            epsilon_alpha = self.epsilon_ast * np.sqrt(1. - self.omega**2) * \
                            np.sqrt(num_snapshots_in_associated_leafs) / np.sqrt(self.rooted_tree_depth - 1)
        else:
            epsilon_alpha = self.epsilon_ast * self.omega * np.sqrt(num_snapshots_in_associated_leafs)
        return epsilon_alpha

    def pod(self, vectorarray, num_snapshots_in_associated_leafs, root_of_tree=False, orthonormalize=True):
        epsilon_alpha = self.get_epsilon_alpha(num_snapshots_in_associated_leafs, root_of_tree)
        return pod(vectorarray, atol=0., rtol=0., l2_err=epsilon_alpha, orthonormalize=orthonormalize, check=False)

    def pod_and_scal(self, vectorarray, num_snapshots_in_associated_leafs, root_of_tree=False, orthonormalize=True):
        modes, svals = self.pod(vectorarray, num_snapshots_in_associated_leafs, root_of_tree=root_of_tree, orthonormalize=orthonormalize)
        vectorarray._list = None
        del vectorarray
        if not root_of_tree:
            modes.scal(svals)
        return modes

    # avoids recalculating the gramian for known modes
    # apart from that, it is just a copy of pymors pod algorithm
    def scal_and_pod_for_hapod(self, modes, svals, next_vectors, num_snapshots_in_associated_leafs, root_of_tree=False, 
                               orthonormalize=True, svals2=None):
        len_modes = len(modes)
        len_next = len(next_vectors)
        if len_next == 0:
            return modes, svals   
        modes.scal(svals)
        if svals2 is not None:
            next_vectors.scal(svals2)

        # calculate gramian without recalculating the gramian for modes
        gramian = np.empty((len_modes + len_next,) * 2)
        gramian[:len_modes, :len_modes] = np.diag(svals)**2
        gramian[len_modes:, len_modes:] = next_vectors.gramian() if svals2 is None else np.diag(svals2)**2
        cross_gramian = modes.dot(next_vectors)
        modes.append(next_vectors)
        next_vectors._list = None
        del next_vectors
        gramian[:len_modes, len_modes:] = cross_gramian
        gramian[len_modes:, :len_modes] = cross_gramian.T
        del cross_gramian
        EVALS, EVECS = eigh(gramian, overwrite_a=True, turbo=True, eigvals=None)
        del gramian

        EVALS = EVALS[::-1]
        EVECS = EVECS.T[::-1, :]  # is this a view? yes it is!

        errs = np.concatenate((np.cumsum(EVALS[::-1])[::-1], [0.]))

        epsilon_alpha = self.get_epsilon_alpha(num_snapshots_in_associated_leafs, root_of_tree)
        below_err = np.where(errs <= epsilon_alpha**2)[0]
        first_below_err = below_err[0]

        svals = np.sqrt(EVALS[:first_below_err])
        EVECS = EVECS[:first_below_err]

        final_modes = modes.lincomb(EVECS / svals[:, np.newaxis])
        modes._list = None
        del modes
        del EVECS

        if orthonormalize:
            final_modes = gram_schmidt(final_modes, copy=False)
        
        return final_modes, svals

    def convert_to_listvectorarray(self, numpy_array):
        listvectorarray = self.empty_vectorarray.zeros(len(numpy_array))
        for v, vv in izip(listvectorarray._list, numpy_array):
            v.data[:] = vv
        return listvectorarray

    def gather_on_rank_0(self, comm, vectorarray, num_snapshots_on_rank, uniform_num_modes=True, 
                         return_displacements=False, svals=None):
        rank = comm.Get_rank()
        if svals is not None:
            assert(len(svals) == len(vectorarray))
        num_snapshots_in_associated_leafs = comm.reduce(num_snapshots_on_rank, op=MPI.SUM, root=0)
        total_num_modes = comm.reduce(len(vectorarray), op=MPI.SUM, root=0)
        # create empty numpy array on rank 0 as a buffer to receive the pod modes from each core
        vectors_gathered = np.empty(shape=(total_num_modes, self.vector_length)) if rank == 0 else None
        svals_gathered = np.empty(shape=(total_num_modes,)) if rank == 0 else None
        # gather the modes (as numpy array, thus the call to data) in vectors_gathered.
        displacements = []
        displacements_svals = []
        if uniform_num_modes:
            comm.Gather(vectorarray.data, vectors_gathered, root=0)
            if svals is not None:
                comm.Gather(svals, svals_gathered, root=0)
        else:
            # Gatherv needed because every process can send a different number of modes
            counts = comm.gather(len(vectorarray)*self.vector_length, root=0)
            if svals is not None:
                counts_svals = comm.gather(len(vectorarray), root=0)
            if rank == 0:
                displacements = [0.]
                for j, count in enumerate(counts[0:len(counts) - 1]):
                    displacements.append(displacements[j] + count)
                comm.Gatherv(vectorarray.data, [vectors_gathered, counts, displacements, MPI.DOUBLE], root=0)
                if svals is not None:
                    displacements_svals = [0.]
                    for j, count in enumerate(counts_svals[0:len(counts_svals) - 1]):
                        displacements_svals.append(displacements_svals[j] + count)
                    comm.Gatherv(svals, [svals_gathered, counts_svals, displacements_svals, MPI.DOUBLE], root=0)
            else:
                comm.Gatherv(vectorarray.data, None, root=0)
                if svals is not None:
                    comm.Gatherv(svals, None, root=0)
        vectorarray._list = None
        if rank == 0:
            vectors_gathered = self.convert_to_listvectorarray(vectors_gathered)
        if return_displacements:
            return vectors_gathered, svals_gathered, num_snapshots_in_associated_leafs, displacements
        else:
            return vectors_gathered, svals_gathered, num_snapshots_in_associated_leafs

    def live_hapod_over_ranks(self, comm, modes=None, singular_values=None, num_snapshots_in_associated_leafs=None, 
                              last_hapod=False, logfile=None, incremental_pod=True):
        rank = comm.Get_rank()
        size = comm.Get_size()
        final_modes = modes if rank == 0 else np.empty(shape=(0, 0))
        svals = singular_values
        total_num_snapshots = num_snapshots_in_associated_leafs
        max_vecs_before_pod = len(final_modes) if final_modes is not None else 0
        max_local_modes = 0 

        if self.size_rank_0_group > 1:
            if rank == 0:
                del modes
                if final_modes is None:
                    final_modes, num_snapshots_in_associated_leafs = modes_creator()
                    max_vecs_before_pod = max(max_vecs_before_pod, len(final_modes))
                total_num_snapshots = num_snapshots_in_associated_leafs
            for current_rank in range(1, comm.Get_size()):
                if rank == current_rank:
                    if modes is None:
                        modes, num_snapshots_in_associated_leafs = modes_creator()
                    else:
                        modes.scal(svals)
                    comm.send(len(modes), dest=0, tag=current_rank+1000)
                    comm.send(num_snapshots_in_associated_leafs, dest=0, tag=current_rank+2000)
                    comm.Send(modes.data, dest=0, tag=current_rank+3000)
                    modes._list = None
                    del modes
                elif rank == 0:
                    len_modes_on_source = comm.recv(source=current_rank, tag=current_rank+1000)
                    total_num_snapshots_on_source = comm.recv(source=current_rank, tag=current_rank+2000)
                    total_num_snapshots += total_num_snapshots_on_source
                    modes_on_source = self.recv_vectorarray(comm, len_modes_on_source, source=current_rank,
                                                            tag=current_rank+3000)
                    max_vecs_before_pod = max(max_vecs_before_pod, len(final_modes) + len_modes_on_source)
                    if svals is None:
                        final_modes.append(modes_on_source)
                        del modes_on_source
                        final_modes, svals = self.pod(final_modes, total_num_snapshots)
                        max_local_modes = max(max_local_modes, len(final_modes))
                        if logfile is not None and self.rank_world == 0:
                            logfile.write("In the live HAPOD over ranks, after rank " + str(current_rank) + " there are " +
                                          str(len(final_modes)) + " of " + str(total_num_snapshots) + " left!\n")
                    else:
                        if incremental_pod:
                            final_modes, svals \
                                = self.scal_and_pod_for_hapod(final_modes, svals, modes_on_source, total_num_snapshots,
                                                              root_of_tree=(current_rank == size - 1 and last_hapod))
                        else:
                            final_modes.scal(svals)
                            final_modes.append(modes_on_source)
                            final_modes, svals = self.pod(final_modes, total_num_snapshots, 
                                                          root_of_tree=(current_rank == size - 1 and last_hapod))
                        max_local_modes = max(max_local_modes, len(final_modes))
                        if logfile is not None and self.rank_world == 0:
                            logfile.write("In the live HAPOD over ranks, after rank " + str(current_rank) + " there are " +
                                          str(len(final_modes)) + " of " + str(total_num_snapshots) + " left!\n")
                        del modes_on_source
        return final_modes, svals, total_num_snapshots, max_vecs_before_pod, max_local_modes

    def hapod_over_node_binary_tree(self, comm, modes_in, svals_in, num_snapshots_in_associated_leafs, 
                                    last_hapod=True, logfile=None, incremental_pod=True):
        rank = comm.Get_rank()
        size = comm.Get_size()
        modes = modes_in
        svals = svals_in
        total_num_snapshots = num_snapshots_in_associated_leafs
        max_vecs_before_pod = len(modes);
        max_local_modes = 0 
        if size > 1:
            node_ranks = range(0, size);
            while len(node_ranks) > 1:
                remaining_ranks = list(node_ranks)
                # nodes with odd index send data to the node with index-1 where the pod is performed
                # this ensures that the modes end up on rank 0 in the end
                for odd_index in range(1, len(node_ranks), 2):
                    sending_rank = node_ranks[odd_index];
                    receiving_rank = node_ranks[odd_index-1];
                    remaining_ranks.remove(sending_rank)
                    if rank == sending_rank:
                        comm.send(len(modes), dest=receiving_rank, tag=sending_rank+1000)
                        comm.send(total_num_snapshots, dest=receiving_rank, tag=sending_rank+2000)
                        if not incremental_pod:
                            modes.scal(svals)
                        else:
                            comm.Send(svals, dest=receiving_rank, tag=sending_rank+4000)
                        comm.Send(modes.data, dest=receiving_rank, tag=sending_rank+3000)
                        modes._list = None
                    elif rank == receiving_rank: 
                        len_modes_on_source = comm.recv(source=sending_rank, tag=sending_rank+1000)
                        total_num_snapshots_on_source = comm.recv(source=sending_rank, tag=sending_rank+2000)
                        total_num_snapshots += total_num_snapshots_on_source
                        modes_on_source = self.recv_vectorarray(comm, len_modes_on_source, source=sending_rank,
                                                                tag=sending_rank+3000)
                        max_vecs_before_pod = max(max_vecs_before_pod, len(modes) + len_modes_on_source)
                        if incremental_pod:
                            svals_on_source = np.empty(shape=(len_modes_on_source,))
                            comm.Recv(svals_on_source, source=sending_rank, tag=sending_rank+4000) 
                            modes, svals = self.scal_and_pod_for_hapod(modes, svals, modes_on_source, total_num_snapshots,
                                                                       svals2=svals_on_source,
                                                                       root_of_tree=((len(node_ranks) == 2) and last_hapod))
                        else:
                            modes.scal(svals)
                            modes.append(modes_on_source)
                            modes, svals = self.pod(modes, total_num_snapshots, 
                                                    root_of_tree=((len(node_ranks) == 2) and last_hapod))
                        del modes_on_source
                        max_local_modes = max(max_local_modes, len(modes))
                        if logfile is not None and self.rank_world == 0:
                            logfile.write("In the binary tree HAPOD over nodes, after rank " + str(sending_rank) + " there are " +
                                          str(len(modes)) + " of " + str(total_num_snapshots) + " left!\n")
                node_ranks = list(remaining_ranks)
                if logfile is not None and self.rank_world == 0:
                    logfile.write("After the tree POD there are " + str(remaining_ranks) + "remaining!\n")
        if modes._list is None:
            modes = np.empty(shape=(0, 0), dtype='d')
        return modes, svals, total_num_snapshots, max_vecs_before_pod, max_local_modes


    def shared_memory_bcast_modes(self, final_modes):
        if final_modes is None:
            final_modes = np.empty(shape=(0, 0), dtype='d')
        final_modes_length = self.comm_world.bcast(len(final_modes), root=0)
        # create shared memory buffer to share final modes between processes on each node
        size = final_modes_length * self.vector_length
        itemsize = MPI.DOUBLE.Get_size()
        num_bytes = size * itemsize if self.rank_proc == 0 else 0
        win = MPI.Win.Allocate_shared(num_bytes, itemsize, comm=self.comm_proc)
        self.allocated_wins.append(win)
        buf, itemsize = win.Shared_query(rank=0)
        assert itemsize == MPI.DOUBLE.Get_size()
        buf = np.array(buf, dtype='B', copy=False)
        final_modes_numpy = np.ndarray(buffer=buf, dtype='d', shape=(final_modes_length, self.vector_length))
        if self.rank_proc == 0:
            if self.rank_world == 0:
                self.comm_rank_0_group.Bcast([final_modes.data, MPI.DOUBLE], root=0)
                for i, v in enumerate(final_modes._list):
                    final_modes_numpy[i, :] = v.data[:]
                    del v
            else:
                self.comm_rank_0_group.Bcast([final_modes_numpy, MPI.DOUBLE], root=0)
        final_modes = NumpyVectorArray(final_modes_numpy, NumpyVectorSpace(self.vector_length))
        self.comm_world.Barrier()
        return final_modes

    def calculate_total_projection_error(self, final_modes, total_num_snapshots):
        trajectory_error = self.calculate_trajectory_error(final_modes)
        trajectory_errors = self.comm_world.gather(trajectory_error, root=0)
        error = None
        if self.rank_world == 0:
            error = np.sqrt(np.sum(trajectory_errors) / total_num_snapshots)
        return error

    def live_hapod_on_trajectory(self):
        modes = self.solver.next_n_time_steps(self.chunk_size, self.with_half_steps)
        total_num_snapshots = len(modes)
        chunks_done = 1
        svals = None
        while not self.solver.finished():
            next_vectors = self.solver.next_n_time_steps(self.chunk_size, self.with_half_steps)
            total_num_snapshots += len(next_vectors)
            chunks_done += 1
            if svals is None:
                modes.append(next_vectors)
                del next_vectors
                modes, svals = self.pod(modes, total_num_snapshots)
            else:
                modes, svals = self.scal_and_pod_for_hapod(modes, svals, next_vectors, total_num_snapshots)
                del next_vectors
        assert chunks_done == int(self.num_chunks)
        return modes, svals, total_num_snapshots


def calculate_error(filename, final_modes, total_num_snapshots, b, grid_size=0):
    # Test: solve problem again to calculate error
    # broadcast final modes to rank 0 on each processor and calculate trajectory error on rank 0
    if b.rank_world == 0:
        log_file = b.get_log_file(filename)
    start = timer()
    err = b.calculate_total_projection_error(final_modes, total_num_snapshots)
    if b.rank_world == 0:
        err = err if grid_size == 0 else err/grid_size
        elapsed = timer() - start
        log_file.write("Time used for calculating error: " + str(elapsed) + "\n")
        log_file.write("l2_error is: " + str(err) + "\n")
        log_file.write("The maximum amount of memory used calculating the error on rank " + str(b.rank_world) +
                       " was: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
        log_file.close()
    
    return err
