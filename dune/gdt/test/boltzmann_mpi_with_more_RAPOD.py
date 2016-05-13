from boltzmann import wrapper
from pymor.basic import *
from pymor.vectorarrays.list import NumpyVector
from mpi4py import MPI
import numpy as np
import resource
from timeit import default_timer as timer
import sys

######### create MPI communicators
# create world communicator
comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

if (rank_world == 0):
    start = timer()

# gather processor names and assign each processor name a unique positive number
proc_name = MPI.Get_processor_name()
proc_names = comm_world.allgather(proc_name)
proc_numbers = dict.fromkeys(set(proc_names), 0)
for i, proc_key in enumerate(proc_numbers):
    proc_numbers[proc_key] = i

# use processor numbers to create a communicator on each processor
comm_proc = MPI.Intracomm.Split(comm_world, proc_numbers[proc_name], rank_world)
size_proc = comm_proc.Get_size()
rank_proc = comm_proc.Get_rank()

# create communicator containing rank 0 processes on each processor
contained_in_rank_0_group = 1 if rank_proc == 0 else 0
comm_rank_0_group = MPI.Intracomm.Split(comm_world, contained_in_rank_0_group, rank_world)
size_rank_0_group = comm_rank_0_group.Get_size()
rank_rank_0_group = comm_rank_0_group.Get_rank()

######### snapshot parameters
sigma_s_scattering_range = range(0, 9, 2)
sigma_s_absorbing_range = range(0, 9, 2)
sigma_a_scattering_range = range(0, 11, 11)
sigma_a_absorbing_range = range(0, 9, 2)

parameters_list=[]
for sigma_s_scattering in sigma_s_scattering_range:
    for sigma_s_absorbing in sigma_s_absorbing_range:
        for sigma_a_scattering in sigma_a_scattering_range:
            for sigma_a_absorbing in sigma_a_absorbing_range:
                parameters_list.append([sigma_s_scattering, sigma_s_absorbing, sigma_s_scattering+sigma_a_scattering, sigma_s_absorbing+sigma_a_absorbing])

parameters=comm_world.scatter(parameters_list, root=0)

######### perform RAPOD with Boltzmann trajectory (using one thread per process)
gridsize=int(sys.argv[1])
chunk_size=int(sys.argv[2])
t_end = 3.2
solver=wrapper.Solver(1, "boltzmann_sigma_s_s_" + str(parameters[0]) + "_a_" + str(parameters[1]) + "sigma_t_s_" + str(parameters[2]) + "_a_" + str(parameters[3]), 2000000, gridsize, False, False, parameters[0], parameters[1], parameters[2], parameters[3])
num_chunks = int(t_end/(10.*solver.time_step_length())) + (not np.isclose(t_end/(10.*solver.time_step_length()), int(t_end/(10.*solver.time_step_length()))))
assert num_chunks >= 2
rooted_tree_depth=num_chunks+size_rank_0_group
epsilon_ast = 1e-4*gridsize
omega=0.5

modes = solver.next_n_time_steps(chunk_size)
vector_length=modes.dim
num_snapshots=len(modes)
chunks_done=1

if rank_world == 0:
    log_file = open("more_RAPOD_gridsize" + str(gridsize) + "chunksize" + str(chunk_size), "w")

while not solver.finished():
    print(str(solver.finished()), str(solver.current_time()))
    next_vectors = solver.next_n_time_steps(chunk_size)
    num_snapshots += len(next_vectors)
    chunks_done += 1
    modes.append(next_vectors)
    modes, singular_values = pod(modes, atol=0., rtol=0., l2_mean_err=epsilon_ast*omega*np.sqrt(num_snapshots)/np.sqrt(len(modes)*(rooted_tree_depth-1)))
    if rank_world == 0:
        log_file.write("After step " + str(chunks_done) + ", " + str(len(modes)) + " of " + str(num_snapshots) + " are left!\n")
    modes.scal(singular_values)
assert chunks_done == num_chunks
del next_vectors
num_modes = len(modes)

######### gather scaled pod modes from trajectories on processor, join to one VectorArray and perform a second pod per processor
# calculate total number of snapshots on processor and number of modes that remain after the POD
num_snapshots_on_proc = comm_proc.reduce(num_snapshots, op=MPI.SUM, root=0)
num_modes_on_proc_before_pod = comm_proc.reduce(num_modes, op=MPI.SUM, root=0)

# create empty numpy array on rank 0 as a buffer to receive the pod modes from each core
modes_on_proc = np.empty(shape=(num_modes_on_proc_before_pod,vector_length), dtype=np.float64) if rank_proc == 0 else None

# calculate number of elements in each modes Vectorarray and the resulting needed displacements in modes_on_proc
counts=comm_proc.gather(num_modes*vector_length, root=0)
if rank_proc == 0:
    displacements=[0.]
    for i, count in enumerate(counts[0:len(counts)-1]):
        displacements.append(displacements[i]+count)
# creaty empty vectorarray to be able to create vectorarrays later on
empty_vectorarray=modes.zeros(0)
# gather the modes (as numpy array, thus the call to data) in modes_on_proc. Gatherv needed because every process
# can send a different number of modes
if rank_proc == 0:
    comm_proc.Gatherv(modes.data, [modes_on_proc, counts, displacements, MPI.DOUBLE], root=0)
else:
    comm_proc.Gatherv(modes.data, None, root=0)
    del modes

# create a pyMOR VectorArray from modes_on_proc and perform the second pod
num_modes_on_proc_after_pod = 0
if rank_proc == 0:
    modes_on_proc_joined = modes.zeros(len(modes_on_proc))
    del modes
    for v, vv in zip(modes_on_proc_joined._list, modes_on_proc):
        v.data[:] = vv
    del modes_on_proc
    epsilon_T_alpha = epsilon_ast*omega*np.sqrt(num_snapshots_on_proc)/np.sqrt(num_modes_on_proc_before_pod*(rooted_tree_depth-1))
    second_modes, second_singular_values = pod(modes_on_proc_joined, atol=0., rtol=0., l2_mean_err=epsilon_T_alpha)
    del modes_on_proc_joined
    second_modes.scal(second_singular_values)
    num_modes_on_proc_after_pod = len(second_modes)
if rank_world == 0:
   log_file.write("On processor level, there are " + str(num_modes_on_proc_after_pod) + " of " + str(num_modes_on_proc_before_pod) + " left!\n")


######### RAPOD for scaled pod modes from second pod
final_modes = np.empty(shape=(0,0))
if rank_world == 0:
    total_num_snapshots = num_snapshots_on_proc
    final_modes = second_modes

if rank_proc == 0:
    for rank in range(1, size_rank_0_group):
        if (rank_rank_0_group == rank):
            comm_rank_0_group.send(len(second_modes), dest=0, tag=rank+1000)
            comm_rank_0_group.send(num_snapshots_on_proc, dest=0, tag=rank+2000)
            comm_rank_0_group.Send(second_modes.data, dest=0, tag=rank+3000)
            del second_modes
        elif rank_world == 0:
            len_curr_second_modes = comm_rank_0_group.recv(source=rank, tag=rank+1000)
            num_snapshots_on_curr_proc = comm_rank_0_group.recv(source=rank, tag=rank+2000)
            total_num_snapshots += num_snapshots_on_curr_proc
            curr_second_modes_numpy = np.empty(shape=(len_curr_second_modes, vector_length))
            comm_rank_0_group.Recv(curr_second_modes, source=rank, tag=rank+3000)
            curr_second_modes = empty_vectorarray.zeros(len_curr_second_modes)
            for v, vv in zip(curr_second_modes._list, curr_second_modes_numpy):
                v.data[:] = vv
            del curr_second_modes_numpy
            final_modes.append(curr_second_modes)
            del curr_second_modes
            curr_epsilon=epsilon_ast*(1-omega)*np.sqrt(total_num_snapshots)/np.sqrt(len(final_modes)) if rank == size_rank_0_group-1 else epsilon_ast*omega*np.sqrt(total_num_snapshots)/np.sqrt(len(final_modes)*(rooted_tree_depth-1))
            final_modes, final_singular_values = pod(final_modes, atol=0., rtol=0., l2_mean_err=curr_epsilon)
            if (rank != size_rank_0_group - 1):
                final_modes.scal(final_singular_values)
            log_file.write("On rank 0, in step " + rank + " there are " + str(len(final_modes)) + " of " + str(total_num_snapshots) + " left!\n")

if rank_world == 0:
    log_file.write("There was a total of " + str(total_num_snapshots) + " snapshots!\n")
    log_file.write("The maximum amount of memory used on rank 0 was: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.**2) + " GB\n")
    elapsed = timer() - start
    log_file.write("time elapsed: " + str(elapsed) + "\n")


#### solve problem again to calculate error
final_modes_length = comm_world.bcast(len(final_modes), root=0)
if rank_world == 0:
    comm_world.Bcast(final_modes.data, root=0)
else:
    final_modes_numpy = np.empty(shape=(final_modes_length,vector_length))
    comm_world.Bcast(final_modes_numpy, root=0)
    final_modes = empty_vectorarray.zeros(final_modes_length)
    for v, vv in zip(final_modes._list, final_modes_numpy):
        v.data[:] = vv
    del final_modes_numpy
    del empty_vectorarray

solver.reset()
chunks_done=0

trajectory_error = 0
while not solver.finished():
    print(str(solver.finished()), str(solver.current_time()))
    next_vectors = solver.next_n_time_steps(chunk_size)
    trajectory_error += np.sum((next_vectors - final_modes.lincomb(next_vectors.dot(final_modes))).l2_norm()**2)
    chunks_done += 1
assert chunks_done == num_chunks

del final_modes
del next_vectors
trajectory_errors = comm_world.gather(trajectory_error, root=0)
if rank_world == 0:
    error=np.sqrt(np.sum(trajectory_errors)/total_num_snapshots)
    log_file.write("l2_mean_error is: " + str(error))
    log_file.close()


