from boltzmann import wrapper
from pymor.basic import *
from pymor.vectorarrays.list import NumpyVector
from mpi4py import MPI
import numpy as np
import resource

######### create MPI communicators
# create world communicator
comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

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

######### snapshot parameters
# snapshot parameters
sigma_s_scattering_range = range(0, 11, 3)
sigma_s_absorbing_range = range(0, 8, 3)
sigma_a_scattering_range = range(0, 11, 3)
sigma_a_absorbing_range = range(0, 11, 3)

parameters_list=[]
for sigma_s_scattering in sigma_s_scattering_range:
    for sigma_s_absorbing in sigma_s_absorbing_range:
        for sigma_a_scattering in sigma_a_scattering_range:
            for sigma_a_absorbing in sigma_a_absorbing_range:
                parameters_list.append([sigma_s_scattering, sigma_s_absorbing, sigma_s_scattering+sigma_a_scattering, sigma_s_absorbing+sigma_a_absorbing])

parameters=comm_world.scatter(parameters_list, root=0)

######### calculate Boltzmann problem trajectory (using one thread per process)
solver=wrapper.Solver(1, "boltzmann_sigma_s_s_" + str(parameters[0]) + "_a_" + str(parameters[1]) + "sigma_t_s_" + str(parameters[2]) + "_a_" + str(parameters[3]), 2000000, 20, False, False, parameters[0], parameters[1], parameters[2], parameters[3])
result = solver.solve()
num_snapshots=len(result)
vector_length=result.dim

######### perform first POD with each trajectory
epsilon_ast = 1e-8
omega=0.5
rooted_tree_depth=3
modes, singular_values = pod(result, atol=0., rtol=0., l2_mean_err=epsilon_ast*omega/np.sqrt(rooted_tree_depth-1))
modes.scal(singular_values)
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
    epsilon_T_alpha = epsilon_ast*omega*np.sqrt(num_snapshots_on_proc/(num_modes_on_proc_before_pod*(rooted_tree_depth-1)))
    second_modes, second_singular_values = pod(modes_on_proc_joined, atol=0., rtol=0., l2_mean_err=epsilon_T_alpha)
    del modes_on_proc_joined
    second_modes.scal(second_singular_values)
    num_modes_on_proc_after_pod = len(second_modes)
    print("memory used 1: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0)

######### gather all (scaled) pod modes from second pod on world rank 0 and perform a third pod with the joined pod modes
total_num_snapshots = comm_rank_0_group.reduce(num_snapshots_on_proc, op=MPI.SUM, root=0) if rank_proc == 0 else None
total_num_modes = comm_rank_0_group.reduce(num_modes_on_proc_after_pod, op=MPI.SUM, root=0) if rank_proc == 0 else None
all_second_modes = np.empty(shape=(total_num_modes,vector_length)) if rank_world == 0 else None

counts=comm_rank_0_group.gather(num_modes_on_proc_after_pod*vector_length, root=0)
if rank_proc == 0:
    if rank_world == 0:
        displacements=[0.]
        for i, count in enumerate(counts[0:len(counts)-1]):
            displacements.append(displacements[i]+count)
        comm_rank_0_group.Gatherv(second_modes.data, [all_second_modes, counts, displacements, MPI.DOUBLE], root=0)
        print("memory used 2: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0)
    else:
        comm_rank_0_group.Gatherv(second_modes.data, None, root=0)
        del second_modes

final_modes=0
if rank_world == 0:
    all_second_modes_joined = second_modes.zeros(total_num_modes)
    del second_modes
    for v, vv in zip(all_second_modes_joined._list, all_second_modes):
        v.data[:] = vv
    del all_second_modes
    epsilon_T_gamma = epsilon_ast*(1-omega)*np.sqrt(total_num_snapshots/total_num_modes)
    final_modes, final_singular_values = pod(all_second_modes_joined, atol=0., rtol=0., l2_mean_err=epsilon_T_gamma)
    del all_second_modes_joined
    print("memory used 3: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0)
    print(final_singular_values)
    #final_modes.scal(final_singular_values)
final_modes=comm_world.bcast(final_modes, root=0)

trajectory_error = np.sum((result - final_modes.lincomb(result.dot(final_modes))).l2_norm()**2)
trajectory_errors = comm_world.gather(trajectory_error, root=0)
if rank_world == 0:
    error=np.sqrt(np.sum(trajectory_errors)/total_num_snapshots)
    print(error)