from boltzmann import wrapper
from pymor.basic import *
from mpi4py import MPI
import numpy as np

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

# calculate Boltzmann problem trajectory (using one thread per process)
solver = wrapper.Solver(1, "boltzmann", 2000000, 50, False, False)
result = solver.solve()
num_snapshots=len(result)

# get pod modes from each trajectory
epsilon_ast = 4e-4
omega=0.5
rooted_tree_depth=3
modes, singular_values = pod(result, atol=0., rtol=0., l2_mean_err=epsilon_ast*omega/np.sqrt(rooted_tree_depth-1))
modes.scal(singular_values)

# gather scaled pod modes from trajectories on processor, join to one VectorArray and perform a second pod per processor
modes_on_proc = comm_proc.gather(modes, root=0)
num_snapshots_on_proc = comm_proc.reduce(num_snapshots, op=MPI.SUM, root=0)
if rank_proc == 0:
    modes_on_proc_joined = modes.empty()
    for vectorarray in modes_on_proc:
        modes_on_proc_joined.append(vectorarray)
    epsilon_T_alpha = epsilon_ast*omega*np.sqrt(num_snapshots_on_proc/(len(modes_on_proc_joined)*(rooted_tree_depth-1)))
    print("epsilon_t_alpha: ", epsilon_T_alpha)
    second_modes, second_singular_values = pod(modes_on_proc_joined, atol=0., rtol=0., l2_mean_err=epsilon_T_alpha)
    second_modes.scal(second_singular_values)

# gather all scaled pod modes from second pod on world rank 0 and perform a third pod with the joined pod modes
if rank_proc == 0:
    all_second_modes = comm_rank_0_group.gather(second_modes, root=0)
    total_num_snapshots = comm_rank_0_group.reduce(num_snapshots_on_proc, op=MPI.SUM, root=0)

final_modes=0
if rank_world == 0:
    all_second_modes_joined = modes.empty()
    for vectorarray in all_second_modes:
        all_second_modes_joined.append(vectorarray)
    epsilon_T_gamma = epsilon_ast*(1-omega)*np.sqrt(total_num_snapshots/len(all_second_modes_joined))
    print("epsilon_t_gamma: ", epsilon_T_gamma)
    final_modes, final_singular_values = pod(all_second_modes_joined, atol=0., rtol=0., l2_mean_err=epsilon_T_gamma)
    print(final_singular_values)
    #final_modes.scal(final_singular_values)
final_modes=comm_world.bcast(final_modes, root=0)

trajectory_error = np.sum((result - final_modes.lincomb(result.dot(final_modes))).l2_norm()**2)
trajectory_errors = comm_world.gather(trajectory_error, root=0)
if rank_world == 0:
    error=np.sqrt(np.sum(trajectory_errors)/total_num_snapshots)
    print(error)
