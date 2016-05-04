from boltzmann import wrapper
from pymor.basic import *
from mpi4py import MPI

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

# gather processor names and assign each processor name a unique positive number
processor_name = MPI.Get_processor_name()
processor_names = comm_world.gather(processor_name, root=0)
processor_numbers = dict()
if rank_world == 0:
    processor_set = set(processor_names)
    processor_numbers = dict.fromkeys(processor_set, 0)
    current_num = 0
    for processor_key in processor_numbers:
        processor_numbers[processor_key] = current_num
        current_num += 1
processor_numbers = comm_world.bcast(processor_numbers, root=0)

# use processor numbers to create a communicator on each processor
comm_proc = MPI.Intracomm.Split(comm_world, processor_numbers[processor_name], rank_world)
size_proc = comm_proc.Get_size()
rank_proc = comm_proc.Get_rank()

# create communicator containing rank 0 processes on each processor
contained_in_rank_0_group = 0
if rank_proc == 0:
    contained_in_rank_0_group = 1
comm_rank_0_group = MPI.Intracomm.Split(comm_world, contained_in_rank_0_group, rank_world)

# calculate Boltzmann problem trajectory (using one thread per process)
solver = wrapper.Solver(1, "boltzmann", -1, 200, false, true)
result = solver.solve()

# get pod modes from each trajectory
pod_modes, ignored_singular_values = pod(result)

# gather pod modes from trajectories on processor, join to one VectorArray and perform a second pod per processor
pod_modes_on_processor = comm_proc.gather(pod_modes, root=0)
pod_modes_on_processor_joined = pod_modes.empty()
second_pod_modes = 0
if rank_proc == 0:
    for vectorarray in pod_modes_on_processor:
        pod_modes_on_processor_joined.append(vectorarray)
    second_pod_modes, ignored_singular_values = pod(pod_modes_on_processor_joined)

# gather all pod modes from second pod on world rank 0 and perform a third pod with the joined pod modes
all_second_pod_modes = comm_rank_0_group.gather(second_pod_modes, root=0)
all_second_pod_modes_joined = pod_modes.empty()
final_pod_modes = 0
final_singular_values = 0
if rank_world == 0:
    for vectorarray in all_second_pod_modes:
        all_second_pod_modes_joined.append(vectorarray)
    final_pod_modes, final_singular_values = pod(all_second_pod_modes_joined)
    print(final_pod_modes, final_singular_values)
