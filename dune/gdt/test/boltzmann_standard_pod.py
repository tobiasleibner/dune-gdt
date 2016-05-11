from boltzmann import wrapper
from pymor.basic import *
from pymor.vectorarrays.list import NumpyVector
from mpi4py import MPI
import numpy as np
import resource
import pickle

######### create MPI communicators
# create world communicator
comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()


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

######### gather scaled pod modes from trajectories on processor, join to one VectorArray and perform a second pod per processor
# calculate total number of snapshots on processor and number of modes that remain after the POD
total_num_snapshots = comm_world.reduce(num_snapshots, op=MPI.SUM, root=0)

# create empty numpy array on rank 0 as a buffer to receive the pod modes from each core
all_snapshots = np.empty(shape=(total_num_snapshots,vector_length), dtype=np.float64) if rank_world == 0 else None

# calculate number of elements in each modes Vectorarray and the resulting needed displacements in modes_on_proc
comm_world.Gather(result.data, all_snapshots, root=0)

# create a pyMOR VectorArray from modes_on_proc and perform the second pod
if rank_world == 0:
    all_snapshots_vectorarray = result.zeros(total_num_snapshots)
    for v, vv in zip(all_snapshots_vectorarray._list, all_snapshots):
        v.data[:] = vv
    arrayfile = open("vectorarray_pickled_laxfriedrichs", "w")
    pickle.dump(all_snapshots_vectorarray.data, arrayfile)
    arrayfile.close()
    epsilon=1e-4
    modes, singularvalues = pod(all_snapshots_vectorarray, atol=0., rtol=0., l2_mean_err=epsilon)
    print(singularvalues)
    f = open("singular_values", "w")
    f.write(str(singularvalues))
    f.close()
    print("memory used 1: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0)


