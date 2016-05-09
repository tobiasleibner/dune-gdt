from boltzmann import wrapper
from mpi4py import MPI
import numpy as np

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

sigma_s_scattering_range = np.arange(0, 1, 0.3)
sigma_s_absorbing_range = np.arange(0, 1, 0.3)
sigma_t_scattering_range = range(0, 11, 3)
sigma_t_absorbing_range = range(0, 11, 3)

parameters_list=[]
for sigma_s_scattering in sigma_s_scattering_range:
    for sigma_s_absorbing in sigma_s_absorbing_range:
        for sigma_t_scattering in sigma_t_scattering_range:
            for sigma_t_absorbing in sigma_s_absorbing_range:
                parameters_list.append([sigma_s_scattering, sigma_s_absorbing, sigma_t_scattering, sigma_t_absorbing])

parameters=comm_world.scatter(parameters_list, root=0)

solver=wrapper.Solver(1, "boltzmann_sigma_s_s_" + str(parameters[1]) + "_a_" + str(parameters[2]) + "sigma_t_s_" + str(parameters[3]) + "_a_" + str(parameters[4]), 2, 50, True, True, parameters[1], parameters[2], parameters[3], parameters[4])
solver.solve()
print(parameters, " done ")

