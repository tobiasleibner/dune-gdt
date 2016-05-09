from boltzmann import wrapper
from mpi4py import MPI

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

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

solver=wrapper.Solver(1, "/scratch/tmp/l_tobi01/boltzmann_sigma_s_s_" + str(parameters[0]) + "_a_" + str(parameters[1]) + "sigma_t_s_" + str(parameters[2]) + "_a_" + str(parameters[3]), 1, 100, True, True, parameters[0], parameters[1], parameters[2], parameters[3])
solver.solve()
print(parameters, " done ")

