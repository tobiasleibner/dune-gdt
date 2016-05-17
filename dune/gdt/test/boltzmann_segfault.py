from boltzmann.wrapper import DuneDiscretization
d = DuneDiscretization()
U = d.solution_space.zeros()
mu = [0., 0., 0., 0.]
d.operators['rhs'].apply(U, mu=mu)
