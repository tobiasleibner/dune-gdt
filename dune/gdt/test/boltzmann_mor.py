from __future__ import print_function

import sys
import numpy as np
from matplotlib.pyplot import *

from pymor.basic import *
from boltzmann.wrapper import DuneDiscretization

set_log_levels({'boltzmann': 'INFO'})
d = DuneDiscretization()

mu = d.parse_parameter([0., 5., 1., 10.])
# mu = d.parameter_space.sample_randomly(1)[0]

# basis generation (POD on single trajectory)
U, U_half = d.solve(mu, return_half_steps=True)
snapshots = U.copy()
snapshots.append(U_half)
V, svals = pod(U, modes=40)
V2, svals2 = pod(snapshots, modes=40)

rd, rc, _ = reduce_generic_rb(d.as_generic_type(), V)
rd2, rc2, _ = reduce_generic_rb(d.as_generic_type(), V2)

errs = []
for dim in range(len(V)):
    print('.', end=''); sys.stdout.flush()
    rrd, rrc, _ = reduce_to_subbasis(rd, dim, rc)
    U_rb = rrc.reconstruct(rrd.solve(mu))
    errs.append(np.sqrt(np.sum((U - U_rb).l2_norm()**2) / len(U)))

errs2 = []
for dim in range(len(V2)):
    print('.', end=''); sys.stdout.flush()
    rrd, rrc, _ = reduce_to_subbasis(rd2, dim, rc2)
    U_rb = rrc.reconstruct(rrd.solve(mu))
    errs2.append(np.sqrt(np.sum((U - U_rb).l2_norm()**2) / len(U)))

semilogy(errs, label='errs')
semilogy(errs2, label='errs2')
legend()
show()
