from itertools import product

import numpy as np

from pymor.discretizations.basic import DiscretizationBase
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import VectorOperator, ConstantOperator, LincombOperator
from pymor.parameters.base import ParameterType
from pymor.parameters.functionals import ExpressionParameterFunctional, ProjectionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.list import VectorInterface, ListVectorArray

import libboltzmann
from libboltzmann import CommonDenseVector
IMPL_TYPES = (CommonDenseVector,)


PARAMETER_TYPE = ParameterType({'s': (4,)})


class Solver(object):

    def __init__(self, *args):
        self.impl = libboltzmann.BoltzmannSolver(*args)
        self.last_mu = None

    def solve(self, with_half_steps=False):
        result = self.impl.solve(with_half_steps)
        return ListVectorArray(map(DuneStuffVector, result))

    def next_n_time_steps(self, n, with_half_steps=False):
        result = self.impl.next_n_time_steps(n, with_half_steps)
        return ListVectorArray(map(DuneStuffVector, result))

    def reset(self):
        self.impl.reset()

    def finished(self):
        return self.impl.finished()

    def current_time(self):
        return self.impl.current_time()

    def set_current_time(self, time):
        return self.impl.set_current_time(time)

    def set_current_solution(self, vec):
        return self.impl.set_current_solution(vec)

    def time_step_length(self):
        return self.impl.time_step_length()

    def get_initial_values(self):
        return DuneStuffVector(self.impl.get_initial_values())

    def apply_LF_operator(self, source, time):
        return DuneStuffVector(self.impl.apply_LF_operator(source.impl, time))

    def apply_godunov_operator(self, source, time):
        return DuneStuffVector(self.impl.apply_godunov_operator(source.impl, time))

    def apply_rhs_operator(self, source, *args):
        return DuneStuffVector(self.impl.apply_rhs_operator(source.impl, *args))

    def set_rhs_operator_params(self, sigma_s_scattering=1, sigma_s_absorbing=0, sigma_t_scattering=1, sigma_t_absorbing=10):
        mu = (sigma_s_scattering, sigma_s_absorbing, sigma_t_scattering, sigma_t_absorbing)
        if mu != self.last_mu:
            self.last_mu = mu
            self.impl.set_rhs_operator_parameters(*mu)



class BoltzmannDiscretizationBase(DiscretizationBase):

    special_operators = frozenset({'lf', 'rhs'})
    special_vector_operators = frozenset({'initial_data'})

    def __init__(self, nt=60, dt=0.056, initial_data=None, lf=None, rhs=None, operators=None, functionals=None,
                 vector_operators=None, products=None, estimator=None, visualizer=None, parameter_space=None,
                 cache_region=None, name=None):
        super(BoltzmannDiscretizationBase, self).__init__(
            initial_data=initial_data, lf=lf, rhs=rhs,
            operators=operators, functionals=functionals, vector_operators=vector_operators, products=products,
            estimator=estimator, visualizer=visualizer, cache_region=cache_region, name=name
        )
        self.nt = nt
        self.dt = dt
        self.solution_space = self.initial_data.range
        self.build_parameter_type(PARAMETER_TYPE, local_global=True)
        self.parameter_space = parameter_space

    def _solve(self, mu=None, return_half_steps=False):
        U = self.initial_data.as_vector(mu)
        U_half = U.empty()
        U_last = U.copy()
        rhs = self.rhs.assemble(mu)
        for n in range(self.nt):
            self.logger.info('Time step {}'.format(n))
            V = U_last - self.lf.apply(U_last) * self.dt
            if return_half_steps:
                U_half.append(V)
            U_last = V + rhs.apply(V, mu=mu) * self.dt
            U.append(U_last)
        if return_half_steps:
            return U, U_half
        else:
            return U

    def as_generic_type(self):
        return BoltzmannDiscretizationBase(**{k: getattr(self, k) for k in BoltzmannDiscretizationBase._init_arguments})


class DuneDiscretization(BoltzmannDiscretizationBase):

    def __init__(self, *args):
        self.solver = solver = Solver(*args)
        initial_data = VectorOperator(ListVectorArray([solver.get_initial_values()]))
        dim = initial_data.range.dim
        lf_operator = LFOperator(self.solver, dim)
        self.non_decomp_rhs_operator = ndrhs = RHSOperator(self.solver, dim)
        affine_part = ConstantOperator(ndrhs.apply(initial_data.range.zeros(), mu=[0., 0., 0., 0.]), initial_data.range)
        rhs_operator = affine_part + \
            LincombOperator([LinearOperator(RHSWithFixedMuOperator(self.solver, dim, mu=[1., 0., 0., 0.]) - affine_part),
                             LinearOperator(RHSWithFixedMuOperator(self.solver, dim, mu=[0., 1., 0., 0.]) - affine_part),
                             LinearOperator(RHSWithFixedMuOperator(self.solver, dim, mu=[1., 0., 1., 0.]) - affine_part),
                             LinearOperator(RHSWithFixedMuOperator(self.solver, dim, mu=[0., 1., 0., 1.]) - affine_part)],
                            [ExpressionParameterFunctional('s[0] - s[2]', PARAMETER_TYPE),
                             ExpressionParameterFunctional('s[1] - s[3]', PARAMETER_TYPE),
                             ExpressionParameterFunctional('s[2]', PARAMETER_TYPE),
                             ExpressionParameterFunctional('s[3]', PARAMETER_TYPE)])
        param_space = CubicParameterSpace(PARAMETER_TYPE, 0., 10.)
        super(DuneDiscretization, self).__init__(initial_data=initial_data, lf=lf_operator, rhs=rhs_operator, nt=60, dt=0.056,
                                                 parameter_space=param_space, name='DuneDiscretization')

    def _solve(self, mu=None, return_half_steps=False):
        return self.as_generic_type().with_(rhs=self.non_decomp_rhs_operator).solve(mu=mu, return_half_steps=return_half_steps)


class LinearOperator(OperatorBase):

    linear = True

    def __init__(self, op):
        self.source = op.source
        self.range = op.range
        self.op = op
        self.name = op.name + '_wrapped_as_linear_op'

    def apply(self, U, ind=None, mu=None):
        return self.op.apply(U, ind=ind, mu=mu)


class DuneOperatorBase(OperatorBase):

    linear = True

    def __init__(self, solver, dim):
        self.solver = solver
        self.source = self.range = DuneStuffVectorSpace(CommonDenseVector, dim)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        mu = self.parse_parameter(mu)
        vectors = U._list if ind is None else [U._list[i] for i in ind]
        result = [DuneStuffVector(self._apply_vector(u.impl, mu)) for u in vectors]
        return ListVectorArray(result, subtype=U.subtype)


class LFOperator(DuneOperatorBase):

    def _apply_vector(self, u, mu):
        return self.solver.impl.apply_LF_operator(u, 0.)  # assume operator is not time-dependent


class GodunovOperator(DuneOperatorBase):

    def _apply_vector(self, u, mu):
        return self.solver.impl.apply_godunov_operator(u, 0.)  # assume operator is not time-dependent


class RHSOperator(DuneOperatorBase):

    linear = False

    def __init__(self, solver, dim):
        super(RHSOperator, self).__init__(solver, dim)
        self.build_parameter_type(PARAMETER_TYPE, local_global=True)

    def _apply_vector(self, u, mu):
        self.solver.set_rhs_operator_params(*map(float, mu['s']))
        return self.solver.impl.apply_rhs_operator(u, 0.)


class RHSWithFixedMuOperator(DuneOperatorBase):

    linear = False

    def __init__(self, impl, dim, mu):
        super(RHSWithFixedMuOperator, self).__init__(impl, dim)
        self.mu = mu

    def _apply_vector(self, u, mu):
        self.solver.set_rhs_operator_params(*self.mu)
        return self.solver.impl.apply_rhs_operator(u, 0.)


class DuneStuffVector(VectorInterface):

    def __init__(self, impl):
        self.impl = impl

    @classmethod
    def make_zeros(cls, subtype):
        impl = subtype[0](subtype[1], 0.)
        return DuneStuffVector(impl)

    @property
    def dim(self):
        return self.impl.dim()

    @property
    def subtype(self):
        return (type(self.impl), self.impl.dim())

    @property
    def data(self):
        return np.frombuffer(self.impl.buffer(), dtype=np.double)

    def copy(self, deep=False):
        return DuneStuffVector(self.impl.copy())

    def scal(self, alpha):
        self.impl.scal(alpha)

    def axpy(self, alpha, x):
        self.impl.axpy(alpha, x.impl)

    def dot(self, other):
        return self.impl.dot(other.impl)

    def l1_norm(self):
        return self.impl.l1_norm()

    def l2_norm(self):
        return self.impl.l2_norm()

    def sup_norm(self):
        return self.impl.sup_norm()

    def components(self, component_indices):
        if len(component_indices) == 0:
            return np.array([], dtype=np.intp)
        assert 0 <= np.min(component_indices)
        assert np.max(component_indices) < self.dim
        return np.array([self.impl[i] for i in component_indices])

    def amax(self):
        return self.impl.amax()

    def __add__(self, other):
        return DuneStuffVector(self.impl + other.impl)

    def __iadd__(self, other):
        self.impl += other.impl
        return self

    __radd__ = __add__

    def __sub__(self, other):
        return DuneStuffVector(self.impl - other.impl)

    def __isub__(self, other):
        self.impl -= other.impl
        return self

    def __mul__(self, other):
        return DuneStuffVector(self.impl * other)

    def __neg__(self):
        return DuneStuffVector(-self.impl)

    def __getstate__(self):
        return type(self.impl), self.data

    def __setstate__(self, state):
        self.impl = state[0](len(state[1]), 0.)
        self.data[:] = state[1]


def DuneStuffVectorSpace(impl_type, dim):
    return VectorSpace(ListVectorArray, (DuneStuffVector, (impl_type, dim)))
