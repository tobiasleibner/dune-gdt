from itertools import product

import numpy as np

from pymor.vectorarrays.interfaces import VectorSpace
from pymor.vectorarrays.list import VectorInterface, ListVectorArray

import libboltzmann
from libboltzmann import CommonDenseVector
IMPL_TYPES = (CommonDenseVector,)


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
