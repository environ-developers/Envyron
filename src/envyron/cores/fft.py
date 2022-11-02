from numpy import ndarray

from envyron.cores.core import NumericalCore

from ..representations import EnvironDensity, EnvironGradient, EnvironHessian
from ..representations.functions import FunctionContainer


class FFTCore(NumericalCore):
    """docstring"""
