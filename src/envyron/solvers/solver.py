from abc import ABC

from multimethod import multimethod

from ..cores import CoreContainer
from ..representations import EnvironDensity, EnvironGradient
from ..physical import (
    EnvironCharges,
    EnvironElectrolyte,
    EnvironSemiconductor,
    EnvironDielectric,
)

from inspect import signature

class ElectrostaticSolverMeta:
    """
    Provides decorator to automatically pass attributes of EnvironCharges object
    """

    @classmethod
    def charge_operation(cls, func):
        func = multimethod(func)

        @func.register(cls, EnvironCharges)
        def _(self: cls, charges: EnvironCharges, *args, **kwargs):
            # Get argnames that the explicit function definition expects.
            # This happens as the function is called, not as it is registered.
            # However, @func.register does apparently not overwrite the original
            # signature, so we can still use it after func.register was called.
            argnames = str(signature(func)).split(',')
            argnames = [name.split(':')[0].split('=')[0].strip() for name in argnames]

            charge_args = []

            # Get argnames that are attributes of charges, and extract actual arg
            # if explicit function definition expects arg of same name.
            for name in argnames[1:]:
                for arg in dir(charges):
                    if name == arg:
                        charge_args.append(getattr(charges, arg))

            # Pass args that could be extracted from charges to original function.
            # We assume here that the explicit definition expects any args that
            # can be extracted from charges first, and any args that are passed
            # from somewhere else after that.
            return func(self, *charge_args, *args, **kwargs)

        return func

class ElectrostaticSolver(ABC, ElectrostaticSolverMeta):
    """
    An Electrostatic Solver.
    """

    def __init__(self, cores: CoreContainer) -> None:
        self.cores = cores

    @multimethod
    def poisson(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def poisson(self, charges: EnvironCharges) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def grad_poisson(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def grad_poisson(self, charges: EnvironCharges) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def generalized(
        self,
        density: EnvironDensity,
        dielectric: EnvironDielectric,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def generalized(self, charges: EnvironCharges) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def linearized_pb(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte,
        dielectric: EnvironDielectric = None,
        screening: EnvironDensity = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def linearized_pb(
        self,
        charges: EnvironCharges,
        screening: EnvironDensity = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def pb_nested(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte,
        dielectric: EnvironDielectric = None,
        inner: 'ElectrostaticSolver' = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @multimethod
    def pb_nested(
        self,
        charges: EnvironCharges,
        inner: 'ElectrostaticSolver' = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()
