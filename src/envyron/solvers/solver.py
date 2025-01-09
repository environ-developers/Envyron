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

class ElectrostaticSolverMeta:
    """
    Provides decorator to automatically pass attributes of EnvironCharges object
    """

    @classmethod
    def charge_operation(cls, func):
        func = multimethod(func)

        @func.register(cls, EnvironCharges)
        def _(self: cls, charges: EnvironCharges, **kwargs):
            return func(
                self,
                density=charges.density,
                electrons=charges.electrons,
                ions=charges.ions,
                externals=charges.externals,
                dielectric=charges.dielectric,
                electrolyte=charges.electrolyte,
                semiconductor=charges.semiconductor,
                additional=charges.additional,
                **kwargs
            )

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
