from typing import overload

from abc import ABC
from multipledispatch import dispatch

from ..cores import CoreContainer
from ..representations import EnvironDensity, EnvironGradient
from ..physical import (
    EnvironCharges,
    EnvironElectrolyte,
    EnvironSemiconductor,
    EnvironDielectric,
)


class ElectrostaticSolver(ABC):
    """
    An Electrostatic Solver.
    """

    def __init__(self, cores: CoreContainer) -> None:
        self.cores = cores

    @overload
    @dispatch(
        EnvironDensity,
        EnvironElectrolyte,
        EnvironSemiconductor,
    )
    def poisson(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(EnvironCharges)
    def poisson(self, charges: EnvironCharges) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(
        EnvironDensity,
        EnvironElectrolyte,
        EnvironSemiconductor,
    )
    def grad_poisson(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(EnvironCharges)
    def grad_poisson(self, charges: EnvironCharges) -> EnvironGradient:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(
        EnvironDensity,
        EnvironDielectric,
        EnvironElectrolyte,
        EnvironSemiconductor,
    )
    def generalized(
        self,
        density: EnvironDensity,
        dielectric: EnvironDielectric,
        electrolyte: EnvironElectrolyte = None,
        semiconductor: EnvironSemiconductor = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(EnvironCharges)
    def generalized(self, charges: EnvironCharges) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(
        EnvironDensity,
        EnvironElectrolyte,
        EnvironDielectric,
        EnvironDensity,
    )
    def linearized_pb(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte,
        dielectric: EnvironDielectric = None,
        screening: EnvironDensity = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(
        EnvironCharges,
        EnvironDensity,
    )
    def linearized_pb(
        self,
        charges: EnvironCharges,
        screening: EnvironDensity = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(
        EnvironDensity,
        EnvironElectrolyte,
        EnvironDielectric,
        'ElectrostaticSolver',
    )
    def pb_nested(
        self,
        density: EnvironDensity,
        electrolyte: EnvironElectrolyte,
        dielectric: EnvironDielectric = None,
        inner: 'ElectrostaticSolver' = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()

    @overload
    @dispatch(
        EnvironCharges,
        'ElectrostaticSolver',
    )
    def pb_nested(
        self,
        charges: EnvironCharges,
        inner: 'ElectrostaticSolver' = None,
    ) -> EnvironDensity:
        """docstring"""
        raise NotImplementedError()
