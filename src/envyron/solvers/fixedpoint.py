# Using fixed point iteration to solve the poisson equation
# Using class and inheritance to implement the fixed point iteration
import numpy as np
from ..domains import DirectGrid
from ..representations import DirectDensity, DirectField, DirectGradient
from .iterative import IterativeSolver


class FixedPointSolver(IterativeSolver):
    """
    Fixed point iteration solver for the Poisson equation
    Uses the FixedPointSolver class to implement the fixed point iteration
    We use rho, epsilon, the gradient of log epsilon and the mixing paraetr 
    """
    def __init__(self, rho, epsilon, grad_log_epsilon, mixing, max_iter, tol ):
        self.rho = rho
        self.epsilon = epsilon
        self.grad_log_epsilon = grad_log_epsilon
        self.mixing = mixing
        self.max_iter = max_iter
        self.tol = tol 
        def set_rho(sel, rho):
            self.rho = rho
        def setepsilon(self, epsilon):
            self.epsilon = epsilon
        def setgrad_log_epsilon(self):
            self.grad_log_epsilon = grad_log_epsilon
        def getrho(self):
            return self.rho
        def getepsilon(self):
            return self.epsilon
        def getgrad_log_epsilon(self):
            return self.grad_log_epsilon
        super().__init__(rho, epsilon, grad_log_epsilon, mixing, max_iter, tol)
    
    #initialize the iterative solver to zero
    def pol_iter(self):
        return DirectField(grid = self.rho.grid, data = np.zeros(self.rho.grid.shape))
# Compute the fixed polarization charge density
    def pol_fixed(self):
        return (1-self.epsilon)*self.rho/ self.espsilon
    def pol_fixed_charge(self):
        return self.pol_fixed.integral()
# Compute the charge of the input density
    def rho_charge(self):
        return self.rho.integral()



   #Initialize the iterative solver
    def solveer(self):
        iteration = 0
        while iteration < self.max_iter:
            rho_total = self.rho+self.pol_iter + self.pol_fixed
            #Compute the electrostatic field from the total charge density
            def calc_electrostatic_field(self):
                return self.calc_electrostatic_field(rho_total)
            pol_new = np.einsum('lijk, lijk->ijk', self.grad_log_epsilon, self.calc_electrostatic_field/(4*np.pi))
            pol_res = (self.mixing -1)*(self.pol_iter - pol_new)
            self.pol_iter = self.pol_iter + pol_res
            self.pol_fixed_charge = self.pol_iter.integral()
        #Check the convergence of the fixed point iteration
            def mean_squared_density(self):
                return self.mean_squared_denisty(pol_res)
            iteration += 1
            if self.mean_squared_density < self.tol:
                return self.pol_iter + self.pol_fixed
            else:
                raise ValueError('The fixed point iteration did not converge')
            
    
    


   