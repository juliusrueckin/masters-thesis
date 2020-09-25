import numpy as np
from scipy import optimize


class DynamicSystem:
    """
    DaynmicSystem class is a wrapper the system dynamics, its Jacobian and functions to compute hyperbolic fixed
    points of the system.
    """

    def __init__(self, phi_non_identified: callable, d_phi: callable, m_id: float):
        """
        Args:
            phi_non_identified (callable): function that implements the system dynamics/ the system's flow
            d_phi (callable): function that implements the Jacobian of the systems flow
            m_id (float): torus identification constant (x mod m_id) defining the cosets of x
        """
        self.phi_non_identified = phi_non_identified
        self.d_phi = d_phi
        self.m_id = m_id
        self.fixed_point = None

    def phi(self, x: np.array) -> np.array:
        """
        Implements the system dynamics over the torus, hence evaluates the system's flow mod m_id.

        Args:
            x: point in system's phase space

        Returns:
            (np.array): point in phase space under application of system's flow
        """
        return self.phi_non_identified(x) % self.m_id

    def identification_occurs(self, x: np.array) -> bool:
        """
        Checks if identification of x with a representative of its torus coset occurs.

        Args:
            x (np.array): point in euclidean plane

        Returns:
            (bool): indicates, if identification of x with coset occurs
        """
        return np.any(x != x % self.m_id)

    def fixed_point_is_hyperbolic(self) -> bool:
        """
        Checks if system's fixed point is hyperbolic, i.e. if absolute values of eigenvalues of Jacobian in fixed
        point are != 1.

        Returns:
            (bool): indicates if computed fixed point is a hyperbolic one
        """
        eig_vals = np.linalg.eigvals(self.d_phi(self.fixed_point))
        return np.all(eig_vals != 1)

    def is_fixed_point(self, x: np.array):
        """
        Checks if a point x in the phase space is a fixed point, i.e. if phi(x) = x.

        Args:
            x (np.array): point in phase space

        Returns:
            (bool): indicates if a given point x is a fixed point
        """
        return np.allclose(x, self.phi(x), atol=10e-8)

    def fixed_point_objective_func(self, x: np.array) -> np.array:
        """
        Implements objective function for applying Levenberg-Marquardt to find fixed point, i.e. phi(x) -x.

        Args:
            x (np.array): point in phase space

        Returns:
            (np.array): value of objective function evaluated at x
        """
        return self.phi_non_identified(x) - x

    def fixed_point_jacobian(self, x: np.array) -> np.array:
        """
        Implements Jacobian of objective function to compute fixed point, i.e. D_phi(x) - I.

        Args:
            x (np.array): point in phase space

        Returns:
            (np.array): Jacobian of objective function evaluated at x
        """
        return self.d_phi(x) - np.identity(n=2)

    def compute_fixed_point(self, init_x: np.array = np.array([4, 4])) -> np.array:
        """
        Computes fixed point of system dynamics via Levenberg-Marquardt algorithm that solves root finding
        problem phi(x) - x = 0, whch is equivalent to finding a fixed point phi(x) = x.

        Args:
            init_x (np.array): point in phase space in which we start the Levenberg-Marquardt algorithm

        Returns:
            (np.array): iteratively calculated fixed point of system dynamics
        """
        root_sol = optimize.root(self.fixed_point_objective_func, init_x, jac=self.fixed_point_jacobian, method="lm")
        if not root_sol.success:
            print(f"Failed to find a fixed point. Error: {root_sol.message}")
            return None

        self.fixed_point = root_sol.x % self.m_id
        return self.fixed_point
