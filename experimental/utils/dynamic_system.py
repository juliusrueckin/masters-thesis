import numpy as np
from scipy import optimize


class DynamicSystem:
    def __init__(self, phi_non_identified: callable, d_phi: callable, m_id: float):
        self.phi_non_identified = phi_non_identified
        self.d_phi = d_phi
        self.m_id = m_id
        self.fixed_point = None

    def phi(self, x: np.array) -> np.array:
        return self.phi_non_identified(x) % self.m_id

    def identification_occurs(self, x: np.array) -> bool:
        return np.any(x != x % self.m_id)

    def fixed_point_is_hyperbolic(self) -> bool:
        eig_vals = np.linalg.eigvals(self.d_phi(self.fixed_point))
        return np.all(eig_vals != 1)

    def is_fixed_point(self, x: np.array):
        return np.allclose(x, self.phi(x), atol=10e-8)

    def fixed_point_objective_func(self, x: np.array) -> np.array:
        return self.phi_non_identified(x) - x

    def fixed_point_jacobian(self, x: np.array) -> np.array:
        return self.d_phi(x) - np.identity(n=2)

    def compute_fixed_point(self, init_x: np.array = np.array([4, 4])) -> np.array:
        root_sol = optimize.root(self.fixed_point_objective_func, init_x, jac=self.fixed_point_jacobian, method="lm")
        if not root_sol.success:
            print(f"Failed to find a fixed point. Error: {root_sol.message}")
            return None

        self.fixed_point = root_sol.x % self.m_id
        return self.fixed_point
