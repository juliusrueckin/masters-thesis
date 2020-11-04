from typing import List, Optional, Tuple

import numpy as np
import os
from shapely.geometry import MultiPolygon, Point
from multiprocessing import Pool
import copy

from experimental.utils.dynamic_system import DynamicSystem


class MarkovDecisionProcess:
    """
    This class implements the necessary abstractions to define a Markov decision process based on
    a dynamic system and its Markov partition. It also provides several Monte Carlo algorithms
    presented in the thesis in order to estimate the state transition probability kernel.
    """

    def __init__(
        self,
        dynamic_system: DynamicSystem,
        markov_partition: List[MultiPolygon],
        partition_vertices: List[List[np.array]] = None,
        target_state: np.array = None,
        gamma: float = 0.1,
    ):
        """
        Args:
            dynamic_system (DynamicSystem): including functionality for calculating system dynamics
            markov_partition (list): of multipolygons defining the subsets of the Markov partitions
            partition_vertices (list): list of vertices of each subset of the partition
            target_state (np.array): desired target state of agent in phase space
            gamma (float): discount factor of MDP
        """
        self.dynamic_system = dynamic_system
        self.markov_partition = markov_partition
        self.partition_vertices = partition_vertices
        self.transition_prob_matrix = None
        self.gamma = gamma
        self.target_state = target_state
        if target_state is None:
            self.target_state = self.dynamic_system.compute_fixed_point(init_x=np.array([0.5, 0.5]))

    def estimate_probability_matrix_random_walker_method(
        self, l: int = 100, m: int = 1000, max_sample_trials: int = 1000
    ) -> np.array:
        """
        Implementation of Algorithm 3 presented in the thesis. Monte Carlo method to estimate state
        transition probability matrix by executing m random walks of length l. Motivated by particle
        simulations and graph-theoretic arguments of shifts of finite type.

        Args:
            l (int): length/number of steps of one random walk
            m (int): number of executed random walks
            max_sample_trials (int): maximal number of trials to uniformly sample a point in a subset

        Returns:
            (np.array): estimated state transition probability matrix under certain system/agent dynamics
        """
        n = len(self.markov_partition)
        P = np.zeros((n, n))
        C = np.zeros((n, n))

        num_workers = np.minimum(os.cpu_count(), n)
        distributed_rand_walks = 0
        m_chunk = np.ceil(m / num_workers)
        random_walks_params = []
        for i in range(num_workers):
            if distributed_rand_walks + m_chunk < m:
                random_walks_params.append((l, m_chunk, max_sample_trials))
                distributed_rand_walks += m_chunk
            else:
                random_walks_params.append((l, m - distributed_rand_walks, max_sample_trials))
                break

        with Pool(processes=num_workers) as pool:
            count_matrices = pool.starmap(self.execute_random_walks, random_walks_params)

        for count_matrix in count_matrices:
            C += count_matrix

        for i in range(n):
            if np.linalg.norm(C[i, :], ord=1) > 0:
                P[i, :] = C[i, :] / np.linalg.norm(C[i, :], ord=1)
            else:
                P[i, :] = 0

        self.transition_prob_matrix = P

        return P

    def execute_random_walks(self, l: int = 100, m: int = 1000, max_sample_trials: int = 1000) -> np.array:
        """
        Args:
            l (int): length/number of steps of one random walk
            m (int): number of executed random walks
            max_sample_trials (int): maximal number of trials to uniformly sample a point in a subset

        Returns:
            (np.array): count matrix for random walker's state transitions from subsets i to k
        """
        n = len(self.markov_partition)
        C = np.zeros((n, n))
        num_walks = 0

        while num_walks < m:
            num_steps = 0
            i = np.random.randint(low=0, high=n)
            while num_steps < l:
                p = self.sample_uniform_random_point(self.markov_partition[i], max_sample_trials)
                next_p = Point(self.dynamic_system.phi(p))
                k = self.get_subset_index_of_point(next_p)
                C[i, k] += 1
                i = k
                num_steps += 1
            num_walks += 1

        return C

    def estimate_probabilities_of_state(
        self, i: int, c: int, tau: float, max_sample_trials: int
    ) -> Tuple[int, np.array]:
        """
        Estimate transition probabilities given agent is in state with index i.

        Args:
            i (int): index of subset/state in Markov partition
            c (int): number of sample steps after which we update the probability estimate
            tau (float): threshold for update difference that approximately indicates converge
            max_sample_trials (int): maximal number of trials to uniformly sample a point in a subset

        Returns:
            i (int): index of subset/state in Markov partition
            (np.array): transition probabilities given agent is in state with index i
        """
        n = len(self.markov_partition)
        P = np.zeros(n)
        P_old = np.ones(n)
        C = np.zeros(n)
        samples = 0

        while np.max(np.abs(P - P_old)) >= tau:
            p = self.sample_uniform_random_point(self.markov_partition[i], max_sample_trials)
            assert p is not None, f"Error: Did not find "
            next_p = Point(self.dynamic_system.phi(p))
            k = self.get_subset_index_of_point(next_p)
            assert k is not None, f"Error: Cannot find respective subset of {p}"
            C[k] += 1
            samples += +1
            if samples % c == 0:
                P_old = P
                P = C / np.linalg.norm(C, ord=1)

        return i, P

    def estimate_probability_matrix_pi_method(
        self, c: int = 100, tau: float = 0.001, max_sample_trials: int = 1000
    ) -> np.array:
        """
        Implementation of Algorithm 5 presented in the thesis. Monte Carlo method to estimate state
        transition probability matrix for each subset of the partition separately, so no random walk
        is performed. Additionally, parameters c and tau approximately detect convergence. Motivated
        by the classical Monte Carlo algorithm to estimate pi by approximating the disk's area.

        Args:
            c (int): number of sample steps after which we update the probability estimate
            tau (float): threshold for update difference that approximately indicates converge
            max_sample_trials (int): maximal number of trials to uniformly sample a point in a subset

        Returns:
            (np.array): estimated state transition probability matrix under certain system/agent dynamics
        """
        n = len(self.markov_partition)
        P = np.zeros((n, n))

        state_params = [(i, c, tau, max_sample_trials) for i in range(n)]
        num_workers = np.minimum(os.cpu_count(), n)

        with Pool(processes=num_workers) as pool:
            transition_prob_results = pool.starmap(self.estimate_probabilities_of_state, state_params)

        for transition_prob_result in transition_prob_results:
            i, transition_probs = transition_prob_result
            P[i, :] = transition_probs

        self.transition_prob_matrix = P

        return P

    @staticmethod
    def sample_uniform_random_point(area: MultiPolygon, max_sample_trials: int = 1000) -> Optional[Point]:
        """
        Try to sample a point uniform at random over an arbitrary polygon shape by uniformly sampling
        from its covering rectangle. Exit, if it takes more than max_sample_trials.

        Args:
            area (MultiPolygon): subset of the Markov partition from which we sample a point at random
            max_sample_trials: number of trials to sample a point that is also within the polygon

        Returns:

        """
        min_x, min_y, max_x, max_y = area.bounds
        point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        sample_trials = 1

        while not point.intersects(area) and sample_trials < max_sample_trials:
            point = Point([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            sample_trials += 1

        return point if point.intersects(area) else None

    def get_subset_index_of_point(self, point: Point) -> Optional[int]:
        """
        Args:
            point (Point): some point in the phase space of the dynamic system

        Returns:
            (int): unique index of the subset of the Markov partition that contains the given point
        """
        n = len(self.markov_partition)
        for k in range(n):
            if point.intersects(self.markov_partition[k]):
                return k
        return None

    def ground_truth_probability_matrix(self, markov_partition_phi: List[MultiPolygon]) -> np.array:
        """
        If a partition of one-time applied dynamics to the original Markov partition is available,
        we produce ground truth for the probability estimate by calculating areas of intersection
        between states in self.markov_partition and markov_partition_phi.

        Args:
            markov_partition_phi: partition of one-time applied dynamics to the original Markov partition

        Returns:
            (np.array): exact state transition probability matrix under certain system/agent dynamics
        """
        n = len(self.markov_partition)
        intersection_area_mat = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                area_of_intersection = markov_partition_phi[i].intersection(self.markov_partition[j]).area
                if area_of_intersection < 10e-8:
                    area_of_intersection = 0
                intersection_area_mat[i, j] = area_of_intersection

        probability_mat = np.zeros((n, n))
        for i in range(n):
            probability_mat[i, :] = intersection_area_mat[i, :] / np.linalg.norm(intersection_area_mat[i, :], ord=1)

        return probability_mat

    def euclidean_dist_over_torus(self, x: np.array, y: np.array) -> float:
        """
        Compute euclidean distance between two points over the torus. Not equivalent to the euclidean distance
        over the plane, since identification at the edges of the unit square needs to be considered as
        potentially shorter paths between two points, even if their plane distance is large.

        Args:
            x (np.array): first point on the torus
            y (np.array): second point on the torus

        Returns:
            (float): distance between x and y over the torus
        """
        one_vec = np.ones(x.shape[0]) * self.dynamic_system.m_id
        return np.min(
            [
                np.linalg.norm(x - y, ord=2),
                np.linalg.norm(x - y - one_vec, ord=2),
                np.linalg.norm(x - y + one_vec, ord=2),
            ]
        )

    def g(self, state_idx: int) -> float:
        """
        Cost function g measuring the shortest distance between the center of subset i of the
        partition and the target point over the torus.

        Args:
            state_idx (int): index of current state

        Returns:
            (float): distance between center of subset i and target point
        """
        dists = []
        for x in self.partition_vertices[state_idx]:
            dists.append(self.euclidean_dist_over_torus(x, self.target_state))

        return sum(dists) / len(dists)

    def policy_bellman_operator(self, V: np.array, state_idx: int, cost_func):
        """
        Computes the expected value of following the policy in the current state.

        Args:
            V:
            state_idx (int): index of current state in partition list
            cost_func (callable): cost function g1 or g2

        Returns:
            (float): expected value of following the policy in the current state.
        """
        return cost_func(state_idx) + self.gamma * np.sum(self.transition_prob_matrix[state_idx, :] * V)

    def policy_evaluation(self, cost_func: callable, epsilon: float = 2 * np.finfo(float).eps) -> Tuple[np.array, int]:
        """
        Computes a policy evaluation step of the policy iteration algorithm. For all states, the
        expected long-term value of following the current policy from this state on is calculated.

        Args:
            cost_func (callable): cost function g1 or g2
            epsilon (float): convergence threshold for expected value function over all states

        Returns:
            (np.array): expected value function while following the current policy
        """
        n = len(self.markov_partition)
        V = np.ones(n)
        V_old = np.zeros(n)
        num_iters = 0

        while np.max(np.abs(V_old - V)) > epsilon:
            num_iters += 1
            V_old = copy.deepcopy(V)

            for state_idx in range(n):
                V[state_idx] = self.policy_bellman_operator(V, state_idx, cost_func)

        return V, num_iters
