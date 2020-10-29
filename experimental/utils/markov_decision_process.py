from typing import List, Optional, Tuple

import numpy as np
import os
from shapely.geometry import MultiPolygon, Point
from multiprocessing import Pool

from experimental.utils.dynamic_system import DynamicSystem


class MarkovDecisionProcess:
    """
    This class implements the necessary abstractions to define a Markov decision process based on
    a dynamic system and its Markov partition. It also provides several Monte Carlo algorithms
    presented in the thesis in order to estimate the state transition probability kernel.
    """

    def __init__(self, dynamic_system: DynamicSystem, markov_partition: List[MultiPolygon]):
        """
        Args:
            dynamic_system (DynamicSystem): including functionality for calculating system dynamics
            markov_partition (list): of multipolygons defining the subsets of the Markov partitions
        """
        self.dynamic_system = dynamic_system
        self.markov_partition = markov_partition

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
