from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from experimental.utils.dynamic_system import DynamicSystem
from shapely.geometry import MultiLineString, Point, MultiPoint


class Partition:
    def __init__(self, dynamic_system: DynamicSystem):
        self.dynamic_system = dynamic_system
        self.branches = None
        self.intersection_points = None

    def compute_unstable_eigenspace(self, x: np.array):
        eig_vals, eig_vects = np.linalg.eig(self.dynamic_system.d_phi(x))
        eig_vects = np.transpose(eig_vects)
        msk = eig_vals > 1
        return eig_vects[msk][0]

    def compute_stable_eigenspace(self, x: np.array):
        eig_vals, eig_vects = np.linalg.eig(self.dynamic_system.d_phi(x))
        eig_vects = np.transpose(eig_vects)
        msk = eig_vals < 1
        return eig_vects[msk][0]

    def wt_s(self, last_wt_s: np.array, delta: float=10e-3, reverse_direction: bool=False) -> np.array:
        direction_sign = 1 if not reverse_direction else -1
        return last_wt_s + direction_sign * delta * self.compute_stable_eigenspace(last_wt_s)

    def wt_u(self, last_wt_u: np.array, delta: float=10e-3, reverse_direction: bool=False) -> np.array:
        direction_sign = 1 if not reverse_direction else -1
        return last_wt_u + direction_sign * delta * self.compute_unstable_eigenspace(last_wt_u)

    @staticmethod
    def get_num_of_intersections(intersection_points: Union[Point, MultiPoint]) -> int:
        if intersection_points.is_empty:
            return 0
        elif isinstance(intersection_points, Point):
            return 1
        else:
            return np.array(intersection_points).shape[0]

    @staticmethod
    def get_new_intersection_point(intersection_points: np.array, num_intersections: int, overall_intersection_points: np.array) -> Optional[np.array]:
        if num_intersections == 1:
            return intersection_points

        msk = np.ones(intersection_points.shape[0], dtype=bool)
        for j in range(intersection_points.shape[0]):
            for k in range(overall_intersection_points.shape[0]):
                if np.allclose(intersection_points[j, :], overall_intersection_points[k]):
                    msk[j] = False
                    break

        if msk.sum() != 1:
            return None

        return intersection_points[msk][0]

    def compute_partition(self, num_iters: int, delta: float) -> Tuple[Optional[Dict], Optional[List]]:
        if self.dynamic_system.fixed_point is None:
            if self.dynamic_system.compute_fixed_point() is None:
                print(f"Failed calculating a partition, since no fixed point found.")
                return None, None

        if not self.dynamic_system.is_fixed_point(self.dynamic_system.fixed_point):
            print(f"Failed calculating a partition, since calculated fixed point is not a real fixed point.")
            return None, None

        if not self.dynamic_system.fixed_point_is_hyperbolic():
            print(f"Failed calculating a partition, since fixed point is not a hyperbolic one.")
            return None, None

        branches = {"W_u1": [[self.dynamic_system.fixed_point]], "W_u2": [[self.dynamic_system.fixed_point]], "W_s1": [[self.dynamic_system.fixed_point]], "W_s2": [[self.dynamic_system.fixed_point]]}
        unstable_branches = ["W_u1", "W_u2"]
        stable_branches = ["W_s1", "W_s2"]
        approx_funcs = {"W_u1": (self.wt_u, True), "W_u2": (self.wt_u, False), "W_s1": (self.wt_s, True), "W_s2": (self.wt_s, False)}
        overall_intersection_points = []

        for i in range(num_iters):
            print(f"ITERATION: {i+1}")
            stopped_branches = []
            trace_branches = list(branches.keys())
            while len(stopped_branches) < len(branches.keys()):
                trace_branches = [branch for branch in trace_branches if branch not in stopped_branches]
                for branch_key in trace_branches:
                    last_wt = branches[branch_key][-1][-1]
                    approx_func, rev_dir = approx_funcs[branch_key]
                    new_wt = approx_func(last_wt, delta=delta, reverse_direction=rev_dir)

                    if not self.dynamic_system.identification_occurs(new_wt):
                        branches[branch_key][-1].append(new_wt)
                    else:
                        if len(branches[branch_key][-1]) == 1:
                            branches[branch_key][-1].append(branches[branch_key][-1][0])

                        new_wt = new_wt % self.dynamic_system.m_id
                        branches[branch_key].append([new_wt, new_wt])

                for unstable_branch in unstable_branches:
                    for stable_branch in stable_branches:
                        if unstable_branch in stopped_branches and stable_branch in stopped_branches:
                            continue
                        intersection_points = MultiLineString(branches[unstable_branch]).intersection(MultiLineString(branches[stable_branch]))
                        intersection_points = intersection_points.difference(Point(self.dynamic_system.fixed_point))
                        num_intersections = self.get_num_of_intersections(intersection_points)

                        if num_intersections > i:
                            intersection_point = self.get_new_intersection_point(np.array(intersection_points), num_intersections, np.array(overall_intersection_points))
                            if intersection_point is None:
                                    continue

                            dist_unstable = np.linalg.norm(np.array(intersection_point) - branches[unstable_branch][-1][-1], ord=2)
                            dist_stable = np.linalg.norm(np.array(intersection_point) - branches[stable_branch][-1][-1], ord=2)
                            latter_branch = unstable_branch if dist_unstable < dist_stable else stable_branch
                            if latter_branch not in stopped_branches:
                                stopped_branches.append(latter_branch)
                                print(f"Stop {latter_branch} at {intersection_point}.")
                                overall_intersection_points.append(intersection_point)

        self.branches = branches
        self.intersection_points = overall_intersection_points
        return self.branches, self.intersection_points

    def plot_partition(self):
        m_id = self.dynamic_system.m_id
        unit_square = np.array([[0,0], [0,m_id], [m_id,m_id], [m_id,0], [0,0]])
        plt.plot(unit_square[:, 0], unit_square[:, 1], "r-")

        for branch in self.branches.keys():
            for line in MultiLineString(self.branches[branch]):
                plt.plot(np.array(line.coords[:])[:, 0], np.array(line.coords[:])[:, 1], "b-")

        plt.plot(self.dynamic_system.fixed_point[0], self.dynamic_system.fixed_point[1], "yo")

        intersection_points = np.array(self.intersection_points)
        plt.plot(intersection_points[:, 0], intersection_points[:, 1], "go")

        plt.show()
