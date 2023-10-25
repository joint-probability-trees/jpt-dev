from collections import OrderedDict, deque
from math import prod
from operator import itemgetter, attrgetter
from typing import Any, Iterator, List

import numpy as np
from dnutils import first, ifnone, project

from jpt.base.utils import Heap
from jpt.distributions import Numeric
from jpt.variables import VariableMap, Variable, ValueAssignment


class MPEState:

    def __init__(
            self,
            assignment: VariableMap,
    ):
        self.assignment = assignment
        self.latest_variable = None

    def copy(self):
        return MPEState(
            assignment=self.assignment.copy(deep=True),
        )

    def assign(self, variable: Variable or str, value: Any) -> 'MPEState':
        state_ = self.copy()
        state_.assignment[variable] = value
        state_.latest_variable = variable
        return state_

    def __repr__(self):
        return '<MPESearchStat assignment: %s>' % (
            self.assignment
        )


class BnBNode:

    def __init__(self, solver, state, actual_cost):
        self.solver: MPESolver = solver
        self.old_state: MPEState = state
        self.variable: Variable = first(self.solver.variable_order(state))
        self.values: Iterator[Any] = iter(
            self.solver.constraints[self.variable].keys() if self.variable is not None else iter([])
        )
        self.actual_cost = actual_cost
        self.step_cost = 0
        self.state: MPEState = self._generate_next()

    def _generate_next(self):
        try:
            value = next(self.values)
        except StopIteration:
            return None
        else:
            self.step_cost = self.solver.constraints[self.variable][value]
            return self.old_state.assign(self.variable, value)

    def pop(self):
        solution_node = None
        if not self.free_variables:
            solution_node = BnBSolution(
                self.state,
                self.cost
            )
        cost = self.actual_cost + self.step_cost
        state = self.state
        self.state = self._generate_next()
        if solution_node is not None:
            return solution_node
        return BnBNode(
            self.solver,
            state,
            cost
        )

    @property
    def cost(self):
        return self.actual_cost + self.step_cost + self.solver.lb.get(self.variable, 0)

    @property
    def free_variables(self):
        return set(self.solver.constraints.keys()).difference(self.state.assignment.keys())


class BnBSolution:

    def __init__(self, state: MPEState, cost: float):
        self.state = state
        self.cost = cost


# ----------------------------------------------------------------------------------------------------------------------

class MPESolver:
    '''
    Solver for k-MPE inference about n independent variables

    This algorithm iteratively constructs all ``k`` most probable explanations
    in descending order by performing a branch-and-bound search in the space of
    atomic areas of the respective distributions. In constrast to classic BnB search,
    we do not throw away the pruned states but save them in a priority queue, where
    we can continue the search for the subsequent 2nd, 3rd, ..., k-th best solution.
    '''

    def __init__(
            self,
            distributions: VariableMap,
            likelihood_divisor: float = None
    ):
        # save the distributions
        self.distributions: VariableMap = distributions.copy()

        # Set up the WCSP variables and their domains and constraints.
        self.domains = VariableMap(variables=self.distributions.variables)
        self.constraints = VariableMap(variables=self.distributions.variables)

        likelihoods = []

        # fill domains and constraints
        for var, dist in self.distributions.items():
            k_mpe = list(dist._k_mpe())
            self.domains[var] = {
                frozenset(k) if isinstance(k, set) else k: v for k, v in k_mpe
            }

            if isinstance(dist, Numeric):
                _, likelihood_max = dist._mpe()
                likelihoods.append(likelihood_max)

        # Build the unary constraints given by the log of
        # an events probability
        likelihood_divisor = ifnone(
            likelihood_divisor,
            max(likelihoods) if likelihoods else 1
        )

        for var, dist in self.distributions.items():
            self.constraints[var] = OrderedDict(
                list(
                    sorted([
                        (
                            frozenset(val) if not isinstance(dist, Numeric) else val,
                            -np.log(
                                (dist.pdf / likelihood_divisor).eval(val.any_point())
                                if isinstance(dist, Numeric)
                                else dist._p(val)
                            )
                        )
                        for val in self.domains[var]],
                        key=itemgetter(1)
                    )
                )
            )
        # Ensure node consistency, i.e. remove values with infinite costs
        for var, constraints in self.constraints.items():
            for val, cost in list(constraints.items()):
                if np.isposinf(cost):
                    del constraints[val]
                elif np.isneginf(cost):
                    constraints[val] = 0

        # Construct the local lower bounds
        self.lb = {
            v: c_min for v, c_min in zip(
                self.variable_order(),
                list(
                    np.cumsum(
                        [min(self.constraints[var].values()) for var in reversed(list(self.variable_order()))]
                    )[:-1][::-1]
                ) + [0]
            )
        }

    def likelihood(self, solution: BnBSolution):
        return prod([
            self.domains[var][val] for var, val in solution.state.assignment.items()
        ])

    def is_goal_state(self, state: MPEState) -> bool:
        return not set(
            self.constraints.keys()
        ).difference(
            state.assignment.keys()
        )

    def variable_order(self, state: MPEState = None) -> Iterator[Variable]:
        """
        :param state: The state to start from
        :return: An iterator over free variables for a state sorted by number of different possible states
        (lowest amount of values first).
        """
        if state is not None:
            for var in self.variable_order():  # min([(var, len(dom)) for var, dom in state.domains.items()], key=itemgetter(1))[0]
                if var not in state.assignment:
                    yield var
            return
        yield from project(
            sorted([(var, len(constraints)) for var, constraints in self.constraints.items()], key=itemgetter(1)),
            0
        )

    def solve(
            self,
            k: int = 0
    ) -> Iterator[List[ValueAssignment]]:
        '''
        Generate ``k`` MPE states with decreasing probability.

        :return:
        '''
        ub = np.inf  # Global upper bound
        solutions = Heap(key=attrgetter('cost'))  # Buffer for all solutions
        pruned = Heap(key=attrgetter('cost'))  # Buffer for pruned nodes for later continuation
        fringe = deque([  # FIFO queue for depth-first search
            BnBNode(
                self,
                MPEState(
                    VariableMap(variables=self.constraints.keys())
                ),
                0
            )]
        )
        count = 0

        while fringe or solutions:
            while fringe:
                node = fringe.popleft()

                if self.is_goal_state(node.state):
                    if node.cost <= ub:
                        ub = node.cost
                        solutions.push(node.pop())
                        if node.state is not None:  # We can encounter equivalent solutions, but no better ones
                            fringe.appendleft(node)
                    elif node.state is not None:
                        pruned.push(node)
                else:
                    if ub < node.cost < np.inf:
                        pruned.push(node)
                    else:
                        new_node = node.pop()
                        if node.state is not None:
                            fringe.appendleft(node)
                        fringe.appendleft(
                            new_node,
                        )
            while solutions and solutions[0].cost == ub:
                solution = solutions.pop()
                count += 1
                yield solution.state.assignment, self.likelihood(solution)
                if k and count >= k:
                    return
            ub = solutions[0].cost if solutions else np.inf

            while pruned:  # Continue the search at the states that have been pruned
                fringe.append(pruned.pop())
