'''
Ensembles of Joint Probability Trees.

This module provides first-class ensemble support for JPTs, in three
flavors that share one taxonomy:

  - :class:`MixtureJPT` --- the abstract base: a finite convex mixture
    :math:`\\bar P(z) = \\sum_m w_m P^{(m)}(z)` of JPTs over a shared set
    of variables. Mixtures of normalized joints are again normalized
    joints, so the full generative query surface (``infer``,
    ``posterior``, ``expectation``, ``likelihood``, ``pdf``, ``mpe``,
    ``sample``) is preserved.
  - :class:`JPTForest` --- bagging: members are trained independently on
    bootstrap resamples and mixed uniformly. Reduces variance, keeps the
    joint.
  - :class:`JPTLikelihoodBoost` --- generative boosting: greedy additive
    mixture grown by (approximate) Frank--Wolfe ascent of the empirical
    joint log-likelihood. Each round trains a component on data
    reweighted towards the samples the current mixture explains worst
    (a weighted-log-likelihood surrogate of the exact linear oracle
    :math:`w_i \\propto 1/P_{t-1}(z_i)`) and mixes it in with a
    line-searched step size. Keeps the joint; no partition function is
    ever needed because the mixture stays normalized by construction.
  - :class:`JPTBoost` --- *discriminative* gradient boosting with JPTs
    as base learners (squared-error regression). It models the
    conditional :math:`E[y \\mid x]` by an additive expansion and
    deliberately gives up the joint --- this is the trade that makes
    additive boosting tractable. Classification (softmax-gradient) is
    future work.

Which variants preserve the generative joint:

  - ``JPTForest``: yes.
  - ``JPTLikelihoodBoost``: yes.
  - ``JPTBoost``: no --- conditional model only.
'''
import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from dnutils import first, ifnone

from .base.errors import Unsatisfiability
from .base.intervals import ContinuousSet, UnionSet
from .base.utils import format_path, normalized
from .base.utils.multicore import DummyPool, Pool
from .distributions import Integer, Multinomial, Numeric
from .trees import JPT
from .variables import (
    LabelAssignment,
    Variable,
    VariableAssignment,
    VariableMap,
    infer_from_dataframe,
)

logger = logging.getLogger('/jpt/ensembles')

_EPS = 1e-12


# ----------------------------------------------------------------------------------------------------------------------

def _fit_member(args: Tuple[JPT, pd.DataFrame, int | None]) -> JPT:
    '''Worker: learn an untrained member tree on its resample.

    Module-level so it is picklable by ``multiprocessing``.
    '''
    tree, data, multicore = args
    tree.learn(data, multicore=multicore)
    return tree


def _format_evidence(evidence: Any) -> str:
    '''Readable rendering of an evidence term for error messages.'''
    if isinstance(evidence, VariableMap):
        return format_path(evidence)
    return repr(evidence)


def _representative_point(value: Any) -> Any:
    '''Extract a representative scalar/label from an MPE state value.

    MPE states map variables to *sets* (intervals for numeric variables,
    label sets for symbolic ones) over which the maximum density is
    attained; density evaluation needs a point from that set.
    '''
    if isinstance(value, UnionSet):
        value = first(value.intervals)
    if isinstance(value, ContinuousSet):
        lower, upper = value.lower, value.upper
        if np.isfinite(lower) and np.isfinite(upper):
            return .5 * (lower + upper)
        elif np.isfinite(lower):
            return lower
        elif np.isfinite(upper):
            return upper
        return 0.
    if isinstance(value, (set, frozenset, list, tuple)):
        return first(value)
    if hasattr(value, 'min') and np.isfinite(value.min):
        return value.min
    return value


# ----------------------------------------------------------------------------------------------------------------------

class MixtureJPT:
    '''A finite convex mixture of JPTs over a shared set of variables.

    The mixture :math:`\\bar P(z) = \\sum_m w_m P^{(m)}(z)` of normalized
    joint distributions is itself a normalized joint distribution, so all
    *unconditional* query functionals distribute linearly over the
    members. Conditional queries do **not** commute with the mixture
    weights; they require the evidence-adjusted *responsibilities*
    :math:`r_m(e) \\propto w_m P^{(m)}(e)`, which is what
    :meth:`posterior` and :meth:`infer` compute.

    This class is the common base of :class:`JPTForest` and
    :class:`JPTLikelihoodBoost`; subclasses implement :meth:`learn`.
    '''

    logger = logger

    def __init__(
            self,
            variables: List[Variable] | None = None,
            targets: List[str | Variable] | None = None,
            min_samples_leaf: float | int = .01,
            min_impurity_improvement: float | None = None,
            max_depth: int | None = None
    ) -> None:
        '''
        :param variables: the variables of the joint. If ``None``, they are
            inferred from the training DataFrame on :meth:`learn` via
            :func:`jpt.variables.infer_from_dataframe`.
        :param targets: optional target designation passed through to the
            member trees (see :class:`jpt.trees.JPT`).
        :param min_samples_leaf: member-tree leaf size (int: absolute,
            float: fraction of the data).
        :param min_impurity_improvement: member-tree split threshold.
        :param max_depth: member-tree depth cap.
        '''
        self._variables: List[Variable] | None = (
            list(variables) if variables is not None else None
        )
        self.targets = targets
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement
        self.max_depth = max_depth
        self.members: List[JPT] = []
        self.weights: List[float] = []

    # ------------------------------------------------------------------------------------------------------------------

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return tuple(ifnone(self._variables, ()))

    @property
    def varnames(self) -> Dict[str, Variable]:
        return {var.name: var for var in self.variables}

    def __len__(self) -> int:
        return len(self.members)

    def _assert_fitted(self) -> None:
        if not self.members:
            raise RuntimeError(
                '%s is not fitted yet.' % type(self).__name__
            )

    def _init_variables(self, data: pd.DataFrame) -> None:
        '''Resolve the shared variable set from ``data`` if not preset.

        All members are constructed over the *same* ``Variable`` objects so
        that member densities are comparable and the mixture is
        well-defined.
        '''
        if self._variables is None:
            self._variables = infer_from_dataframe(
                data,
                scale_numeric_types=False
            )

    def _blueprint(self) -> JPT:
        '''An untrained member tree over the shared variables.'''
        return JPT(
            variables=self._variables,
            targets=self.targets,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_improvement=self.min_impurity_improvement,
            max_depth=self.max_depth
        )

    def learn(self, data: pd.DataFrame) -> 'MixtureJPT':
        raise NotImplementedError()

    def fit(self, data: pd.DataFrame) -> 'MixtureJPT':
        return self.learn(data)

    # ------------------------------------------------------------------------------------------------------------------
    # Generative query surface

    def likelihood(
            self,
            data: pd.DataFrame | np.ndarray,
            **kwargs
    ) -> np.ndarray:
        '''Per-row mixture density :math:`\\bar P(z_i) = \\sum_m w_m P^{(m)}(z_i)`.

        Keyword arguments are passed through to :meth:`jpt.trees.JPT.likelihood`.
        '''
        self._assert_fitted()
        result = np.zeros(len(data))
        for weight, member in zip(self.weights, self.members):
            result += weight * np.asarray(member.likelihood(data, **kwargs))
        return result

    def log_likelihood(self, data: pd.DataFrame | np.ndarray) -> float:
        '''Mean joint log-likelihood of ``data`` under the mixture.'''
        return float(
            np.mean(np.log(np.clip(self.likelihood(data), _EPS, None)))
        )

    def pdf(self, values: VariableAssignment | Dict[str, Any]) -> float:
        '''Mixture density of one fully assigned world.'''
        self._assert_fitted()
        return sum(
            weight * member.pdf(
                member.bind(values) if isinstance(values, dict) else values
            )
            for weight, member in zip(self.weights, self.members)
        )

    def infer(
            self,
            query: Dict[Variable | str, Any] | VariableAssignment,
            evidence: Dict[Variable | str, Any] | VariableAssignment = None,
            fail_on_unsatisfiability: bool = True
    ) -> float | None:
        '''Mixture conditional :math:`\\bar P(q \\mid e)`.

        Computed as
        :math:`\\sum_m w_m P^{(m)}(q, e) \\,/\\, \\sum_m w_m P^{(m)}(e)`
        --- the responsibility-weighted form; the naive
        :math:`\\sum_m w_m P^{(m)}(q \\mid e)` would be incorrect because
        conditioning does not commute with the mixture weights.
        '''
        self._assert_fitted()
        p_q, p_e = 0., 0.
        for weight, member in zip(self.weights, self.members):
            p_e_m = (
                member.infer(evidence, fail_on_unsatisfiability=False)
                if evidence else 1.
            )
            if not p_e_m:
                continue
            cond_m = member.infer(
                query,
                evidence,
                fail_on_unsatisfiability=False
            )
            p_e += weight * p_e_m
            if cond_m is not None:
                p_q += weight * cond_m * p_e_m
        if not p_e:
            if fail_on_unsatisfiability:
                raise Unsatisfiability(
                    'Evidence %s is unsatisfiable.' % _format_evidence(evidence)
                )
            return None
        return p_q / p_e

    def posterior(
            self,
            variables: List[Variable | str] = None,
            evidence: Dict[Variable | str, Any] | VariableAssignment = None,
            fail_on_unsatisfiability: bool = True
    ) -> VariableMap | None:
        '''Posterior distribution of every variable in ``variables``.

        Member posteriors are merged with the evidence-adjusted
        responsibilities :math:`r_m \\propto w_m P^{(m)}(e)`.
        '''
        self._assert_fitted()
        responsibilities = []
        posteriors = []
        for weight, member in zip(self.weights, self.members):
            p_e_m = (
                member.infer(evidence, fail_on_unsatisfiability=False)
                if evidence else 1.
            )
            posterior_m = (
                member.posterior(
                    variables,
                    evidence,
                    fail_on_unsatisfiability=False
                ) if p_e_m else None
            )
            if posterior_m is None:
                continue
            responsibilities.append(weight * p_e_m)
            posteriors.append(posterior_m)
        try:
            responsibilities = normalized(responsibilities)
        except ValueError:
            posteriors = []
        if not posteriors:
            if fail_on_unsatisfiability:
                raise Unsatisfiability(
                    'Evidence %s is unsatisfiable.' % _format_evidence(evidence)
                )
            return None
        result = VariableMap()
        for var in first(posteriors).keys():
            dists = [p[var] for p in posteriors]
            if var.numeric:
                result[var] = Numeric.merge(dists, weights=responsibilities)
            elif var.symbolic:
                result[var] = Multinomial.merge(dists, weights=responsibilities)
            elif var.integer:
                result[var] = Integer.merge(dists, weights=responsibilities)
        return result

    def expectation(
            self,
            variables: Iterable[Variable | str] | None = None,
            evidence: Dict[Variable | str, Any] | VariableAssignment = None,
            fail_on_unsatisfiability: bool = True
    ) -> VariableMap | None:
        '''Expected value of all ``variables`` under the mixture posterior.'''
        posteriors = self.posterior(
            variables,
            evidence,
            fail_on_unsatisfiability=fail_on_unsatisfiability
        )
        if posteriors is None:
            return None
        result = VariableMap()
        for var, dist in posteriors.items():
            result[var] = dist.expectation()
        return result

    def mpe(
            self,
            evidence: Dict[Variable | str, Any] | VariableAssignment = None,
            fail_on_unsatisfiability: bool = True
    ) -> Tuple[List[LabelAssignment], float] | None:
        '''Approximate most probable explanation under the mixture.

        Exact MPE of a mixture requires maximizing a *sum* of tree
        densities, which does not decompose over any single tree's leaves.
        This implementation enumerates the members' individual MPE states
        as candidates and re-scores each candidate under the full mixture
        density (at a representative point of the candidate region),
        returning the best-scoring state. The result is a lower bound on
        the true mixture MPE.
        '''
        self._assert_fitted()
        candidates: List[LabelAssignment] = []
        for member in self.members:
            result = member.mpe(evidence, fail_on_unsatisfiability=False)
            if result is None:
                continue
            candidates.extend(result[0])
        if not candidates:
            if fail_on_unsatisfiability:
                raise Unsatisfiability(
                    'Evidence %s is unsatisfiable.' % _format_evidence(evidence)
                )
            return None
        best_state, best_score = None, -np.inf
        for state in candidates:
            point = {
                var.name: _representative_point(value)
                for var, value in state.items()
            }
            score = self.pdf(point)
            if score > best_score:
                best_state, best_score = state, score
        return [best_state], best_score

    def sample(self, amount: int) -> np.ndarray:
        '''Draw ``amount`` samples: pick a member by weight, sample from it.'''
        self._assert_fitted()
        counts = np.random.multinomial(amount, np.asarray(self.weights))
        samples = np.vstack([
            member.sample(int(count))
            for count, member in zip(counts, self.members)
            if count
        ])
        np.random.shuffle(samples)
        return samples

    # ------------------------------------------------------------------------------------------------------------------
    # Serialization

    def _hyperparams_to_json(self) -> Dict[str, Any]:
        return {
            'targets': (
                [v.name if isinstance(v, Variable) else v for v in self.targets]
                if self.targets else None
            ),
            'min_samples_leaf': self.min_samples_leaf,
            'min_impurity_improvement': self.min_impurity_improvement,
            'max_depth': self.max_depth,
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            'type': type(self).__name__,
            **self._hyperparams_to_json(),
            'variables': [v.to_json() for v in self.variables],
            'weights': list(self.weights),
            'members': [m.to_json() for m in self.members],
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'MixtureJPT':
        '''Reconstruct a mixture from its JSON representation.

        Dispatches on the ``'type'`` key to the concrete subclass.
        '''
        clazz = {
            c.__name__: c
            for c in (JPTForest, JPTLikelihoodBoost)
        }.get(data['type'])
        if clazz is None:
            raise TypeError(
                'Unknown mixture type: %s' % data['type']
            )
        return clazz._from_json(data)

    @classmethod
    def _restore_members(cls, mixture: 'MixtureJPT', data: Dict[str, Any]) -> None:
        variables = [Variable.from_json(d) for d in data['variables']]
        mixture._variables = variables
        mixture.weights = list(data['weights'])
        mixture.members = [
            JPT.from_json(d, variables=variables)
            for d in data['members']
        ]

    def __getstate__(self):
        return self.to_json()

    def __setstate__(self, state):
        self.__dict__ = MixtureJPT.from_json(state).__dict__

    def __eq__(self, other) -> bool:
        return all((
            type(self) is type(other),
            np.allclose(self.weights, other.weights),
            self.members == other.members,
            self.min_samples_leaf == other.min_samples_leaf,
            self.min_impurity_improvement == other.min_impurity_improvement,
            self.max_depth == other.max_depth,
        ))


# ----------------------------------------------------------------------------------------------------------------------

class JPTForest(MixtureJPT):
    '''Bootstrap-aggregated JPTs ("bagging"): a uniform mixture of trees
    trained independently on bootstrap resamples of the data.

    Fully generative: the forest is a normalized joint distribution and
    supports the complete JPT query surface. Members are independent, so
    training parallelizes over :class:`jpt.base.utils.multicore.Pool`.
    '''

    def __init__(
            self,
            variables: List[Variable] | None = None,
            n_estimators: int = 10,
            targets: List[str | Variable] | None = None,
            min_samples_leaf: float | int = .01,
            min_impurity_improvement: float | None = None,
            max_depth: int | None = None,
            bootstrap: bool = True,
            random_state: int | None = None
    ) -> None:
        '''
        :param n_estimators: the number of member trees.
        :param bootstrap: resample with replacement (``True``, default) or
            train every member on the full data (``False`` --- only useful
            with a randomized member learner).
        :param random_state: seed for the bootstrap resampling.
        '''
        super().__init__(
            variables=variables,
            targets=targets,
            min_samples_leaf=min_samples_leaf,
            min_impurity_improvement=min_impurity_improvement,
            max_depth=max_depth
        )
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state

    def learn(
            self,
            data: pd.DataFrame,
            multicore: int | None = 1,
            verbose: bool = False
    ) -> 'JPTForest':
        '''Train ``n_estimators`` member trees on bootstrap resamples.

        :param data: the training data.
        :param multicore: number of worker processes for parallel member
            training (``1``: serial, ``None``: all cores).
        :param verbose: log progress.
        '''
        self._init_variables(data)
        rng = np.random.RandomState(self.random_state)
        resamples = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.randint(0, len(data), len(data))
                resamples.append(data.iloc[idx].reset_index(drop=True))
            else:
                resamples.append(data)
        pool_class = DummyPool if multicore == 1 else Pool
        # Pool workers are daemonic and may not fork their own children, so
        # parallel member training forces the inner C4.5 to run serially.
        inner_multicore = None if multicore == 1 else 0
        with pool_class(processes=multicore) as pool:
            self.members = list(pool.map(
                _fit_member,
                [
                    (self._blueprint(), resample, inner_multicore)
                    for resample in resamples
                ]
            ))
        self.weights = [1. / self.n_estimators] * self.n_estimators
        if verbose:
            logger.info(
                'JPTForest: trained %d members, %s leaves total.' % (
                    len(self.members),
                    sum(len(m.leaves) for m in self.members)
                )
            )
        return self

    def _hyperparams_to_json(self) -> Dict[str, Any]:
        return {
            **super()._hyperparams_to_json(),
            'n_estimators': self.n_estimators,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> 'JPTForest':
        forest = cls(
            n_estimators=data['n_estimators'],
            targets=data['targets'],
            min_samples_leaf=data['min_samples_leaf'],
            min_impurity_improvement=data['min_impurity_improvement'],
            max_depth=data['max_depth'],
            bootstrap=data['bootstrap'],
            random_state=data['random_state']
        )
        cls._restore_members(forest, data)
        return forest


# ----------------------------------------------------------------------------------------------------------------------

class JPTLikelihoodBoost(MixtureJPT):
    '''Generative boosting of the joint likelihood by greedy additive
    mixtures (approximate Frank--Wolfe in density space).

    Starting from a single JPT, each round :math:`t` forms

    .. math:: P_t = (1 - \\alpha_t) P_{t-1} + \\alpha_t C_t

    where the new component :math:`C_t` is a JPT trained on a resample
    that up-weights the samples the current mixture explains worst. The
    weights :math:`w_i \\propto \\exp(\\tau \\cdot \\mathrm{nll}_i)`
    (standardized, temperature :math:`\\tau`) are a weighted-log-likelihood
    *surrogate* of the exact Frank--Wolfe linear oracle
    :math:`w_i \\propto 1/P_{t-1}(z_i)`, so the procedure is approximate
    Frank--Wolfe ascent of the empirical joint log-likelihood. The step
    size :math:`\\alpha_t` is found by an exact concave line search.
    Because every component is normalized, the mixture remains a valid
    joint at all times --- no partition function is involved.

    Rounds that yield no improvement are skipped; training stops early
    after ``patience`` consecutive non-improving rounds.
    '''

    def __init__(
            self,
            variables: List[Variable] | None = None,
            n_rounds: int = 8,
            temperature: float = 1.,
            targets: List[str | Variable] | None = None,
            min_samples_leaf: float | int = .05,
            min_impurity_improvement: float | None = None,
            max_depth: int | None = None,
            patience: int = 2,
            random_state: int | None = None
    ) -> None:
        '''
        :param n_rounds: maximal number of boosting rounds (components
            beyond the initial tree).
        :param temperature: reweighting temperature :math:`\\tau` applied
            to the standardized per-sample negative log-likelihoods.
        :param patience: stop after this many consecutive rounds without
            improvement of the training log-likelihood.
        :param random_state: seed for the reweighted resampling.
        '''
        super().__init__(
            variables=variables,
            targets=targets,
            min_samples_leaf=min_samples_leaf,
            min_impurity_improvement=min_impurity_improvement,
            max_depth=max_depth
        )
        self.n_rounds = n_rounds
        self.temperature = temperature
        self.patience = patience
        self.random_state = random_state
        self.history: List[float] = []

    # ------------------------------------------------------------------------------------------------------------------

    def _density(self, member: JPT, data: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(member.likelihood(data)), _EPS, None)

    @staticmethod
    def _line_search(
            mix: np.ndarray,
            component: np.ndarray,
            tol: float = 1e-4
    ) -> Tuple[float, float]:
        '''Maximize the concave :math:`g(\\alpha) = \\frac1n \\sum_i \\log\\big((1-\\alpha)\\,p_i + \\alpha\\, c_i\\big)`
        over :math:`\\alpha \\in [0, 1]` by golden-section search.

        :return: the maximizing step size and the attained value.
        '''

        def g(alpha: float) -> float:
            return float(np.mean(np.log(
                np.clip((1 - alpha) * mix + alpha * component, _EPS, None)
            )))

        invphi = (np.sqrt(5) - 1) / 2
        lo, hi = 0., 1.
        x1 = hi - invphi * (hi - lo)
        x2 = lo + invphi * (hi - lo)
        g1, g2 = g(x1), g(x2)
        while hi - lo > tol:
            if g1 >= g2:
                hi, x2, g2 = x2, x1, g1
                x1 = hi - invphi * (hi - lo)
                g1 = g(x1)
            else:
                lo, x1, g1 = x1, x2, g2
                x2 = lo + invphi * (hi - lo)
                g2 = g(x2)
        alpha = .5 * (lo + hi)
        return alpha, g(alpha)

    def learn(
            self,
            data: pd.DataFrame,
            verbose: bool = False
    ) -> 'JPTLikelihoodBoost':
        '''Greedily grow the additive mixture on ``data``.

        :param data: the training data.
        :param verbose: log per-round training log-likelihood.
        '''
        self._init_variables(data)
        rng = np.random.RandomState(self.random_state)
        first_component = self._blueprint().learn(data.copy())
        self.members = [first_component]
        self.weights = [1.]
        mix = self._density(first_component, data)
        self.history = [float(np.mean(np.log(mix)))]
        stale = 0
        for t in range(self.n_rounds):
            nll = -np.log(mix)
            z = (nll - nll.mean()) / (nll.std() + _EPS)
            w = np.exp(self.temperature * z)
            w = np.clip(w, 0, 10 * w.mean())
            w /= w.sum()
            idx = rng.choice(len(data), len(data), p=w)
            component = self._blueprint().learn(
                data.iloc[idx].reset_index(drop=True)
            )
            density = self._density(component, data)
            alpha, ll = self._line_search(mix, density)
            if ll <= self.history[-1] + 1e-9 or alpha <= 0:
                stale += 1
                if stale >= self.patience:
                    if verbose:
                        logger.info(
                            'JPTLikelihoodBoost: early stop after '
                            'round %d (no improvement).' % (t + 1)
                        )
                    break
                continue
            stale = 0
            self.weights = [w_ * (1 - alpha) for w_ in self.weights]
            self.weights.append(alpha)
            self.members.append(component)
            mix = (1 - alpha) * mix + alpha * density
            self.history.append(ll)
            if verbose:
                logger.info(
                    'JPTLikelihoodBoost: round %d, alpha=%.3f, '
                    'train LL=%.4f' % (t + 1, alpha, ll)
                )
        return self

    def _hyperparams_to_json(self) -> Dict[str, Any]:
        return {
            **super()._hyperparams_to_json(),
            'n_rounds': self.n_rounds,
            'temperature': self.temperature,
            'patience': self.patience,
            'random_state': self.random_state,
            'history': list(self.history),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> 'JPTLikelihoodBoost':
        boost = cls(
            n_rounds=data['n_rounds'],
            temperature=data['temperature'],
            targets=data['targets'],
            min_samples_leaf=data['min_samples_leaf'],
            min_impurity_improvement=data['min_impurity_improvement'],
            max_depth=data['max_depth'],
            patience=data['patience'],
            random_state=data['random_state']
        )
        boost.history = list(data.get('history', []))
        cls._restore_members(boost, data)
        return boost


# ----------------------------------------------------------------------------------------------------------------------

class JPTBoost:
    '''Discriminative gradient boosting with JPTs as base learners
    (squared-error regression).

    Models the conditional expectation by an additive expansion

    .. math:: F_T(x) = F_0 + \\nu \\sum_{t=1}^{T} h_t(x)

    where :math:`F_0` is the target mean and each :math:`h_t` is a weak
    JPT fitted to the current residuals :math:`y_i - F_{t-1}(x_i)`.
    Prediction routes a feature row through each member's decision splits
    and reads the residual expectation at the resulting leaf (or
    prior-weighted leaves).

    .. warning::
        This ensemble is **discriminative**: it models
        :math:`E[y \\mid x]` only and gives up JPT's generative joint ---
        no ``infer``/``posterior``/``sample`` over arbitrary variables.
        That is the deliberate trade additive boosting makes: tilting a
        *joint* multiplicatively would require an intractable partition
        function over the whole data space, while conditioning collapses
        it to a per-row normalization. Use :class:`JPTForest` or
        :class:`JPTLikelihoodBoost` if you need the joint.

    Classification (softmax-gradient boosting) is future work.
    '''

    logger = logger

    def __init__(
            self,
            target: str,
            n_rounds: int = 40,
            learning_rate: float = .1,
            min_samples_leaf: float | int = .05,
            max_depth: int | None = None
    ) -> None:
        '''
        :param target: name of the (numeric) target column.
        :param n_rounds: number of boosting rounds.
        :param learning_rate: shrinkage :math:`\\nu` applied to every
            round's correction.
        :param min_samples_leaf: member-tree leaf size; large values give
            the *weak* learners boosting expects.
        :param max_depth: member-tree depth cap.
        '''
        self.target = target
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.f0: float = 0.
        self.members: List[JPT] = []
        self.feature_names: List[str] | None = None

    def _predict_member(
            self,
            member: JPT,
            data: pd.DataFrame
    ) -> np.ndarray:
        '''Leaf-routing prediction of the member's target expectation.'''
        predictions = np.empty(len(data))
        for i, (_, row) in enumerate(data.iterrows()):
            evidence = {name: row[name] for name in self.feature_names}
            leaves = list(member.apply(evidence))
            if not leaves:
                predictions[i] = 0.
                continue
            priors = np.asarray([leaf.prior for leaf in leaves])
            total = priors.sum()
            priors = (
                priors / total if total > 0
                else np.full(len(leaves), 1. / len(leaves))
            )
            predictions[i] = sum(
                prior * leaf.distributions[self.target].expectation()
                for prior, leaf in zip(priors, leaves)
            )
        return predictions

    def learn(
            self,
            data: pd.DataFrame,
            verbose: bool = False
    ) -> 'JPTBoost':
        '''Fit the boosted ensemble on ``data``.

        :param data: training data containing the target column and the
            feature columns.
        :param verbose: log per-round training MSE.
        '''
        if self.target not in data.columns:
            raise ValueError(
                'Target column %s not in data.' % repr(self.target)
            )
        self.feature_names = [c for c in data.columns if c != self.target]
        y = data[self.target].astype(float).to_numpy()
        self.f0 = float(y.mean())
        f = np.full(len(data), self.f0)
        features = data[self.feature_names]
        self.members = []
        for t in range(self.n_rounds):
            stage = features.copy()
            stage[self.target] = y - f
            variables = infer_from_dataframe(
                stage,
                scale_numeric_types=False
            )
            member = JPT(
                variables=variables,
                targets=[self.target],
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth
            ).learn(stage)
            self.members.append(member)
            f = f + self.learning_rate * self._predict_member(member, features)
            if verbose:
                logger.info(
                    'JPTBoost: round %d, train MSE=%.6f' % (
                        t + 1, float(np.mean((y - f) ** 2))
                    )
                )
        return self

    def fit(self, data: pd.DataFrame) -> 'JPTBoost':
        return self.learn(data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        '''Predict :math:`E[y \\mid x]` for every row of ``data``.'''
        if not self.members:
            raise RuntimeError('JPTBoost is not fitted yet.')
        features = data[self.feature_names]
        f = np.full(len(data), self.f0)
        for member in self.members:
            f = f + self.learning_rate * self._predict_member(member, features)
        return f

    # ------------------------------------------------------------------------------------------------------------------
    # Serialization

    def to_json(self) -> Dict[str, Any]:
        return {
            'type': type(self).__name__,
            'target': self.target,
            'n_rounds': self.n_rounds,
            'learning_rate': self.learning_rate,
            'min_samples_leaf': self.min_samples_leaf,
            'max_depth': self.max_depth,
            'f0': self.f0,
            'feature_names': self.feature_names,
            'members': [m.to_json() for m in self.members],
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'JPTBoost':
        boost = JPTBoost(
            target=data['target'],
            n_rounds=data['n_rounds'],
            learning_rate=data['learning_rate'],
            min_samples_leaf=data['min_samples_leaf'],
            max_depth=data['max_depth']
        )
        boost.f0 = data['f0']
        boost.feature_names = data['feature_names']
        boost.members = [JPT.from_json(d) for d in data['members']]
        return boost

    def __getstate__(self):
        return self.to_json()

    def __setstate__(self, state):
        self.__dict__ = JPTBoost.from_json(state).__dict__

    def __eq__(self, other) -> bool:
        return all((
            type(self) is type(other),
            self.target == other.target,
            self.f0 == other.f0,
            self.learning_rate == other.learning_rate,
            self.feature_names == other.feature_names,
            self.members == other.members,
        ))
