from typing import Optional
import numpy as np

from cor_control_benchmarks.control_benchmark import ControlBenchmark

from benchmark_methods.common.experience_buffer import BaseExperienceBuffer


class DiscreteActions(object):
    """Discretize the action space"""

    def __init__(self, action_shape: tuple, actions_per_dimension: int):
        assert len(action_shape) == 1
        dimensionality = action_shape[0]
        actions_per_dim = np.linspace(-1, 1, actions_per_dimension)
        meshed = np.meshgrid(*tuple([actions_per_dim for _ in range(dimensionality)]))
        self.actions = np.reshape(np.stack(meshed, axis=-1), (-1, dimensionality))

    def __call__(self, *args, **kwargs):
        chosen_action = args[0]
        i = int(chosen_action)
        assert np.abs(i - chosen_action) < 1e-3, f'{chosen_action} is not an int'
        return np.copy(self.actions[i])

    def __len__(self):
        return self.actions.shape[0]


class AbstractPolicy(object):
    """Base class with some shared policy methods to ensure consistency."""

    def __init__(self,
                 environment: Optional[ControlBenchmark], experience_buffer: Optional[BaseExperienceBuffer]) -> None:
        """Initialize the environment by giving it some common components needed for learning
        :param environment: optional benchmark, used for online learning strategies
        :param experience_buffer: optional experience buffer, used for offline learning
        """
        self.experience_buffer = experience_buffer
        self.environment = environment
        self._state: Optional[np.ndarray] = None  # the state for which to calculate the policy action

    def __call__(self, *args, **kwargs):
        """Get the policy action for the given state"""
        assert len(args) == 1, 'Policy expects one unnamed argument: the state.'
        assert isinstance(args[0], np.ndarray), 'Policy expects the state as an numpy ndarray'
        self._state = args[0][None]

    def train(self, **kwargs):
        """Train the policy"""
        raise NotImplementedError

    def finished_episode_with_score(self, last_reward_sum):
        """Tell the policy that the last episode was finished and what the score was.
        Used for episode based exploration."""
        pass
