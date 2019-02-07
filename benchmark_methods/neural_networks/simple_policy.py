from typing import List

import tensorflow as tf
import numpy as np
import trfl

from cor_control_benchmarks.control_benchmark import ControlBenchmark

from benchmark_methods.common.experience_buffer import BaseExperienceBuffer
from benchmark_methods.common.policy import AbstractPolicy
from benchmark_methods.neural_networks.models import NAFNetwork


class BasicNAFPolicy(AbstractPolicy):
    """ A stripped down implementation that enables standard training and querying of a NAF policy"""

    def __init__(self, environment: ControlBenchmark, experience_buffer: BaseExperienceBuffer,
                 tensorflow_session: tf.Session,
                 gamma: float,
                 layer_sizes: List[int],
                 layer_activations: List[str],
                 shared_layers: int,
                 tau: float,
                 optimizer: tf.train.Optimizer,
                 batch_size: int,
                 ) -> None:
        super().__init__(environment=environment, experience_buffer=experience_buffer)
        self.shared_layers = shared_layers
        self.tensorflow_session = tensorflow_session
        self.batch_size = batch_size

        self.Q = NAFNetwork(layer_sizes=layer_sizes, layer_activations=layer_activations, shared_layers=shared_layers,
                            state_shape=environment.state_shape, action_shape=environment.action_shape)
        self.Q_lowpass = NAFNetwork(layer_sizes=layer_sizes, layer_activations=layer_activations,
                                    shared_layers=shared_layers, state_shape=environment.state_shape,
                                    action_shape=environment.action_shape)

        self.Q_lowpass.model.set_weights(self.Q.model.get_weights())

        self.observation_input = tf.keras.Input(shape=self.environment.state_shape, name='state')
        self.next_observation_input = tf.keras.Input(shape=self.environment.state_shape, name='next_state')
        self.action_input = tf.keras.Input(shape=self.environment.action_shape, name='action_placeholder')
        self.reward_input = tf.keras.Input(shape=(), name='reward')
        self.terminal_input = tf.keras.Input(shape=(), name='terminal')

        self.p_continue = gamma * (1 - self.terminal_input)

        self.frozen_parameter_update_op = trfl.periodic_target_update(
            target_variables=self.Q_lowpass.model.variables,
            source_variables=self.Q.model.variables,
            update_period=1,
            tau=tau
        )

        self.q_values_policy, self.mu_policy, _ = self.Q(state_action=[self.observation_input, self.action_input])
        _, _, self.vt_lowpass = self.Q_lowpass(state_action=[self.next_observation_input, self.action_input])
        # action is not actually used here to calculate the value

        self.target = self.reward_input + self.p_continue * self.vt_lowpass
        rl_loss = tf.reduce_mean(0.5 * (self.q_values_policy - self.target) ** 2)
        self.train_op = optimizer.minimize(rl_loss)

        self._initialize_tf_variables()

    def train(self, **kwargs) -> None:
        """Performs a single training step"""
        tf.keras.backend.set_learning_phase(1)

        batch = self.experience_buffer.sample_batch(batch_size=self.batch_size)
        feed_dict = {
            self.observation_input: batch.experiences.state,
            self.next_observation_input: batch.experiences.next_state,
            self.action_input: batch.experiences.action.reshape(
                -1, self.environment.action_shape[0]),
            self.reward_input: batch.experiences.reward,
            self.terminal_input: batch.experiences.terminal
        }

        self.tensorflow_session.run(fetches=[self.train_op, self.frozen_parameter_update_op], feed_dict=feed_dict)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Get the policy action for a given state"""
        super().__call__(*args, **kwargs)
        tf.keras.backend.set_learning_phase(0)
        action = self.tensorflow_session.run(self.mu_policy,
                                             feed_dict={self.observation_input: self._state})
        return np.squeeze(action)

    def save_params_to_dir(self, save_dir):
        self.Q.model.save_weights(save_dir)

    def load_params_form_dir(self, load_dir):
        for model in (self.Q.model, self.Q_lowpass.model):
            model.load_weights(load_dir)

    def _initialize_tf_variables(self):
        self.tensorflow_session.run(tf.global_variables_initializer())
