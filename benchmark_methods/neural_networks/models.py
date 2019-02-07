from typing import List, Any

import tensorflow as tf
from tensorflow.python.keras import layers, backend


class LayerNorm(layers.Layer):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-6
        self.scale = None
        self.bias = None

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(int(input_shape[-1]),),
                                     trainable=True,
                                     initializer=tf.ones_initializer,
                                     name='{}_scale'.format(self.name))
        self.bias = self.add_weight(shape=(int(input_shape[-1]),),
                                    trainable=True,
                                    initializer=tf.zeros_initializer,
                                    name='{}_bias'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        mean = backend.mean(x, axis=-1, keepdims=True)
        variance = backend.mean(backend.square(x - mean), axis=-1, keepdims=True)
        std = backend.sqrt(variance + self.epsilon)
        x = (x - mean) / std
        return x * self.scale + self.bias
        # return x

    def compute_output_shape(self, input_shape):
        return input_shape


class NAFNetwork(object):

    def __init__(self, layer_sizes: List[int], layer_activations: List[Any], state_shape: tuple, action_shape: tuple,
                 shared_layers: int):
        def layer_with_layer_norm(x, layer_index, branch_name):
            layer_name = f'{branch_name}_{layer_index}'
            x = layers.Dense(units=layer_sizes[layer_index], use_bias=False, name=f'{layer_name}_W')(x)
            x = LayerNorm(name=layer_name)(x)
            x = layers.Activation(layer_activations[layer_index])(x)
            return x

        def q_prediction(l_mu_v_a):
            l_var = l_mu_v_a[0]
            mu_var = l_mu_v_a[1]
            v_var = l_mu_v_a[2]
            a_var = l_mu_v_a[3]
            matrix = tf.reshape(l_var, (-1, action_shape[0], action_shape[0]))
            l_matrix = tf.matrix_band_part(matrix, -1, 0) - \
                       tf.matrix_diag(tf.matrix_diag_part(matrix)) + \
                       tf.matrix_diag(tf.exp(tf.matrix_diag_part(matrix)))
            # calc P
            p_matrix = tf.matmul(l_matrix, tf.matrix_transpose(l_matrix))

            # calc A
            delta_var = a_var - mu_var
            delta_mat_var = tf.reshape(delta_var, (-1, action_shape[0], 1))
            p_delta_var = tf.squeeze(tf.matmul(p_matrix, delta_mat_var), [2])
            adv = -0.5 * tf.reduce_sum(delta_var * p_delta_var, 1)
            return adv + v_var

        state = tf.keras.Input(shape=state_shape, name='observation_input')
        action = tf.keras.Input(shape=action_shape, name='action_input')
        x = state
        idx = 0
        while idx < shared_layers:
            x = layer_with_layer_norm(x, layer_index=idx, branch_name='shared')
            idx += 1

        # L branch
        xl = x
        idx_l = idx
        while idx_l < len(layer_sizes):
            xl = layer_with_layer_norm(xl, idx_l, 'L_branch')
            idx_l += 1

        xl = layers.Dense(units=action_shape[0] ** 2, use_bias=True, activation=None, name='L_branch_final')(xl)

        # V branch
        xv = x
        idx_v = idx
        while idx_v < len(layer_sizes):
            xv = layer_with_layer_norm(xv, idx_v, 'V_branch')
            idx_v += 1
        xv = layers.Dense(units=1, use_bias=True, activation=None, name='V_branch_final')(xv)

        # mu branch
        xm = x
        idx_m = idx
        while idx_m < len(layer_sizes):
            xm = layer_with_layer_norm(xm, idx_m, 'mu_branch')
            idx_m += 1
        xm = layers.Dense(units=action_shape[0], use_bias=True, activation='tanh', name='mu_branch_final')(xm)

        q_pred = layers.Lambda(q_prediction)([xl, xm, xv, action])

        self.model = tf.keras.Model(inputs=[state, action], outputs=[q_pred, xm, xv])
        # from keras.utils import plot_model
        # plot_model(self.model, to_file='NAF_model.png')

    def __call__(self, state_action, *args, **kwargs):
        return self.model(inputs=state_action, *args, **kwargs)
