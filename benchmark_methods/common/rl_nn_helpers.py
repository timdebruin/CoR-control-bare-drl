import tensorflow as tf
from typing import List


def periodic_target_update(
        target_variables: List[tf.Variable],
        source_variables: List[tf.Variable],
        update_period: int,
        tau: float) -> tf.Operation:

    if update_period >= 1:
        counter = tf.get_variable(
            "counter",
            shape=[],
            dtype=tf.int64,
            trainable=False,
            initializer=tf.constant_initializer(update_period, dtype=tf.int64))

        ops = []
        for src, t in zip(source_variables, target_variables):
            ops.append(tf.assign(t, tau * src + (1.0 - tau) * t))

        def update_op():
            with tf.control_dependencies(ops):
                # Done the deed, resets the counter.
                return counter.assign(1)

        return tf.cond(
            tf.equal(counter, update_period), update_op, lambda: counter.assign_add(1))
    else:
        return tf.no_op()



