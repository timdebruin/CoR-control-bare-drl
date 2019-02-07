from typing import Callable

import cor_control_benchmarks as cb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

SAMPLES_PER_DIMENSION = 100
STATE_DIM = 2  # these plots only work for 2d state spaces
TICK_LOCS = [-1., 0., 1.]


def policy_plot(env: cb.control_benchmark.ControlBenchmark, policy: Callable):
    actions_per_dim = np.linspace(-1, 1, SAMPLES_PER_DIMENSION)
    meshed = np.meshgrid(*tuple([actions_per_dim for _ in range(STATE_DIM)]))
    state_space = np.reshape(np.stack(meshed, axis=-1), (-1, STATE_DIM))
    xx, yy = meshed
    policy_actions = []
    print('sampling policy')
    for i in range(SAMPLES_PER_DIMENSION ** STATE_DIM):
        policy_actions.append(policy(state_space[i]))
    policy_actions = np.array(policy_actions).reshape((SAMPLES_PER_DIMENSION, SAMPLES_PER_DIMENSION, -1))
    actions = policy_actions.shape[2]

    print('plotting result')
    fig, axes = plt.subplots(nrows=1, ncols=actions, sharey='all', sharex='all', squeeze=False)
    levels = np.arange(-1.05, 1.05, 0.03)
    norm = cm.colors.Normalize(vmax=1., vmin=-1.)
    if all([x >= 0 for x in env.denormalize_action(-np.ones(env.action_shape))]):  # all positive actions
        cmap = cm.hot
    else:
        cmap = cm.seismic
    for action in range(actions):
        ax: plt.Axes = axes[0][action]
        cmapset = ax.contourf(xx, yy, policy_actions[..., action], levels, norm=norm,
                              cmap=cm.get_cmap(cmap, len(levels) - 1))

    true_states = np.array([env.denormalize_state(np.array([x, x])) for x in TICK_LOCS])
    true_actions = np.array([env.denormalize_action(np.array([x for _ in range(actions)])) for x in TICK_LOCS])
    cbar = fig.colorbar(cmapset, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(true_actions[:, -1])
    cbar.ax.set_ylabel(env.action_names[-1])
    plt.xticks(TICK_LOCS, [f'{s:.4}' for s in true_states[:, 0]])
    plt.yticks(TICK_LOCS, [f'{s:.4}' for s in true_states[:, 1]])
    plt.xlabel(env.state_names[0])
    plt.ylabel(env.state_names[1])
    plt.title(env.name)
    plt.show()
