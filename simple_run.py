import cor_control_benchmarks as cb
import tensorflow as tf
import tqdm

from benchmark_methods.common.experience_buffer import BaseExperienceBuffer, Experience
from benchmark_methods.common.exploration import EpsilonGreedy
from benchmark_methods.neural_networks.simple_policy import BasicNAFPolicy

# environment settings
from interpretation.policy_plot import policy_plot

SAMPLING_TIME = 0.02  # seconds between control decisions
EPISODE_SECONDS = 2.00  # seconds per episode
EPISODES = 1000

# experience replay settings
BUFFER_SIZE = int(1e5)  # number of experience samples in the experience buffer
BATCH_SIZE = 64  # The number of experiences used to average the gradients for one optimization step
TRAIN_INTERVAL = 4  # The number of environment interactions between optimization steps (note that in expectation,
# every experience will be seen BATCH_SIZE / TRAIN_INTERVAL times)

# NAF network architecture ( see https://arxiv.org/abs/1603.00748 )
HIDDEN_LAYER_SIZES = [32, 32]  # size of the hidden layers
HIDDEN_LAYER_ACTIVATIONS = ['elu', 'elu']  # the activation functions for the hidden layers (output activations are:
#  tanh for the policy action, linear for the V and L branches
NR_SHARED_LAYERS = 1  # The number of hidden layers that are shared before the policy, V and L branches split
TARGET_NETWORK_LOWPASS = 1e-3  # often called tau, the constant for updating the target network parameters

# exploration
EGREEDY_EPSILON = 0.2  # probability of taking a random action uniformly at random

# reinforcement learning target
GAMMA = 0.9  # the network is trained to predict Q(s,a) = r + GAMMA * Q-(s', a'*)  Where Q- is the target network, the
# parameters of which slowly track Q (see TARGET_NETWORK_LOWPASS above)

# network parameter optimization
LEARNING_RATE = 1e-4
OPTIMIZER = tf.train.AdamOptimizer(LEARNING_RATE)


def main():
    env = cb.MagmanBenchmark(sampling_time=SAMPLING_TIME, max_seconds=EPISODE_SECONDS,
                             reward_type=cb.RewardType.QUADRATIC, magnets=4, do_not_normalize=False)
    # The control benchmarks use normalized states and actions, which the rest of the code relies on (for instance by
    # using tanh outputs for the controller, which ensures control decisions are always between -1 and 1. If you set
    # do_not_normalize to True, the states and actions will be like those of the real magman, but be sure to compensate
    # in the rest of the code.

    experience_buffer = BaseExperienceBuffer(BUFFER_SIZE, normalize_reward_on_first_batch_sample=False)

    diagnostics = cb.Diagnostics(benchmark=env, log=cb.LogType.BEST_AND_LAST_TRAJECTORIES)

    sess = tf.Session()
    policy = BasicNAFPolicy(
        environment=env, experience_buffer=experience_buffer, tensorflow_session=sess,
        layer_sizes=HIDDEN_LAYER_SIZES, layer_activations=HIDDEN_LAYER_ACTIVATIONS, shared_layers=NR_SHARED_LAYERS,
        gamma=GAMMA, tau=TARGET_NETWORK_LOWPASS, optimizer=OPTIMIZER, batch_size=BATCH_SIZE)

    exploration_function = EpsilonGreedy(epsilon=EGREEDY_EPSILON, repeat=0)
    # it is possible to reduce epsilon over time by updating exploration_function.epsilon .
    # Repeat can be used to repeat the exploratory actions for more than one step, which can help exploration in
    # domains where the environment dynamics are slow relative to the sampling time. To make things work well, it would
    # probably be a good idea to implement a better form of exploration. Examples include:
    # Ornstein-Uhlenbeck noise, as in https://arxiv.org/abs/1509.02971 and
    # parameter noise exploration https://arxiv.org/abs/1706.01905

    step = 0

    with sess:
        progress_bar = tqdm.tqdm(range(EPISODES), unit='episode', desc=env.name)
        for _ in progress_bar:
            terminal = False
            state = env.reset()
            while not terminal:
                action = exploration_function(policy(state))
                next_state, reward, terminal, _ = env.step(action)

                saved_terminal = True if terminal and not env.max_steps_passed else False  # restore markov property
                # by not counting tne last transition in episodes ended due to max steps as terminal
                experience_buffer.add_experience(Experience(
                    state=state, action=action, next_state=next_state, reward=reward, terminal=saved_terminal
                ))

                state = next_state
                step += 1

                if step % TRAIN_INTERVAL == 0:
                    policy.train()

            progress_bar.set_postfix({
                'best reward sum': diagnostics.best_reward_sum,
                'last_reward_sum': f'{diagnostics.last_reward_sum:4.f}'
            })
            # if diagnostics.best_episode:
            #     policy.save_params_to_dir(save_dir)

        diagnostics.plot_best_trajectory(state=True, action=True, rewards=False)
        diagnostics.plot_reward_sum_per_episode()
        policy_plot(env, policy)
        input("press enter to close the plots")


if __name__ == '__main__':
    main()
