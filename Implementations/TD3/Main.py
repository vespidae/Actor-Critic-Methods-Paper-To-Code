import gym
import numpy as np
from Utils import plot_learning_curve
from Agent import Agent


if __name__ == '__main__':
    n_games = 1500
    actor_lr = 1e-3
    critic_lr = 1e-3
    soft_target_update = 5e-3
    discount_factor = 0.99
    hlOne = 400
    hlTwo = 300
    buffer_size = int(3e4)
    minibatch_size = 64

    env = gym.make('BipedalWalker-v3')

    input_space = env.observation_space.shape
    action_space = env.action_space.shape[0]

    agent = Agent(alpha=actor_lr, beta=critic_lr, input_dims=input_space, n_actions=action_space, env=env,
                  tau=soft_target_update, layer1_size=hlOne, layer2_size=hlTwo, buffer_size=buffer_size,
                  batch_size=minibatch_size)

    filename = 'Biped-2D_' + str(n_games) + '.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        null_obv = env.reset()
        score = 0
        done = False

        while not done:
            # choice = agent.choose_action(null_obv)
            # action = choice.argmax(0)
            action = agent.choose_action(null_obv)
            prime_obv, reward, done, info = env.step(action)
            # env.render()
            agent.remember(null_obv, action, reward, prime_obv, done)
            score += reward
            agent.learn()
            null_obv = prime_obv

        # env.close()

        score_history.append(score)
        running_avg_score = np.mean(score_history[-100:])

        if running_avg_score > best_score:
            best_score = running_avg_score
            print("New best score!\nScore:\t{}".format(best_score))
            agent.save_models()

        print("Episode: {}\t\tScore: {}\t\tAverage Score: {}".format(i, score, running_avg_score))

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(score_history, x, figure_file)
