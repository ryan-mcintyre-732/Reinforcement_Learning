from DeepQNetwork import Agent
import numpy as np
import gym
from utils import plot_learning_curve
import tensorflow as tf

LOAD_MODEL = True

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    learning_rate = 0.001
    n_games = 500
    agent = Agent(
        gamma=0.99,
        epsilon_initial=0.01 if LOAD_MODEL else 1.0,
        learning_rate=learning_rate,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.n,
        mem_size=1000000,
        batch_size=64
    )
    if LOAD_MODEL:
        agent.load_model()
    scores = []
    epsilon_history = []
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        epsilon_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print(
            'episode ', i,
            'score %2f' % score,
            'average_score %2f' % avg_score,
            'epsilon %2f' % agent.epsilon
        )

    filename = '../lunar-lander_tf2.png'
    agent.save_model()
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, epsilon_history, filename)
