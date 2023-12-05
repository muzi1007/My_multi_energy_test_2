import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import sys
sys.path.append("..")
from env_v2 import *

if __name__ == '__main__':
    env = env_v2()
    agent = Agent(input_dims=env.state_space, env=env, n_actions=env.action_space)

    n_episode = 3000
    filename = 'plt-v1.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    n_steps = 0

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    for i in range(n_episode):
        observation = env.reset()[0]
        done = False
        score = 0
        #n_steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)[0:4]
            n_steps += 1
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

            if n_steps % 1000 == 0:
                done = True

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            '''
            if not load_checkpoint:
                agent.save_models()
            '''
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'time_steps', n_steps)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

